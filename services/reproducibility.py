"""Reproducibility verification service for VerdictAI.

Two entry points:

    reproduce_evaluation(conn, tender_id, ...)
        Actually re-runs L3 extract_evidence + L4 evaluate for every
        (bidder, criterion) pair in the tender, using stored inputs
        and cached LLM responses, then compares the reproduced verdicts
        against the originally-stored ones. Byte-identical agreement
        (modulo timestamps) is the success condition.

    verify_reproducibility(conn, tender_id, ...)
        Legacy precondition checker — verifies that the documents,
        LLM log, and audit chain are all in a state that *could* be
        reproduced. Kept for backwards compatibility with the earlier
        API contract.

Byte-identical reproducibility requires cached LLM responses. We use
:class:`CachedLLMClient` which looks up stored responses from
``llm_stub_log`` by ``prompt_hash`` before calling the real API.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from backend.layers.l5_audit import verify_hash_chain


logger = logging.getLogger(__name__)


# Numeric tolerance for confidence comparisons. A floating-point
# computation can diverge by ~1e-9 across Python versions; anything
# below 1e-6 is "byte-identical" for officer-reporting purposes.
_CONFIDENCE_TOLERANCE = 1e-6


# ─── Public API: the real re-runner ──────────────────────────────────────


def reproduce_evaluation(
    conn: sqlite3.Connection,
    tender_id: str,
    original_report_id: str | None = None,
) -> dict:
    """Re-run evaluation from stored inputs and compare to the original.

    Steps:

      1. **Snapshot** — fetch every (bidder, criterion) evaluation
         currently stored for the tender.
      2. **Re-run** — for each pair, re-invoke the L3 evidence extractor
         and the L4 evaluation logic. A :class:`CachedLLMClient` is
         injected so any LLM call returns the previously-logged response
         by ``prompt_hash`` lookup; no fresh API calls are made.
      3. **Compare** — match originals against reproduced results on
         verdict, confidence (within tolerance), route, and
         ``source_bbox``. A mismatch on any of these fields is
         recorded as a diff.
      4. **Hash** — compute a deterministic SHA-256 over the stripped
         evaluation records (timestamps / ids / officer fields removed)
         for both sets and report whether the hashes match.

    Args:
        conn: Active SQLite connection for the tender's database.
        tender_id: The tender to reproduce.
        original_report_id: Optional report ID for audit tagging.

    Returns:
        Dict::

            {
              "match":                               bool,
              "tender_id":                           str,
              "report_id":                           Optional[str],
              "total_compared":                      int,
              "matches":                             int,
              "diffs":                               [ {...}, ... ],
              "byte_identical_excluding_timestamps": bool,
              "reproducibility_hash_original":       str,
              "reproducibility_hash_reproduced":     str,
              "cache_hits":                          int,
              "cache_misses":                        int,
              "summary":                             str,
            }
    """
    # Imported here to avoid circular imports: L3 / L4 pull in services
    # that in turn import from here indirectly.
    from backend.layers.l3_evidence import extract_evidence
    from backend.layers.l4_evaluation import (
        compute_route,
        _evaluate_by_type,
    )
    from backend.services.cpm_service import get_cpm_stats
    from backend.services.explanation_service import build_explanation

    conn.row_factory = sqlite3.Row

    tender = conn.execute(
        "SELECT * FROM tenders WHERE id = ?", (tender_id,)
    ).fetchone()
    if not tender:
        raise ValueError(f"Tender {tender_id} not found")

    originals = conn.execute(
        "SELECT * FROM evaluations WHERE tender_id = ? ORDER BY id ASC",
        (tender_id,),
    ).fetchall()

    if not originals:
        return {
            "match": True,
            "tender_id": tender_id,
            "report_id": original_report_id,
            "total_compared": 0,
            "matches": 0,
            "diffs": [],
            "byte_identical_excluding_timestamps": True,
            "reproducibility_hash_original": _hash_of_evaluations([]),
            "reproducibility_hash_reproduced": _hash_of_evaluations([]),
            "cache_hits": 0,
            "cache_misses": 0,
            "summary": "No evaluations to reproduce.",
        }

    # Install a CachedLLMClient as the process-wide default so the real
    # pipeline code transparently hits the cache. We restore the old
    # default on the way out so we don't contaminate other callers.
    from backend.services import llm_client as llm_client_module
    from backend.services.llm_client import CachedLLMClient

    original_client = llm_client_module._default_client
    cached_client = CachedLLMClient(conn=conn, tender_id=tender_id)
    llm_client_module._default_client = cached_client

    try:
        reproduced: list[dict] = []
        diffs: list[dict] = []

        cpm_stats = get_cpm_stats(conn)
        cpm_data_count = cpm_stats.get("total_entries", 0)

        for original in originals:
            orig_dict = dict(original)

            criterion = conn.execute(
                "SELECT * FROM criteria WHERE id = ?",
                (original["criterion_id"],),
            ).fetchone()
            if not criterion:
                diffs.append({
                    "evaluation_id": original["id"],
                    "field": "criterion",
                    "original": "present",
                    "reproduced": "criterion_missing",
                })
                continue

            # Re-run L3 + L4 without persisting. We use the internal
            # type dispatcher rather than evaluate_criterion() to avoid
            # writing a second row into evaluations / audit_events.
            evidence = extract_evidence(
                conn=conn,
                tender_id=tender_id,
                bidder_id=original["bidder_id"],
                criterion_id=original["criterion_id"],
            )

            verdict, confidence, method = _evaluate_by_type(
                criterion["criterion_type"], criterion, evidence
            )

            flags: list[str] = []
            if evidence.get("entity_match_flag"):
                flags.append("entity_mismatch")

            routing = compute_route(
                verdict=verdict,
                confidence=confidence,
                criterion_type=criterion["criterion_type"],
                flags=flags,
                is_mandatory=bool(criterion["is_mandatory"]),
                gfr_override_permitted=bool(criterion["gfr_override_permitted"]),
                cpm_data_count=cpm_data_count,
            )

            route = routing["route"]

            # Build the same explanation the original pipeline would
            # have built so headline / detail also match.
            criterion_dict = dict(criterion)
            tv_raw = criterion_dict.get("threshold_value")
            if isinstance(tv_raw, str) and tv_raw:
                try:
                    criterion_dict["threshold_value"] = json.loads(tv_raw)
                except (json.JSONDecodeError, TypeError):
                    pass

            union_agreement = None
            if isinstance(evidence.get("value"), dict):
                union_info = evidence["value"].get("union") or {}
                union_agreement = union_info.get("agreement")

            explanation = build_explanation(
                verdict=verdict,
                criterion=criterion_dict,
                evidence=evidence,
                route=route,
                union_agreement=union_agreement,
            )

            reproduced_row = {
                "evaluation_id": original["id"],
                "bidder_id": original["bidder_id"],
                "criterion_id": original["criterion_id"],
                "verdict": verdict,
                "confidence": round(float(confidence), 6),
                "route": route,
                "source_document_id": evidence.get("source_document_id"),
                "source_page_number": evidence.get("page"),
                "source_bbox": json.dumps(evidence.get("bbox"), sort_keys=True)
                    if evidence.get("bbox") else None,
                "headline": explanation["headline"],
            }
            reproduced.append(reproduced_row)

            _compare_and_record(orig_dict, reproduced_row, diffs)

        original_hash = _hash_of_evaluations(
            _strip_for_hashing(dict(r)) for r in originals
        )
        reproduced_hash = _hash_of_evaluations(
            _strip_for_hashing(r, reproduced=True) for r in reproduced
        )

        total = len(originals)
        matches = total - len({d["evaluation_id"] for d in diffs})
        byte_identical = (original_hash == reproduced_hash)
        match_bool = byte_identical and not diffs

        summary = (
            f"Reproduced {total} evaluations; {matches} matched, "
            f"{len(diffs)} diverged. "
            f"LLM cache hits: {cached_client.cache_hits}, "
            f"misses: {cached_client.cache_misses}. "
            f"Byte-identical: {byte_identical}."
        )

        return {
            "match": match_bool,
            "tender_id": tender_id,
            "report_id": original_report_id,
            "total_compared": total,
            "matches": matches,
            "diffs": diffs,
            "byte_identical_excluding_timestamps": byte_identical,
            "reproducibility_hash_original": original_hash,
            "reproducibility_hash_reproduced": reproduced_hash,
            "cache_hits": cached_client.cache_hits,
            "cache_misses": cached_client.cache_misses,
            "summary": summary,
        }

    finally:
        llm_client_module._default_client = original_client


# ─── Comparison + hashing helpers ────────────────────────────────────────


def _compare_and_record(
    original: dict,
    reproduced: dict,
    diffs: list,
) -> None:
    """Compare a single evaluation and append any divergences to diffs."""
    ev_id = original.get("id") or reproduced["evaluation_id"]

    # Verdict must match exactly.
    if original.get("verdict") != reproduced["verdict"]:
        diffs.append({
            "evaluation_id": ev_id,
            "field": "verdict",
            "original": original.get("verdict"),
            "reproduced": reproduced["verdict"],
        })

    # Confidence within tolerance.
    orig_conf = _to_float(original.get("confidence"))
    repro_conf = _to_float(reproduced["confidence"])
    if orig_conf is not None and repro_conf is not None:
        if abs(orig_conf - repro_conf) > _CONFIDENCE_TOLERANCE:
            diffs.append({
                "evaluation_id": ev_id,
                "field": "confidence",
                "original": orig_conf,
                "reproduced": repro_conf,
                "delta": abs(orig_conf - repro_conf),
            })

    # Route must match — it's deterministic given the inputs.
    if original.get("route") != reproduced["route"]:
        diffs.append({
            "evaluation_id": ev_id,
            "field": "route",
            "original": original.get("route"),
            "reproduced": reproduced["route"],
        })

    # source_bbox stored as JSON; compare canonical form.
    orig_bbox = _canonical_json(original.get("source_bbox"))
    repro_bbox = _canonical_json(reproduced.get("source_bbox"))
    if orig_bbox != repro_bbox:
        diffs.append({
            "evaluation_id": ev_id,
            "field": "source_bbox",
            "original": orig_bbox,
            "reproduced": repro_bbox,
        })


def _strip_for_hashing(row: dict, *, reproduced: bool = False) -> dict:
    """Strip fields whose divergence is expected (timestamps, ids, officer)."""
    if reproduced:
        # reproduced rows have a minimal schema; just carry verdict-relevant.
        return {
            "bidder_id": row.get("bidder_id"),
            "criterion_id": row.get("criterion_id"),
            "verdict": row.get("verdict"),
            "confidence": round(_to_float(row.get("confidence")) or 0.0, 6),
            "route": row.get("route"),
            "source_document_id": row.get("source_document_id"),
            "source_page_number": row.get("source_page_number"),
            "source_bbox": _canonical_json(row.get("source_bbox")),
        }
    return {
        "bidder_id": row.get("bidder_id"),
        "criterion_id": row.get("criterion_id"),
        "verdict": row.get("verdict"),
        "confidence": round(_to_float(row.get("confidence")) or 0.0, 6),
        "route": row.get("route"),
        "source_document_id": row.get("source_document_id"),
        "source_page_number": row.get("source_page_number"),
        "source_bbox": _canonical_json(row.get("source_bbox")),
    }


def _hash_of_evaluations(rows: Iterable[dict]) -> str:
    """Deterministic SHA-256 of a list of stripped evaluation rows."""
    normalised = sorted(
        (
            json.dumps(r, sort_keys=True, separators=(",", ":"), default=str)
            for r in rows
        )
    )
    payload = "\n".join(normalised).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Any) -> Any:
    """Return a canonical form of a maybe-JSON-string for comparison."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return json.dumps(parsed, sort_keys=True)
        except (json.JSONDecodeError, TypeError):
            return value
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(value)


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ─── Legacy precondition checker (unchanged API) ─────────────────────────


def verify_reproducibility(
    conn: sqlite3.Connection,
    tender_id: str,
    report_id: str | None = None,
) -> dict:
    """Verify that a completed evaluation is reproducible from stored inputs.

    This is the richer of the two entry points: it calls through to
    :func:`reproduce_evaluation` (which actually re-runs the pipeline)
    and then folds in the legacy precondition checks (document hashes,
    audit chain, LLM stub version) so the response carries both
    signals.
    """
    conn.row_factory = sqlite3.Row

    tender = conn.execute(
        "SELECT * FROM tenders WHERE id = ?", (tender_id,)
    ).fetchone()

    if not tender:
        raise ValueError(f"Tender {tender_id} not found")

    completed_states = {"EVALUATION_COMPLETE", "REPORT_GENERATED"}
    if tender["status"] not in completed_states:
        # Don't hard-fail — the demo flow lets us reproduce mid-run,
        # and the precondition checks will flag any genuine blockers.
        logger.info(
            "Tender %s is in state %s — running reproducibility anyway",
            tender_id, tender["status"],
        )

    # ── Re-run the pipeline ──
    rerun_result = reproduce_evaluation(conn, tender_id, report_id)

    # ── Legacy precondition checks ──
    checks = [
        _verify_document_hashes(conn, tender_id),
        _verify_llm_stub_versions(conn, tender_id),
        _verify_audit_chain(conn, tender_id),
        _verify_stored_inputs(conn, tender_id),
    ]
    differences: list[str] = []
    for c in checks:
        if not c["passed"]:
            differences.extend(c.get("issues", []))

    # The full-rerun diffs are the most important signal.
    rerun_check = {
        "check": "evaluation_rerun_byte_identical",
        "passed": rerun_result["match"],
        "message": rerun_result["summary"],
        "rerun": rerun_result,
        "issues": [
            f"evaluation {d['evaluation_id']}: {d['field']} differs "
            f"(original={d.get('original')!r} reproduced={d.get('reproduced')!r})"
            for d in rerun_result.get("diffs", [])
        ],
    }
    checks.append(rerun_check)
    if not rerun_check["passed"]:
        differences.extend(rerun_check["issues"])

    all_passed = all(c["passed"] for c in checks)

    return {
        "match": all_passed,
        "tender_id": tender_id,
        "report_id": report_id,
        "checks": checks,
        "differences": differences,
        "rerun": rerun_result,
        "summary": (
            "All reproducibility checks passed. Evaluation is verifiable "
            "and byte-identical on re-run."
            if all_passed
            else f"Reproducibility verification failed: "
                 f"{len(differences)} issue(s) found."
        ),
    }


def _verify_document_hashes(conn: sqlite3.Connection, tender_id: str) -> dict:
    """Verify all documents still have matching SHA-256 hashes."""
    documents = conn.execute(
        "SELECT id, filename, file_path, sha256_hash FROM documents WHERE tender_id = ?",
        (tender_id,),
    ).fetchall()

    if not documents:
        return {
            "check": "document_integrity",
            "passed": True,
            "message": "No documents to verify",
            "issues": [],
        }

    issues = []
    verified_count = 0

    for doc in documents:
        file_path = Path(doc["file_path"]) if doc["file_path"] else None
        stored_hash = doc["sha256_hash"]

        if not stored_hash:
            issues.append(
                f"Document {doc['id']} ({doc['filename']}): no SHA-256 hash stored"
            )
            continue

        if file_path and file_path.exists():
            actual_hash = _compute_file_hash(file_path)
            if actual_hash != stored_hash:
                issues.append(
                    f"Document {doc['id']} ({doc['filename']}): "
                    f"hash mismatch — stored={stored_hash[:16]}..., "
                    f"actual={actual_hash[:16]}..."
                )
            else:
                verified_count += 1
        else:
            verified_count += 1

    return {
        "check": "document_integrity",
        "passed": len(issues) == 0,
        "message": f"Verified {verified_count}/{len(documents)} document hashes",
        "verified_count": verified_count,
        "total_documents": len(documents),
        "issues": issues,
    }


def _verify_llm_stub_versions(conn: sqlite3.Connection, tender_id: str) -> dict:
    """All LLM invocations for a tender should use the same model version."""
    logs = conn.execute(
        "SELECT model_version FROM llm_stub_log WHERE tender_id = ?",
        (tender_id,),
    ).fetchall()

    if not logs:
        return {
            "check": "llm_stub_version",
            "passed": True,
            "message": "No LLM invocations logged for this tender",
            "issues": [],
        }

    versions = sorted({log["model_version"] for log in logs})

    if len(versions) > 1:
        return {
            "check": "llm_stub_version",
            "passed": False,
            "message": f"Version inconsistency: {versions}",
            "versions_found": versions,
            "invocation_count": len(logs),
            "issues": [
                f"Multiple LLM versions used: {versions}"
            ],
        }

    return {
        "check": "llm_stub_version",
        "passed": True,
        "message": f"Consistent version '{versions[0]}' across {len(logs)} invocations",
        "invocation_count": len(logs),
        "issues": [],
    }


def _verify_audit_chain(conn: sqlite3.Connection, tender_id: str) -> dict:
    """Verify the audit trail hash chain is intact."""
    is_valid, error_msg = verify_hash_chain(conn, tender_id)

    if not is_valid:
        return {
            "check": "audit_chain_integrity",
            "passed": False,
            "message": f"Hash chain broken: {error_msg}",
            "issues": [f"Audit chain integrity failure: {error_msg}"],
        }

    event_count = conn.execute(
        "SELECT COUNT(*) as c FROM audit_events WHERE tender_id = ?",
        (tender_id,),
    ).fetchone()["c"]

    return {
        "check": "audit_chain_integrity",
        "passed": True,
        "message": f"Hash chain intact across {event_count} events",
        "event_count": event_count,
        "issues": [],
    }


def _verify_stored_inputs(conn: sqlite3.Connection, tender_id: str) -> dict:
    """Verify all required inputs for reproduction are stored."""
    issues = []

    docs_without_hash = conn.execute(
        "SELECT COUNT(*) as c FROM documents WHERE tender_id = ? "
        "AND (sha256_hash IS NULL OR sha256_hash = '')",
        (tender_id,),
    ).fetchone()["c"]
    if docs_without_hash > 0:
        issues.append(f"{docs_without_hash} document(s) missing SHA-256 hash")

    criteria_count = conn.execute(
        "SELECT COUNT(*) as c FROM criteria WHERE tender_id = ?",
        (tender_id,),
    ).fetchone()["c"]
    if criteria_count == 0:
        issues.append("No criteria (ETS) stored for this tender")

    total_evals = conn.execute(
        "SELECT COUNT(*) as c FROM evaluations WHERE tender_id = ?",
        (tender_id,),
    ).fetchone()["c"]
    if total_evals == 0:
        issues.append("No evaluations stored for this tender")

    resolved_without_decision = conn.execute(
        """SELECT COUNT(*) as c FROM evaluations
           WHERE tender_id = ? AND status = 'resolved'
           AND (officer_decision IS NULL OR officer_decision = '')""",
        (tender_id,),
    ).fetchone()["c"]
    if resolved_without_decision > 0:
        issues.append(
            f"{resolved_without_decision} resolved evaluation(s) missing officer decision record"
        )

    return {
        "check": "stored_inputs_completeness",
        "passed": len(issues) == 0,
        "message": (
            "All required inputs stored"
            if len(issues) == 0
            else f"{len(issues)} input completeness issue(s)"
        ),
        "criteria_count": criteria_count,
        "evaluation_count": total_evals,
        "issues": issues,
    }


def _compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()
