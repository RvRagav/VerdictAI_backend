"""Report generation service for VerdictAI.

Produces an officer-grade, print-ready PDF evaluation report suitable
for signing and archival alongside the tender file. The PDF includes:

    1. Cover page         — department header, tender meta, officer,
                            report ID, date.
    2. Executive summary  — color-coded per-bidder status table.
    3. Per-bidder pages   — company details + debarment status + a
                            criterion-by-criterion matrix with verdict
                            badges, confidence, and source references.
    4. Officer overrides  — any resolved-by-override criterion gets a
                            highlighted banner with the structured
                            reason and second-officer confirmation.
    5. Audit trail summary — event count, hash chain status.
    6. Signature page     — SHA-256 audit hash (monospace), QR code
                            encoding the hash, and officer signature
                            placeholder.

The report is backed by the same SHA-256 audit trail hash that is
written into the ``report_generated`` audit event, so the paper copy
and the digital record verify each other.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from backend.layers.l5_audit import append_audit_event, get_audit_trail, verify_hash_chain

logger = logging.getLogger(__name__)


# ─── Public API ──────────────────────────────────────────────────────────


def generate_report(
    conn: sqlite3.Connection,
    tender_id: str,
    officer_id: str,
) -> dict:
    """Generate an evaluation report for a tender.

    Fetches all evaluations, bidders, and criteria for the tender,
    computes a SHA-256 hash of the complete audit trail, emits JSON +
    PDF, and appends a ``report_generated`` audit event.
    """
    conn.row_factory = sqlite3.Row

    tender = conn.execute(
        "SELECT * FROM tenders WHERE id = ?", (tender_id,)
    ).fetchone()
    if not tender:
        raise ValueError(f"Tender {tender_id} not found")

    evaluations = conn.execute(
        "SELECT * FROM evaluations WHERE tender_id = ?", (tender_id,)
    ).fetchall()
    bidders = conn.execute(
        "SELECT * FROM bidders WHERE tender_id = ?", (tender_id,)
    ).fetchall()
    criteria = conn.execute(
        "SELECT * FROM criteria WHERE tender_id = ?", (tender_id,)
    ).fetchall()

    audit_trail = get_audit_trail(conn, tender_id)
    audit_hash = _compute_audit_trail_hash(audit_trail)

    # Audit chain integrity status (verified, not just read) goes on
    # the signature page so the officer can see it matches.
    chain_valid, chain_error = verify_hash_chain(conn, tender_id)

    report_id = str(uuid.uuid4())
    generated_at = datetime.now(timezone.utc).isoformat()

    bidder_summaries = [
        _build_bidder_summary(bidder, evaluations, criteria, conn)
        for bidder in bidders
    ]

    # Event-type histogram for the audit-trail summary page.
    event_breakdown: dict[str, int] = {}
    for event in audit_trail:
        event_breakdown[event["event_type"]] = event_breakdown.get(
            event["event_type"], 0
        ) + 1

    report_content = {
        "report_id": report_id,
        "tender": {
            "id": tender["id"],
            "title": tender["title"],
            "department": tender["department"],
            "category": tender["category"],
            "status": tender["status"],
        },
        "generated_at": generated_at,
        "generated_by": officer_id,
        "sha256_hash": audit_hash,
        "audit_chain_valid": chain_valid,
        "audit_chain_error": chain_error,
        "summary": {
            "total_bidders": len(bidders),
            "total_criteria": len(criteria),
            "total_evaluations": len(evaluations),
            "auto_committed":
                sum(1 for e in evaluations if e["status"] == "auto_committed"),
            "officer_reviewed":
                sum(1 for e in evaluations if e["status"] == "resolved"),
            "pending":
                sum(1 for e in evaluations
                    if e["status"] in ("pending_review",
                                       "pending_second_officer")),
        },
        "bidder_results": bidder_summaries,
        "audit_trail_entries": len(audit_trail),
        "event_breakdown": event_breakdown,
    }

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    report_path = reports_dir / f"{report_id}.json"
    report_path.write_text(json.dumps(report_content, indent=2, default=str))

    pdf_path = _generate_pdf_report(report_content, reports_dir, report_id)

    append_audit_event(
        conn=conn,
        tender_id=tender_id,
        event_type="report_generated",
        event_data={
            "report_id": report_id,
            "officer_id": officer_id,
            "sha256_hash": audit_hash,
            "audit_trail_entries": len(audit_trail),
        },
        actor=officer_id,
    )

    download_path = str(pdf_path) if pdf_path else str(report_path)

    return {
        "report_id": report_id,
        "tender_id": tender_id,
        "sha256_hash": audit_hash,
        "generated_at": generated_at,
        "download_path": download_path,
        "summary": report_content["summary"],
    }


def get_report(report_id: str) -> dict | None:
    """Retrieve a previously-generated report by ID."""
    json_path = Path("reports") / f"{report_id}.json"
    if json_path.exists():
        return json.loads(json_path.read_text())
    return None


def get_report_download_path(report_id: str) -> Path | None:
    """Return the PDF path if available, else the JSON, else None."""
    pdf_path = Path("reports") / f"{report_id}.pdf"
    if pdf_path.exists():
        return pdf_path
    json_path = Path("reports") / f"{report_id}.json"
    if json_path.exists():
        return json_path
    return None


# ─── Private: hashing + summary builders ─────────────────────────────────


def _compute_audit_trail_hash(audit_trail: list[dict]) -> str:
    """Compute SHA-256 hash of the complete audit trail."""
    payload = json.dumps(
        audit_trail,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_bidder_summary(
    bidder: sqlite3.Row,
    evaluations: Iterable[sqlite3.Row],
    criteria: Iterable[sqlite3.Row],
    conn: sqlite3.Connection,
) -> dict:
    """Build a per-bidder evaluation summary for the report."""
    criteria_map = {c["id"]: c for c in criteria}
    bidder_evals = [e for e in evaluations if e["bidder_id"] == bidder["id"]]

    criterion_results = []
    for ev in bidder_evals:
        criterion = criteria_map.get(ev["criterion_id"])
        extracted_value, explanation = _parse_stored_value(ev["extracted_value"])

        # Fetch source filename for the PDF "source reference" column.
        source_filename = None
        if ev["source_document_id"]:
            doc = conn.execute(
                "SELECT filename FROM documents WHERE id = ?",
                (ev["source_document_id"],),
            ).fetchone()
            if doc:
                source_filename = doc["filename"]

        criterion_results.append({
            "criterion_id": ev["criterion_id"],
            "criterion_text":
                criterion["criterion_text"] if criterion else "Unknown",
            "criterion_type":
                criterion["criterion_type"] if criterion else "Unknown",
            "is_mandatory":
                bool(criterion["is_mandatory"]) if criterion else False,
            "verdict": ev["verdict"],
            "confidence": ev["confidence"],
            "route": ev["route"],
            "status": ev["status"],
            "routing_reason": ev["routing_reason"],
            "extracted_value": extracted_value,
            "explanation": explanation,
            "officer_decision": ev["officer_decision"],
            "officer_id": ev["officer_id"],
            "officer_reason": _parse_officer_reason(ev["officer_reason"]),
            "second_officer_id": ev["second_officer_id"],
            "second_officer_timestamp": ev["second_officer_timestamp"],
            "source_document_id": ev["source_document_id"],
            "source_filename": source_filename,
            "source_page_number": ev["source_page_number"],
            "source_bbox": ev["source_bbox"],
        })

    pass_count = sum(1 for e in bidder_evals if e["verdict"] == "PASS")
    fail_count = sum(1 for e in bidder_evals if e["verdict"] == "FAIL")
    review_count = sum(1 for e in bidder_evals if e["verdict"] == "REVIEW")
    auto_committed = sum(
        1 for e in bidder_evals if e["status"] == "auto_committed"
    )
    officer_reviewed = sum(
        1 for e in bidder_evals if e["status"] == "resolved"
    )

    # Eligibility summary — any mandatory FAIL disqualifies.
    mandatory_fails = sum(
        1 for r in criterion_results
        if r["is_mandatory"] and r["verdict"] == "FAIL"
           and r["officer_decision"] != "override"
    )
    if mandatory_fails > 0:
        eligibility = "Not Eligible"
    elif review_count > 0 or any(
        r["status"] in ("pending_review", "pending_second_officer")
        for r in criterion_results
    ):
        eligibility = "Under Review"
    else:
        eligibility = "Eligible"

    return {
        "bidder_id": bidder["id"],
        "company_name": bidder["company_name"],
        "pan_number": bidder["pan_number"],
        "registration_number": bidder["registration_number"],
        "status": bidder["status"],
        "debarment_status": bidder["debarment_status"],
        "debarment_check_timestamp": bidder["debarment_check_timestamp"],
        "eligibility": eligibility,
        "total_criteria": len(bidder_evals),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "review_count": review_count,
        "auto_committed": auto_committed,
        "officer_reviewed": officer_reviewed,
        "criterion_results": criterion_results,
    }


def _parse_stored_value(raw: Any) -> tuple[Any, dict | None]:
    """Split the extracted_value column into (value, explanation)."""
    if not raw:
        return (None, None)
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return (raw, None)
    if isinstance(parsed, dict) and "__explanation__" in parsed:
        return (parsed.get("__value__"), parsed.get("__explanation__"))
    return (parsed, None)


def _parse_officer_reason(raw: Any) -> dict | str | None:
    """Officer reasons are JSON {'reason': ..., 'text': ...} strings."""
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


# ─── PDF generation ──────────────────────────────────────────────────────


def _generate_pdf_report(
    report_content: dict,
    reports_dir: Path,
    report_id: str,
) -> Path | None:
    """Build the print-ready PDF. Returns the path on success, None otherwise."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm, mm
        from reportlab.pdfgen import canvas as pdf_canvas
        from reportlab.platypus import (
            BaseDocTemplate,
            Frame,
            Image,
            KeepTogether,
            PageBreak,
            PageTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:
        logger.error("reportlab not available: %s", exc)
        return None

    pdf_path = reports_dir / f"{report_id}.pdf"
    tender = report_content["tender"]
    department = tender.get("department") or "GOVERNMENT OF INDIA"

    # ── Header/footer painter ─────────────────────────────────────────
    def _header_footer(canvas: Any, doc: Any) -> None:
        """Draw the confidential header and footer on every page."""
        canvas.saveState()
        width, height = A4

        # Top banner
        canvas.setFillColor(colors.HexColor("#0b3b5a"))
        canvas.rect(0, height - 20 * mm, width, 20 * mm, stroke=0, fill=1)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 11)
        header = f"GOVERNMENT OF INDIA — {department.upper()}"
        canvas.drawCentredString(width / 2, height - 12 * mm, header)
        canvas.setFont("Helvetica", 8)
        canvas.drawCentredString(
            width / 2,
            height - 17 * mm,
            "VerdictAI — Explainable Procurement Intelligence",
        )

        # Bottom banner
        canvas.setFillColor(colors.HexColor("#8a1c1c"))
        canvas.setFont("Helvetica-Bold", 8)
        canvas.drawCentredString(
            width / 2,
            12 * mm,
            "CONFIDENTIAL — PROCUREMENT RECORD",
        )
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#444444"))
        canvas.drawString(
            20 * mm,
            8 * mm,
            f"Report ID: {report_content['report_id']}",
        )
        canvas.drawRightString(
            width - 20 * mm,
            8 * mm,
            f"Page {doc.page}",
        )
        canvas.restoreState()

    # ── Document template with header/footer frame ────────────────────
    # Top 25 mm reserved for header band, bottom 18 mm for footer.
    frame = Frame(
        20 * mm, 18 * mm,
        A4[0] - 40 * mm, A4[1] - 45 * mm,
        id="normal",
    )

    doc = BaseDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=25 * mm,
        bottomMargin=20 * mm,
        title=f"VerdictAI Report — {tender['title']}",
        author="VerdictAI",
    )
    doc.addPageTemplates([
        PageTemplate(id="default", frames=[frame], onPage=_header_footer)
    ])

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="CoverTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=26,
        textColor=colors.HexColor("#0b3b5a"),
        leading=32,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="CoverSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=14,
        textColor=colors.HexColor("#333333"),
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="CoverMeta",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=11,
        textColor=colors.HexColor("#555555"),
        spaceAfter=2,
    ))
    styles.add(ParagraphStyle(
        name="SectionTitle",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#0b3b5a"),
        fontSize=14,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="BidderName",
        parent=styles["Heading3"],
        textColor=colors.HexColor("#0b3b5a"),
        fontSize=13,
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="OverrideBanner",
        parent=styles["Normal"],
        backColor=colors.HexColor("#ffe5e5"),
        borderColor=colors.HexColor("#8a1c1c"),
        borderWidth=1,
        borderPadding=6,
        textColor=colors.HexColor("#5a0000"),
        spaceBefore=6,
        spaceAfter=6,
        fontSize=9,
    ))
    styles.add(ParagraphStyle(
        name="Mono",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=10,
        alignment=TA_CENTER,
        leading=14,
    ))
    styles.add(ParagraphStyle(
        name="BodySmall",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
    ))
    styles.add(ParagraphStyle(
        name="Fact",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,
        leftIndent=8,
        textColor=colors.HexColor("#333333"),
    ))

    elements: list[Any] = []

    # ── 1. COVER PAGE ─────────────────────────────────────────────────
    elements.append(Spacer(1, 3 * cm))
    elements.append(Paragraph("[ VerdictAI ]", styles["CoverMeta"]))
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(Paragraph("EVALUATION REPORT", styles["CoverTitle"]))
    elements.append(Spacer(1, 0.4 * cm))
    elements.append(Paragraph(
        _esc(tender["title"]), styles["CoverSubtitle"]
    ))
    elements.append(Spacer(1, 0.6 * cm))
    elements.append(Paragraph(
        f"Department: {_esc(tender['department'])}",
        styles["CoverMeta"]))
    elements.append(Paragraph(
        f"Category: {_esc(tender['category'])}",
        styles["CoverMeta"]))
    elements.append(Paragraph(
        f"Tender Status: {_esc(tender['status'])}",
        styles["CoverMeta"]))
    elements.append(Spacer(1, 1.5 * cm))

    cover_meta = Table(
        [
            ["Report ID", report_content["report_id"]],
            ["Generated At", report_content["generated_at"]],
            ["Generated By (Officer-in-Charge)",
             report_content["generated_by"]],
            ["Total Bidders", str(report_content["summary"]["total_bidders"])],
            ["Total Criteria", str(report_content["summary"]["total_criteria"])],
        ],
        colWidths=[7 * cm, 8 * cm],
    )
    cover_meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e9eef3")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#0b3b5a")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b9c6d2")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(cover_meta)

    elements.append(Spacer(1, 2 * cm))
    elements.append(Paragraph(
        "This document is a confidential procurement record generated "
        "by VerdictAI's Explainable Evaluation Engine. Every verdict "
        "in this report is backed by a SHA-256 audit trail printed on "
        "the final page.",
        styles["BodySmall"],
    ))
    elements.append(PageBreak())

    # ── 2. EXECUTIVE SUMMARY ──────────────────────────────────────────
    elements.append(Paragraph("Executive Summary", styles["SectionTitle"]))
    elements.append(Paragraph(
        "Per-bidder outcome summary. Status cells are color-coded: "
        "green = eligible, red = not eligible, amber = under review.",
        styles["BodySmall"],
    ))
    elements.append(Spacer(1, 0.3 * cm))

    summary_header = [
        "Bidder",
        "Status",
        "Auto-Committed",
        "Officer-Reviewed",
        "Pass",
        "Fail",
    ]
    summary_rows = [summary_header]
    row_styles: list[tuple] = []
    for i, bidder in enumerate(report_content["bidder_results"], start=1):
        summary_rows.append([
            _cell_paragraph(bidder["company_name"], styles),
            _cell_paragraph(bidder["eligibility"], styles, bold=True),
            str(bidder["auto_committed"]),
            str(bidder["officer_reviewed"]),
            str(bidder["pass_count"]),
            str(bidder["fail_count"]),
        ])
        color = {
            "Eligible": colors.HexColor("#d4edda"),
            "Not Eligible": colors.HexColor("#f8d7da"),
            "Under Review": colors.HexColor("#fff3cd"),
        }.get(bidder["eligibility"], colors.white)
        row_styles.append(
            ("BACKGROUND", (1, i), (1, i), color)
        )

    summary_table = Table(summary_rows, colWidths=[
        5.5 * cm, 3.0 * cm, 2.8 * cm, 2.8 * cm, 1.5 * cm, 1.5 * cm,
    ])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b3b5a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (2, 1), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b9c6d2")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ] + row_styles))
    elements.append(summary_table)

    elements.append(Spacer(1, 0.5 * cm))
    totals = Table(
        [
            ["Total Evaluations",
             str(report_content["summary"]["total_evaluations"])],
            ["Auto-Committed",
             str(report_content["summary"]["auto_committed"])],
            ["Officer-Reviewed",
             str(report_content["summary"]["officer_reviewed"])],
            ["Pending",
             str(report_content["summary"]["pending"])],
        ],
        colWidths=[7 * cm, 4 * cm],
    )
    totals.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e9eef3")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b9c6d2")),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(totals)
    elements.append(PageBreak())

    # ── 3. PER-BIDDER SECTIONS ────────────────────────────────────────
    for bidder in report_content["bidder_results"]:
        elements.extend(_build_bidder_section(bidder, styles, colors))
        elements.append(PageBreak())

    # ── 4. AUDIT TRAIL SUMMARY ────────────────────────────────────────
    elements.append(Paragraph(
        "Audit Trail Summary", styles["SectionTitle"]))
    elements.append(Paragraph(
        "Every action in this evaluation is logged in an append-only "
        "hash-chained ledger. The total event count and per-type "
        "breakdown are summarised below.",
        styles["BodySmall"],
    ))
    elements.append(Spacer(1, 0.3 * cm))

    breakdown = report_content.get("event_breakdown", {})
    event_rows: list[list[Any]] = [["Event Type", "Count"]]
    for event_type in sorted(breakdown):
        event_rows.append([_esc(event_type), str(breakdown[event_type])])
    event_rows.append(
        ["Total", str(report_content["audit_trail_entries"])]
    )

    event_table = Table(event_rows, colWidths=[10 * cm, 3 * cm])
    event_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b3b5a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#e9eef3")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b9c6d2")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(event_table)
    elements.append(Spacer(1, 0.6 * cm))

    chain_text = (
        "Hash chain integrity: <b>INTACT</b> — every audit event "
        "SHA-256-links to its predecessor."
        if report_content["audit_chain_valid"]
        else (
            "Hash chain integrity: <b>BROKEN</b> — "
            f"{_esc(report_content.get('audit_chain_error') or '')}"
        )
    )
    elements.append(Paragraph(chain_text, styles["BodySmall"]))
    elements.append(PageBreak())

    # ── 5. SIGNATURE PAGE ─────────────────────────────────────────────
    elements.append(Paragraph(
        "Certification & Audit Hash", styles["SectionTitle"]))
    elements.append(Paragraph(
        "The SHA-256 digest below cryptographically fingerprints the "
        "full audit trail for this tender at the time of report "
        "generation. Any modification to the underlying records, "
        "however small, will produce a different digest.",
        styles["BodySmall"],
    ))
    elements.append(Spacer(1, 0.6 * cm))

    hash_hex = report_content["sha256_hash"]
    # Break the hash into 16-char groups for legibility.
    hash_display = " ".join(
        hash_hex[i:i + 16] for i in range(0, len(hash_hex), 16)
    )
    elements.append(Paragraph(hash_display, styles["Mono"]))
    elements.append(Spacer(1, 0.8 * cm))

    # QR code encoding the audit hash.
    qr_img = _build_qr_image(hash_hex)
    if qr_img is not None:
        elements.append(qr_img)
        elements.append(Spacer(1, 0.3 * cm))
        elements.append(Paragraph(
            "Scan the QR code above to verify the audit hash.",
            styles["BodySmall"],
        ))

    elements.append(Spacer(1, 1.2 * cm))

    # Officer signature placeholder.
    sig_table = Table(
        [
            [
                Paragraph(
                    "<u>&nbsp;" + " " * 40 + "&nbsp;</u>",
                    styles["Normal"]),
                Paragraph(
                    "<u>&nbsp;" + " " * 40 + "&nbsp;</u>",
                    styles["Normal"]),
            ],
            [
                Paragraph(
                    "Officer-in-Charge<br/>"
                    f"{_esc(report_content['generated_by'])}",
                    styles["BodySmall"]),
                Paragraph(
                    "Date & Seal",
                    styles["BodySmall"]),
            ],
        ],
        colWidths=[8 * cm, 8 * cm],
    )
    sig_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 30),
        ("TOPPADDING", (0, 1), (-1, 1), 6),
    ]))
    elements.append(sig_table)

    try:
        doc.build(elements)
    except Exception as exc:
        logger.error("PDF build failed: %s: %s", type(exc).__name__, exc)
        return None

    return pdf_path


def _build_bidder_section(
    bidder: dict,
    styles: Any,
    colors: Any,
) -> list[Any]:
    """Build the per-bidder elements (header + matrix + overrides)."""
    from reportlab.platypus import (
        KeepTogether, Paragraph, Spacer, Table, TableStyle,
    )
    from reportlab.lib.units import cm

    elements: list[Any] = []
    elements.append(
        Paragraph(f"Bidder: {_esc(bidder['company_name'])}",
                  styles["BidderName"])
    )

    # Bidder header table (PAN, reg, debarment).
    pan = bidder.get("pan_number") or "—"
    reg = bidder.get("registration_number") or "—"
    debarment = (bidder.get("debarment_status") or "unknown").upper()
    debarment_color = {
        "CLEAR": colors.HexColor("#d4edda"),
        "FLAGGED": colors.HexColor("#f8d7da"),
    }.get(debarment, colors.HexColor("#fff3cd"))

    hdr = Table(
        [
            ["PAN", pan,
             "Registration", reg,
             "Debarment", debarment],
        ],
        colWidths=[1.4 * cm, 3.3 * cm, 2.2 * cm, 3.3 * cm, 2.0 * cm, 2.5 * cm],
    )
    hdr.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME", (0, 0), (0, 0), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, 0), "Helvetica-Bold"),
        ("FONTNAME", (4, 0), (4, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#e9eef3")),
        ("BACKGROUND", (2, 0), (2, 0), colors.HexColor("#e9eef3")),
        ("BACKGROUND", (4, 0), (4, 0), colors.HexColor("#e9eef3")),
        ("BACKGROUND", (5, 0), (5, 0), debarment_color),
        ("FONTNAME", (5, 0), (5, 0), "Helvetica-Bold"),
        ("ALIGN", (5, 0), (5, 0), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b9c6d2")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(hdr)
    elements.append(Spacer(1, 0.3 * cm))

    # Eligibility summary strip.
    summary_strip = Table(
        [[
            f"Eligibility: {bidder['eligibility']}",
            f"Pass: {bidder['pass_count']}",
            f"Fail: {bidder['fail_count']}",
            f"Review: {bidder['review_count']}",
            f"Auto: {bidder['auto_committed']}",
            f"Officer: {bidder['officer_reviewed']}",
        ]],
        colWidths=[4 * cm, 2 * cm, 2 * cm, 2.3 * cm, 2 * cm, 2.4 * cm],
    )
    eligibility_bg = {
        "Eligible": colors.HexColor("#d4edda"),
        "Not Eligible": colors.HexColor("#f8d7da"),
        "Under Review": colors.HexColor("#fff3cd"),
    }.get(bidder["eligibility"], colors.white)
    summary_strip.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), eligibility_bg),
        ("FONTNAME", (0, 0), (0, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b9c6d2")),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]))
    elements.append(summary_strip)
    elements.append(Spacer(1, 0.4 * cm))

    # Criterion matrix.
    matrix_header = ["Criterion", "Verdict", "Conf.", "Source", "Officer"]
    matrix_rows: list[list[Any]] = [matrix_header]
    verdict_cell_styles: list[tuple] = []

    for idx, cr in enumerate(bidder["criterion_results"], start=1):
        verdict_bg = _verdict_color(cr["verdict"], cr.get("officer_decision"),
                                    colors)
        mandatory_marker = " ★" if cr.get("is_mandatory") else ""
        criterion_cell = _cell_paragraph(
            (cr["criterion_text"] or "")[:140] + mandatory_marker,
            styles,
        )
        conf = cr.get("confidence")
        conf_cell = f"{conf:.2f}" if isinstance(conf, (int, float)) else "—"

        source_text = "—"
        if cr.get("source_filename") and cr.get("source_page_number"):
            source_text = f"p.{cr['source_page_number']} {cr['source_filename']}"
        elif cr.get("source_filename"):
            source_text = cr["source_filename"]
        source_cell = _cell_paragraph(source_text, styles, small=True)

        officer_cell_text = ""
        if cr.get("officer_id"):
            officer_cell_text = cr["officer_id"]
            if cr.get("officer_decision") == "override":
                officer_cell_text += " (OVR)"
        officer_cell = _cell_paragraph(officer_cell_text or "—", styles,
                                       small=True)

        matrix_rows.append([
            criterion_cell,
            _cell_paragraph(cr["verdict"] or "—", styles, bold=True,
                            align="CENTER"),
            conf_cell,
            source_cell,
            officer_cell,
        ])
        verdict_cell_styles.append(
            ("BACKGROUND", (1, idx), (1, idx), verdict_bg)
        )

    matrix = Table(
        matrix_rows,
        colWidths=[7.0 * cm, 1.6 * cm, 1.1 * cm, 4.0 * cm, 3.0 * cm],
        repeatRows=1,
    )
    matrix.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b3b5a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (2, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b9c6d2")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ] + verdict_cell_styles))
    elements.append(matrix)

    # Override banners — detailed narrative for every overridden criterion.
    overrides = [
        cr for cr in bidder["criterion_results"]
        if cr.get("officer_decision") == "override"
    ]
    if overrides:
        elements.append(Spacer(1, 0.4 * cm))
        elements.append(Paragraph(
            "Officer Overrides",
            styles["SectionTitle"],
        ))
        for cr in overrides:
            reason = cr.get("officer_reason") or {}
            if isinstance(reason, dict):
                reason_code = reason.get("reason") or "—"
                reason_text = reason.get("text") or ""
            else:
                reason_code = str(reason)
                reason_text = ""
            second_officer = cr.get("second_officer_id")
            second_clause = (
                f"Second-officer confirmation by "
                f"<b>{_esc(second_officer)}</b> on "
                f"{_esc(cr.get('second_officer_timestamp') or '')}."
                if second_officer else
                "No second-officer confirmation recorded."
            )
            banner = (
                f"<b>OVERRIDE</b> — {_esc(cr['criterion_text'][:100])}<br/>"
                f"<b>Officer:</b> {_esc(cr.get('officer_id') or '—')}<br/>"
                f"<b>Structured reason:</b> {_esc(reason_code)}"
                + (f"<br/><b>Note:</b> {_esc(reason_text)}"
                   if reason_text else "")
                + f"<br/>{second_clause}"
            )
            elements.append(Paragraph(banner, styles["OverrideBanner"]))

    return elements


def _verdict_color(verdict: str, officer_decision: str | None, colors: Any):
    """Map a verdict + officer decision to a table-cell background colour."""
    if officer_decision == "override":
        return colors.HexColor("#f3d6f0")  # soft purple = overridden
    return {
        "PASS": colors.HexColor("#d4edda"),
        "FAIL": colors.HexColor("#f8d7da"),
        "REVIEW": colors.HexColor("#fff3cd"),
    }.get(verdict, colors.white)


def _cell_paragraph(
    text: str,
    styles: Any,
    *,
    bold: bool = False,
    small: bool = False,
    align: str = "LEFT",
):
    """Wrap text in a Paragraph so table cells handle wrapping correctly."""
    from reportlab.platypus import Paragraph
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    style_name = "CellStyle"
    ps = ParagraphStyle(
        style_name,
        parent=styles["Normal"],
        fontName="Helvetica-Bold" if bold else "Helvetica",
        fontSize=7 if small else 8,
        leading=9 if small else 10,
        alignment=TA_CENTER if align == "CENTER" else TA_LEFT,
    )
    return Paragraph(_esc(text or ""), ps)


def _esc(value: Any) -> str:
    """HTML-escape a value for ReportLab Paragraph."""
    if value is None:
        return ""
    text = str(value)
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
    )


# ─── QR code builder ─────────────────────────────────────────────────────


def _build_qr_image(payload: str) -> Any | None:
    """Render a QR code for the given payload and return a flowable Image."""
    try:
        import qrcode  # type: ignore
        from reportlab.platypus import Image as RLImage
        from reportlab.lib.units import mm
    except ImportError as exc:
        logger.warning("QR code unavailable: %s", exc)
        return None

    try:
        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=6,
            border=2,
        )
        qr.add_data(payload)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return RLImage(buf, width=35 * mm, height=35 * mm)
    except Exception as exc:
        logger.warning("QR generation failed: %s", exc)
        return None
