"""Real LLM client for VerdictAI via OpenRouter.

OpenRouter exposes an OpenAI-compatible API that proxies every major
model provider (Anthropic, OpenAI, Google, etc.) behind a single key.
This client is provider-agnostic — the caller specifies the model
slug (e.g. ``anthropic/claude-3.5-haiku``) and gets the same response
shape back regardless of the underlying vendor.

Features:
- Strict JSON-only responses for structured extraction
- Automatic retries with exponential backoff on transient failures
- Every invocation logged to ``llm_stub_log`` with deterministic prompt
  hash for full audit-trail reproducibility
- Response includes token counts and cost estimates for observability

Usage:

    from backend.services.llm_client import LLMClient

    client = LLMClient()
    result = client.structured_extraction(
        system="You extract eligibility criteria...",
        user="NIT TEXT: ...",
        schema_hint="List of criteria dicts with keys: ...",
    )
    # result.data is the parsed JSON payload
    # result.raw contains metadata for audit logging
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx


logger = logging.getLogger(__name__)


# ─── Config loading ──────────────────────────────────────────────────────


def _load_env() -> None:
    """Load .env via python-dotenv, idempotent."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # env vars may still be set by the shell


_load_env()


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = os.getenv("LLM_MODEL", "anthropic/claude-3.5-haiku")
DEFAULT_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))
DEFAULT_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))


# ─── Response dataclasses ────────────────────────────────────────────────


@dataclass
class LLMResponse:
    """Parsed LLM response with provenance metadata."""

    data: Any
    """Parsed JSON payload if structured; raw text otherwise."""

    text: str
    """Raw text content returned by the model."""

    model: str
    """Actual model that served the request."""

    prompt_hash: str
    """SHA-256 of the canonical request, for reproducibility."""

    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    attempts: int = 1
    error: Optional[str] = None
    raw_choice: dict = field(default_factory=dict)


# ─── Main client ─────────────────────────────────────────────────────────


class LLMClient:
    """Thin wrapper around OpenRouter's chat completions API.

    The client is stateless; construct once per request-handler call
    (or once at module load — either is fine). API keys are read from
    the environment at construction time so hot-reloading a new key
    picks up on the next instantiation.
    """

    MODEL_VERSION_PREFIX = "openrouter"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.model = model or DEFAULT_MODEL
        self.timeout = timeout or DEFAULT_TIMEOUT
        self.max_tokens = max_tokens or DEFAULT_MAX_TOKENS
        self.temperature = (
            temperature if temperature is not None else DEFAULT_TEMPERATURE
        )

    @property
    def is_configured(self) -> bool:
        """True iff an API key is available AND LLM is not globally disabled."""
        if os.getenv("LLM_DISABLED", "").lower() in ("1", "true", "yes"):
            return False
        return bool(self.api_key)

    @property
    def model_version(self) -> str:
        """Canonical model_version string for audit logging."""
        return f"{self.MODEL_VERSION_PREFIX}:{self.model}"

    # ─── Public API ──────────────────────────────────────────────────

    def chat(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
        max_retries: int = 3,
    ) -> LLMResponse:
        """Send a single-turn chat request.

        Args:
            system: System prompt establishing the role / constraints.
            user: User prompt (the actual question / document text).
            json_mode: If True, request JSON-only output.
            max_retries: Number of retries on transient errors (429/5xx).

        Returns:
            :class:`LLMResponse` with text + parsed data (if json_mode).
        """
        if not self.is_configured:
            return self._config_missing_response(system, user)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        prompt_hash = self._compute_prompt_hash(payload)

        start = time.perf_counter()
        last_err: Optional[str] = None
        for attempt in range(1, max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as http:
                    response = http.post(
                        f"{OPENROUTER_BASE_URL}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                            "HTTP-Referer": "https://verdictai.local",
                            "X-Title": "VerdictAI",
                        },
                        json=payload,
                    )

                if response.status_code == 200:
                    body = response.json()
                    latency_ms = int((time.perf_counter() - start) * 1000)
                    return self._parse_success(
                        body, prompt_hash, json_mode, latency_ms, attempt
                    )

                # Retryable statuses
                if response.status_code in (429, 500, 502, 503, 504):
                    last_err = (
                        f"HTTP {response.status_code}: "
                        f"{response.text[:200]}"
                    )
                    if attempt < max_retries:
                        sleep = 0.5 * (2 ** (attempt - 1))
                        logger.warning(
                            "LLM transient error (attempt %d/%d): %s — "
                            "retrying in %.1fs",
                            attempt, max_retries, last_err, sleep,
                        )
                        time.sleep(sleep)
                        continue

                # Non-retryable
                last_err = (
                    f"HTTP {response.status_code}: {response.text[:500]}"
                )
                break

            except httpx.TimeoutException as exc:
                last_err = f"Timeout after {self.timeout}s: {exc}"
                logger.warning(
                    "LLM timeout (attempt %d/%d): %s",
                    attempt, max_retries, last_err,
                )
                if attempt < max_retries:
                    continue
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                logger.error(
                    "LLM request failed (attempt %d/%d): %s",
                    attempt, max_retries, last_err,
                )
                break

        latency_ms = int((time.perf_counter() - start) * 1000)
        return LLMResponse(
            data=None,
            text="",
            model=self.model,
            prompt_hash=prompt_hash,
            latency_ms=latency_ms,
            attempts=attempt,
            error=last_err or "unknown",
        )

    def structured_extraction(
        self,
        system: str,
        user: str,
        *,
        schema_hint: Optional[str] = None,
    ) -> LLMResponse:
        """Chat with strict JSON-only output.

        The system prompt is augmented to request JSON only; the response
        is parsed as JSON and returned in ``LLMResponse.data``.
        """
        enhanced_system = system.strip()
        if schema_hint:
            enhanced_system += (
                f"\n\nReturn ONLY a valid JSON object matching this schema:\n"
                f"{schema_hint}"
            )
        enhanced_system += (
            "\n\nReturn only valid JSON. Do not include markdown fences, "
            "explanations, or any text outside the JSON object."
        )
        return self.chat(enhanced_system, user, json_mode=True)

    # ─── Helpers ──────────────────────────────────────────────────────

    def _parse_success(
        self,
        body: dict,
        prompt_hash: str,
        json_mode: bool,
        latency_ms: int,
        attempt: int,
    ) -> LLMResponse:
        """Parse an HTTP 200 response from OpenRouter."""
        choices = body.get("choices") or []
        if not choices:
            return LLMResponse(
                data=None,
                text="",
                model=body.get("model", self.model),
                prompt_hash=prompt_hash,
                latency_ms=latency_ms,
                attempts=attempt,
                error="Empty choices in response",
            )

        choice = choices[0]
        text = (choice.get("message") or {}).get("content", "") or ""
        usage = body.get("usage") or {}

        data: Any = None
        if json_mode:
            data = self._parse_json_robust(text)

        return LLMResponse(
            data=data,
            text=text,
            model=body.get("model", self.model),
            prompt_hash=prompt_hash,
            tokens_in=int(usage.get("prompt_tokens", 0)),
            tokens_out=int(usage.get("completion_tokens", 0)),
            latency_ms=latency_ms,
            attempts=attempt,
            raw_choice=choice,
        )

    @staticmethod
    def _parse_json_robust(text: str) -> Any:
        """Parse JSON from a model response, handling markdown fences."""
        if not text:
            return None

        stripped = text.strip()
        # Strip ```json ... ``` fences if present
        if stripped.startswith("```"):
            m = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
            if m:
                stripped = m.group(1).strip()

        # Try direct parse first
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

        # Fall back to finding the outermost { ... } or [ ... ] block
        for open_c, close_c in (("{", "}"), ("[", "]")):
            start = stripped.find(open_c)
            end = stripped.rfind(close_c)
            if start >= 0 and end > start:
                try:
                    return json.loads(stripped[start : end + 1])
                except json.JSONDecodeError:
                    continue

        logger.warning(
            "Failed to parse LLM response as JSON: %s",
            stripped[:200],
        )
        return None

    @staticmethod
    def _compute_prompt_hash(payload: dict) -> str:
        """Deterministic SHA-256 of the canonical request payload."""
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _config_missing_response(self, system: str, user: str) -> LLMResponse:
        """Fallback response when no API key is configured."""
        payload = {"model": self.model, "system": system, "user": user}
        return LLMResponse(
            data=None,
            text="",
            model=self.model,
            prompt_hash=self._compute_prompt_hash(payload),
            error="OPENROUTER_API_KEY not configured",
        )


# ─── Module-level singleton ──────────────────────────────────────────────

_default_client: Optional[LLMClient] = None


def get_default_client() -> LLMClient:
    """Return a process-wide default LLMClient instance (lazy-initialised)."""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


# ─── Cached client for reproducibility ───────────────────────────────────


class CachedLLMClient(LLMClient):
    """LLMClient wrapper that replays logged responses from ``llm_stub_log``.

    Reproducibility requires byte-identical LLM outputs when the
    pipeline is re-run from stored inputs. The real LLM is
    non-deterministic (temperature > 0, model drift, etc.), so instead
    of calling it we look up the previously-logged response by the
    same deterministic prompt_hash and return it.

    Usage::

        client = CachedLLMClient(conn=conn, tender_id=tender_id)
        result = client.chat(system=..., user=...)  # hits cache first

    If a cached response exists for the computed prompt_hash, we
    reconstruct an :class:`LLMResponse` from the log row. Otherwise we
    fall through to the real API (and the result is then logged as a
    new entry, but that indicates a non-reproducible run).
    """

    def __init__(
        self,
        conn,
        tender_id: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._conn = conn
        self._tender_id = tender_id
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def cache_hits(self) -> int:
        return self._cache_hits

    @property
    def cache_misses(self) -> int:
        return self._cache_misses

    def chat(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
        max_retries: int = 3,
    ) -> LLMResponse:
        # Recompute the same prompt_hash the parent would compute, then
        # look it up in llm_stub_log. The log stores the *full* response
        # dict (from LLMStub.log_invocation), so we pull the raw text
        # straight from there rather than calling the API.
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        prompt_hash = self._compute_prompt_hash(payload)
        cached = self._lookup_cached(prompt_hash)
        if cached is not None:
            self._cache_hits += 1
            return cached

        self._cache_misses += 1
        return super().chat(
            system, user, json_mode=json_mode, max_retries=max_retries
        )

    def _lookup_cached(self, prompt_hash: str) -> Optional[LLMResponse]:
        """Return an LLMResponse reconstructed from the log, or None."""
        if self._conn is None:
            return None
        try:
            row = self._conn.execute(
                """SELECT response_content, model_version FROM llm_stub_log
                   WHERE prompt_hash = ?
                   ORDER BY timestamp DESC LIMIT 1""",
                (prompt_hash,),
            ).fetchone()
        except Exception as exc:
            logger.warning("CachedLLMClient lookup failed: %s", exc)
            return None

        if not row:
            return None

        try:
            response_content = row["response_content"]
            model_version = row["model_version"] or self.model
        except (KeyError, TypeError):
            # Row factory may not be Row; fall back to index access.
            response_content = row[0]
            model_version = row[1] if len(row) > 1 else self.model

        try:
            payload = json.loads(response_content)
        except (json.JSONDecodeError, TypeError):
            return None

        # response_content stored by LLMStub.log_invocation has the
        # shape {"result": ..., "confidence": ..., ...}. We map that
        # back into an LLMResponse whose .data is the result payload
        # (which is what structured_extraction returns).
        data = payload.get("result", payload)
        text = json.dumps(data, sort_keys=True) if isinstance(data, (dict, list)) else str(data)

        return LLMResponse(
            data=data,
            text=text,
            model=model_version,
            prompt_hash=prompt_hash,
            tokens_in=0,
            tokens_out=0,
            latency_ms=0,
            attempts=1,
        )
