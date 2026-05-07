"""Legacy import shim for VerdictAI semantic service.

Historically this module held a mock LLM returning pre-configured
scenarios. That stub has been replaced by real pattern matching,
semantic embeddings, and evidence extraction in
:mod:`backend.services.semantic_service`.

This file is preserved only so existing callers (``from
backend.services.llm_stub import LLMStub``) keep working. The class
re-exported here is the real semantic service; it emits
``is_simulated=False`` and the true model version string.

Prefer importing directly from :mod:`backend.services.semantic_service`
for new code.
"""

from services.semantic_service import LLMStub  # noqa: F401

__all__ = ["LLMStub"]
