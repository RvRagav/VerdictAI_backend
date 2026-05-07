"""Shared pytest configuration for VerdictAI backend tests.

Disables the LLM by default so unit tests don't hit the real OpenRouter
API (which is slow and costs money). Tests that specifically want to
exercise LLM integration can set ``monkeypatch.setenv("LLM_DISABLED", "0")``
or skip the auto-use fixture.
"""

import os
import pytest


@pytest.fixture(autouse=True)
def _disable_llm_by_default(monkeypatch):
    """Disable LLM calls in unit tests unless a test opts in explicitly."""
    monkeypatch.setenv("LLM_DISABLED", "1")
    yield
