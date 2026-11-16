"""Opt-in integration probe for the shared client's actual transport selection."""

from __future__ import annotations

import os

import pytest
from dspy import lm15

import dspy_codex_auth

RUN_LIVE_TESTS = os.getenv("DSPY_CODEX_AUTH_RUN_LIVE_TESTS") == "1"


@pytest.mark.live
@pytest.mark.skipif(
    not RUN_LIVE_TESTS,
    reason="set DSPY_CODEX_AUTH_RUN_LIVE_TESTS=1 to run live Codex requests",
)
def test_auto_transport_returns_outputs_and_records_selected_transport() -> None:
    for model in ("gpt-5.6-terra", "gpt-5.6-sol", "gpt-5.6-luna"):
        lm = dspy_codex_auth.LM(f"codex/{model}", cache=False, reasoning_effort="high")
        assert lm.codex_transport == "auto"
        request = lm15.Request(
            model=lm.model,
            messages=(
                lm15.Message.user("Return exactly the single lowercase token: pink"),
            ),
        )
        response = lm(request, timeout=600)
        assert response.text and response.text.strip()
        assert response.provider_data["codex_transport"] in {"http", "websocket"}
