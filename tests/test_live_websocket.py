from __future__ import annotations

import os
from functools import wraps
from typing import Any

import dspy
import pytest

import dspy_codex_auth
import dspy_codex_auth.lm as codex_lm


RUN_LIVE_TESTS = os.getenv("DSPY_CODEX_AUTH_RUN_LIVE_TESTS") == "1"


@pytest.mark.live
@pytest.mark.skipif(
    not RUN_LIVE_TESTS,
    reason="set DSPY_CODEX_AUTH_RUN_LIVE_TESTS=1 to run live Codex requests",
)
def test_default_auto_transport_routes_only_luna_to_websocket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket_models: list[str] = []
    real_websocket_completion = codex_lm._codex_websocket_completion

    @wraps(real_websocket_completion)
    def record_websocket_completion(
        request: dict[str, Any],
        num_retries: int,
        connect_timeout: float,
        idle_timeout: float,
    ) -> Any:
        websocket_models.append(str(request["model"]).removeprefix("openai/"))
        return real_websocket_completion(
            request,
            num_retries,
            connect_timeout,
            idle_timeout,
        )

    monkeypatch.setattr(
        codex_lm,
        "_codex_websocket_completion",
        record_websocket_completion,
    )

    models = ("gpt-5.6-terra", "gpt-5.6-sol", "gpt-5.6-luna")
    prompt = "Return exactly the single lowercase token: pink"

    try:
        dspy_codex_auth.install()
        for model in models:
            lm = dspy.LM(
                f"codex/{model}",
                cache=False,
                reasoning_effort="high",
            )
            assert lm.codex_transport == "auto"

            outputs = lm(prompt)

            assert outputs
            # The shim returns dict outputs ({'text': ..., 'reasoning_content': ...})
            # for reasoning models on EVERY transport — assert on the text payload.
            def _text(output: object) -> str:
                if isinstance(output, dict):
                    return str(output.get("text") or "")
                return str(output or "")

            assert all(_text(output).strip() for output in outputs)
    finally:
        dspy_codex_auth.uninstall()

    assert websocket_models
    assert set(websocket_models) == {"gpt-5.6-luna"}
