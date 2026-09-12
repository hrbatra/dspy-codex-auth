"""Check an installed adapter wheel against the real shared client, offline."""

import asyncio
import importlib.util
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import httpx


def main() -> None:
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    with tempfile.TemporaryDirectory() as cache_dir:
        os.environ["DSPY_CACHEDIR"] = cache_dir
        _check_adapter()


def _check_adapter() -> None:
    import dspy
    import dspy_codex_auth

    assert dspy_codex_auth.CodexTransport is not None
    assert dspy_codex_auth.DEFAULT_CODEX_WEBSOCKET_CONNECT_TIMEOUT == 10.0
    assert dspy_codex_auth.DEFAULT_CODEX_WEBSOCKET_IDLE_TIMEOUT == 300.0
    assert importlib.util.find_spec("dspy_codex_auth.responses_websocket") is None

    request_count = 0

    def handle_request(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        assert request.method == "POST"
        assert str(request.url) == "https://example.invalid/codex/responses"
        assert request.headers["authorization"] == "Bearer synthetic-token"
        assert request.headers["chatgpt-account-id"] == "acct_smoke"
        assert request.headers["user-agent"] == f"DSPy/{dspy.__version__}"
        payload = json.loads(request.content)
        assert payload["model"] == "gpt-test"
        assert payload["instructions"] == "Answer briefly."
        assert payload["input"] == [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Reply with smoke OK."}],
            }
        ]
        assert payload["reasoning"] == {"effort": "low", "summary": "auto"}
        assert payload["store"] is False
        assert payload["stream"] is True
        assert "messages" not in payload
        assert "reasoning_effort" not in payload
        assert "synthetic-token" not in request.content.decode()
        event = {
            "type": "response.completed",
            "response": {
                "id": "resp_smoke",
                "object": "response",
                "created_at": 1,
                "model": "gpt-test",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_smoke",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {"type": "output_text", "text": "smoke OK", "annotations": []}
                        ],
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
            },
        }
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=f"data: {json.dumps(event)}\n\n",
        )

    sync_client = httpx.Client
    async_client = httpx.AsyncClient
    transport = httpx.MockTransport(handle_request)

    def mock_sync_client(*args, **kwargs):
        return sync_client(*args, **kwargs, transport=transport)

    def mock_async_client(*args, **kwargs):
        return async_client(*args, **kwargs, transport=transport)

    original_lm = dspy.LM
    with tempfile.TemporaryDirectory() as tmp:
        dspy_codex_auth.install(auth_storage=Path(tmp) / "missing-auth.json")
        try:
            assert dspy.LM is dspy_codex_auth.LM
            assert dspy.clients.LM is dspy_codex_auth.LM
            passthrough = dspy.LM(
                "openai/test", api_key="dummy", api_base="http://example.invalid"
            )
            assert isinstance(passthrough, dspy_codex_auth.LM)
            lm = dspy.LM(
                "codex/gpt-test",
                api_key="synthetic-token",
                chatgpt_account_id="acct_smoke",
                api_base="https://example.invalid/codex",
                codex_transport="http",
                reasoning_effort="low",
                cache=False,
                num_retries=0,
            )
            messages = [
                {"role": "system", "content": "Answer briefly."},
                {"role": "user", "content": "Reply with smoke OK."},
            ]
            with (
                patch.object(httpx, "Client", mock_sync_client),
                patch.object(httpx, "AsyncClient", mock_async_client),
            ):
                assert lm(messages=messages) == [{"text": "smoke OK"}]
                assert asyncio.run(lm.acall(messages=messages)) == [{"text": "smoke OK"}]
            assert request_count == 2
        finally:
            dspy_codex_auth.uninstall()
    assert dspy.LM is original_lm
    assert dspy.clients.LM is original_lm
    assert not hasattr(dspy, "getauthtoken")
    print("Adapter distribution smoke passed: shared HTTP client, sync/async, install/uninstall.")


if __name__ == "__main__":
    main()
