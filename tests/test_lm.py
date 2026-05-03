from __future__ import annotations

import base64
import json
import time
from types import SimpleNamespace

import dspy
import dspy_lm_auth
from dspy_lm_auth.auth import AuthStorage

import dspy_codex_auth
import dspy_codex_auth.lm as codex_lm


def _b64url(data: dict) -> str:
    raw = json.dumps(data, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def make_fake_jwt(account_id: str = "acct_test") -> str:
    header = _b64url({"alg": "none", "typ": "JWT"})
    payload = _b64url(
        {"https://api.openai.com/auth": {"chatgpt_account_id": account_id}}
    )
    return f"{header}.{payload}.signature"


def make_auth_storage(tmp_path, account_id: str = "acct_test") -> AuthStorage:
    storage = AuthStorage(tmp_path / "auth.json")
    storage.set(
        "openai-codex",
        {
            "type": "oauth",
            "access": make_fake_jwt(account_id),
            "refresh": "refresh-token",
            "expires": int(time.time() * 1000) + 60_000,
            "accountId": account_id,
        },
    )
    return storage


class FakeResponsesStream:
    def __init__(self, events: list[SimpleNamespace], response: SimpleNamespace):
        self._events = events
        self.completed_response = SimpleNamespace(response=response)

    def __iter__(self):
        return iter(self._events)


def make_response(output=None) -> SimpleNamespace:
    return SimpleNamespace(
        output=output or [], model="gpt-5.5", usage={}, _hidden_params={}
    )


def test_install_patches_dspy_lm(tmp_path):
    storage = make_auth_storage(tmp_path)
    original_lm = dspy.LM

    try:
        dspy_codex_auth.install(auth_storage=storage)
        assert dspy.LM is dspy_codex_auth.LM
        lm = dspy.LM("codex/gpt-5.5", cache=False)
        assert isinstance(lm, dspy_codex_auth.LM)
        assert lm.model == "openai/gpt-5.5"
    finally:
        dspy_codex_auth.uninstall()
        assert dspy.LM is original_lm


def test_explicit_codex_auth_provider_sets_codex_originator(tmp_path):
    storage = make_auth_storage(tmp_path)

    lm = dspy_codex_auth.LM(
        "openai/gpt-5.5",
        auth_provider="codex",
        auth_storage=storage,
        cache=False,
    )

    assert lm._uses_codex_route is True
    assert lm.kwargs["headers"]["originator"] == "dspy_codex_auth"


def test_codex_request_strips_token_caps_and_accepts_reasoning_summary():
    request = codex_lm._build_codex_request(
        {
            "model": "openai/gpt-5.5",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 100,
            "max_output_tokens": 100,
            "max_completion_tokens": 100,
            "reasoning_effort": "medium",
            "reasoning_summary": "detailed",
        }
    )

    assert "max_tokens" not in request
    assert "max_output_tokens" not in request
    assert "max_completion_tokens" not in request
    assert request["reasoning"] == {"effort": "medium", "summary": "detailed"}


def test_stream_reconstructs_message_from_output_item_done():
    stream = FakeResponsesStream(
        events=[
            SimpleNamespace(
                type="response.output_item.done",
                output_index=0,
                item=SimpleNamespace(
                    type="message",
                    content=[{"type": "output_text", "text": "hello"}],
                ),
            )
        ],
        response=make_response(),
    )

    response = codex_lm._consume_codex_response_stream(stream)

    assert response.output[0].type == "message"
    assert response.output[0].content[0].text == "hello"


def test_stream_reconstructs_message_from_text_done_when_output_item_is_empty():
    stream = FakeResponsesStream(
        events=[
            SimpleNamespace(
                type="response.output_item.done",
                output_index=0,
                item=SimpleNamespace(type="message", content=[]),
            ),
            SimpleNamespace(
                type="response.output_text.done",
                output_index=0,
                content_index=0,
                text="hello from done",
            ),
        ],
        response=make_response(),
    )

    response = codex_lm._consume_codex_response_stream(stream)

    assert response.output[0].content[0].text == "hello from done"


def test_stream_reconstructs_message_from_deltas():
    stream = FakeResponsesStream(
        events=[
            SimpleNamespace(
                type="response.output_text.delta",
                output_index=0,
                content_index=0,
                delta="hel",
            ),
            SimpleNamespace(
                type="response.output_text.delta",
                output_index=0,
                content_index=0,
                delta="lo",
            ),
        ],
        response=make_response(),
    )

    response = codex_lm._consume_codex_response_stream(stream)

    assert response.output[0].content[0].text == "hello"


def test_stream_preserves_reasoning_summary():
    stream = FakeResponsesStream(
        events=[
            SimpleNamespace(
                type="response.reasoning_summary_text.done",
                output_index=0,
                summary_index=0,
                text="Used the normal CDF difference.",
            ),
            SimpleNamespace(
                type="response.output_text.done",
                output_index=1,
                content_index=0,
                text="0.4332",
            ),
        ],
        response=make_response(),
    )

    response = codex_lm._consume_codex_response_stream(stream)
    lm = dspy_codex_auth.LM(
        "openai/gpt-5.5", api_key="dummy", api_base="http://example.invalid"
    )
    lm.model_type = "responses"

    outputs = lm._process_lm_response(response, prompt="x", messages=None)

    assert outputs == [
        {
            "reasoning_content": "Used the normal CDF difference.",
            "text": "0.4332",
        }
    ]


def test_non_codex_routes_fall_through(monkeypatch):
    called = {}

    def fake_forward(self, prompt=None, messages=None, **kwargs):
        called["prompt"] = prompt
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok", tool_calls=None),
                    logprobs=None,
                )
            ],
            model="fake",
            usage={},
        )

    monkeypatch.setattr(dspy_lm_auth.LM, "forward", fake_forward)

    lm = dspy_codex_auth.LM(
        "openai/test", api_key="dummy", api_base="http://example.invalid"
    )
    assert lm("hello") == ["ok"]
    assert called["prompt"] == "hello"
