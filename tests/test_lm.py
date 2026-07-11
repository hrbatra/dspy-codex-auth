from __future__ import annotations

import asyncio
import base64
import json
import time
from types import SimpleNamespace

import dspy
import pytest

import dspy_codex_auth
from dspy_codex_auth.auth import AuthStorage
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


class RemoteProtocolError(Exception):
    pass


class BrokenResponsesStream:
    def __init__(self):
        self.completed_response = SimpleNamespace(response=make_response())

    def __iter__(self):
        raise RemoteProtocolError(
            "peer closed connection without sending complete message body"
        )


class CompletedResponseLoggingErrorStream:
    def __init__(self):
        self.completed_response = SimpleNamespace(response=make_response_dict())

    def __iter__(self):
        yield SimpleNamespace(
            type="response.output_text.done",
            output_index=0,
            content_index=0,
            text="hello despite logging error",
        )
        raise AttributeError("'dict' object has no attribute 'usage'")


class AsyncCompletedResponseLoggingErrorStream:
    def __init__(self):
        self.completed_response = SimpleNamespace(response=make_response_dict())

    async def __aiter__(self):
        yield SimpleNamespace(
            type="response.output_text.done",
            output_index=0,
            content_index=0,
            text="hello despite async logging error",
        )
        raise AttributeError("'dict' object has no attribute 'usage'")


def make_response(output=None) -> SimpleNamespace:
    return SimpleNamespace(
        output=output or [], model="gpt-5.5", usage={}, _hidden_params={}
    )


def make_response_dict(output=None) -> dict:
    return {
        "output": output or [],
        "model": "gpt-5.5",
        "usage": {},
        "_hidden_params": {},
    }


class FakeUsageTracker:
    def __init__(self):
        self.calls = []

    def add_usage(self, model, usage):
        self.calls.append((model, usage))


def test_forward_skips_usage_tracking_when_usage_is_none(monkeypatch):
    result = SimpleNamespace(output=[], model="gpt-5.4", usage=None, _hidden_params={})

    def fake_get_cached_completion_fn(fn, cache):
        def completion(**kwargs):
            return result

        return completion, {"no-cache": True}

    lm = dspy_codex_auth.LM(
        "openai/gpt-5.4",
        auth_provider="codex",
        api_key="dummy",
        chatgpt_account_id="acct_test",
        cache=False,
    )
    monkeypatch.setattr(lm, "_get_cached_completion_fn", fake_get_cached_completion_fn)

    tracker = FakeUsageTracker()
    with dspy.context(usage_tracker=tracker):
        returned = lm.forward(prompt="hello")

    assert returned is result
    assert returned.usage == {}
    assert tracker.calls == []


def test_aforward_skips_usage_tracking_when_usage_is_none(monkeypatch):
    result = SimpleNamespace(output=[], model="gpt-5.4", usage=None, _hidden_params={})

    def fake_get_cached_completion_fn(fn, cache):
        async def completion(**kwargs):
            return result

        return completion, {"no-cache": True}

    lm = dspy_codex_auth.LM(
        "openai/gpt-5.4",
        auth_provider="codex",
        api_key="dummy",
        chatgpt_account_id="acct_test",
        cache=False,
    )
    monkeypatch.setattr(lm, "_get_cached_completion_fn", fake_get_cached_completion_fn)

    tracker = FakeUsageTracker()

    async def run():
        with dspy.context(usage_tracker=tracker):
            return await lm.aforward(prompt="hello")

    returned = asyncio.run(run())
    assert returned is result
    assert returned.usage == {}
    assert tracker.calls == []


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


def test_codex_request_normalizes_fast_service_tier_to_priority():
    request = codex_lm._build_codex_request(
        {
            "model": "openai/gpt-5.4",
            "messages": [{"role": "user", "content": "hello"}],
            "service_tier": "fast",
        }
    )

    assert request["service_tier"] == "priority"


def test_codex_request_accepts_codex_config_reasoning_aliases():
    request = codex_lm._build_codex_request(
        {
            "model": "openai/gpt-5.4",
            "messages": [{"role": "user", "content": "hello"}],
            "model_reasoning_effort": "low",
            "model_reasoning_summary": "concise",
        }
    )

    assert "model_reasoning_effort" not in request
    assert "model_reasoning_summary" not in request
    assert request["reasoning"] == {"effort": "low", "summary": "concise"}


def test_codex_request_encodes_assistant_messages_as_output_text():
    request = codex_lm._build_codex_request(
        {
            "model": "openai/gpt-5.5",
            "messages": [
                {"role": "system", "content": "Follow the schema."},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "question"}],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "answer"},
                        {"type": "input_text", "text": "more answer"},
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "output_text", "text": "followup"}],
                },
            ],
        }
    )

    assert request["instructions"] == "Follow the schema."
    assert request["input"][0] == {
        "role": "user",
        "content": [{"type": "input_text", "text": "question"}],
    }
    assert request["input"][1] == {
        "role": "assistant",
        "content": [
            {"type": "output_text", "text": "answer"},
            {"type": "output_text", "text": "more answer"},
        ],
    }
    assert request["input"][2] == {
        "role": "user",
        "content": [{"type": "input_text", "text": "followup"}],
    }


def test_labeled_fewshot_demos_build_valid_codex_responses_request():
    from dspy.adapters import ChatAdapter
    from dspy.teleprompt import LabeledFewShot

    class QA(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    trainset = [
        dspy.Example(question="2+2?", answer="4").with_inputs("question"),
    ]
    compiled = LabeledFewShot(k=1).compile(
        dspy.Predict(QA),
        trainset=trainset,
        sample=False,
    )
    predictor = compiled.predictors()[0]
    messages = ChatAdapter().format(
        predictor.signature,
        predictor.demos,
        {"question": "3+3?"},
    )

    assert any(message["role"] == "assistant" for message in messages)

    request = codex_lm._build_codex_request(
        {"model": "openai/gpt-5.5", "messages": messages}
    )

    assistant_messages = [
        message for message in request["input"] if message["role"] == "assistant"
    ]
    assert assistant_messages
    for message in assistant_messages:
        assert message["content"]
        assert all(block["type"] != "input_text" for block in message["content"])


def test_codex_completion_retries_empty_stream_output(monkeypatch):
    calls = 0

    def fake_responses(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return FakeResponsesStream(events=[], response=make_response())
        return FakeResponsesStream(
            events=[
                SimpleNamespace(
                    type="response.output_text.done",
                    output_index=0,
                    content_index=0,
                    text="ok",
                )
            ],
            response=make_response(),
        )

    monkeypatch.setattr(codex_lm.litellm, "responses", fake_responses)

    response = codex_lm._codex_responses_completion(
        {"model": "openai/gpt-5.5", "messages": [{"role": "user", "content": "hi"}]},
        num_retries=1,
    )

    assert calls == 2
    assert response.output[0].content[0].text == "ok"


def test_codex_completion_retries_stream_protocol_errors(monkeypatch):
    calls = 0

    def fake_responses(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return BrokenResponsesStream()
        return FakeResponsesStream(
            events=[
                SimpleNamespace(
                    type="response.output_text.done",
                    output_index=0,
                    content_index=0,
                    text="ok",
                )
            ],
            response=make_response(),
        )

    monkeypatch.setattr(codex_lm.litellm, "responses", fake_responses)

    response = codex_lm._codex_responses_completion(
        {"model": "openai/gpt-5.5", "messages": [{"role": "user", "content": "hi"}]},
        num_retries=1,
    )

    assert calls == 2
    assert response.output[0].content[0].text == "ok"


def test_gepa_reflection_lm_prompt_path_uses_codex_adapter(monkeypatch):
    from dspy.teleprompt.gepa.gepa_utils import DspyAdapter

    captured_request = {}

    def fake_responses(**kwargs):
        captured_request.update(kwargs)
        return FakeResponsesStream(
            events=[
                SimpleNamespace(
                    type="response.output_text.done",
                    output_index=0,
                    content_index=0,
                    text="Use a tighter instruction.",
                )
            ],
            response=make_response(),
        )

    monkeypatch.setattr(codex_lm.litellm, "responses", fake_responses)

    lm = dspy_codex_auth.LM(
        "openai/gpt-5.5",
        auth_provider="codex",
        api_key="dummy",
        chatgpt_account_id="acct_test",
        cache=False,
    )
    adapter = DspyAdapter(
        student_module=dspy.Predict("question -> answer"),
        metric_fn=lambda *args: 1.0,
        feedback_map={},
        reflection_lm=lm,
    )

    assert adapter.stripped_lm_call("Reflect on this trajectory.") == [
        "Use a tighter instruction."
    ]
    assert captured_request["input"] == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "Reflect on this trajectory."}],
        }
    ]


def test_gepa_compile_smoke_uses_codex_lm_without_extra_patches(monkeypatch):
    from dspy.teleprompt import GEPA

    captured_inputs = []

    def fake_responses(**kwargs):
        captured_inputs.append(kwargs["input"])
        input_text = "\n".join(
            block.get("text", "")
            for message in kwargs["input"]
            for block in message["content"]
            if isinstance(block, dict)
        )
        if "Your task is to write a new instruction" in input_text:
            text = (
                "```Given the fields `question`, produce the fields `answer`. "
                "Return exactly the expected answer.```"
            )
        else:
            text = "[[ ## answer ## ]]\n4\n\n[[ ## completed ## ]]"

        return FakeResponsesStream(
            events=[
                SimpleNamespace(
                    type="response.output_text.done",
                    output_index=0,
                    content_index=0,
                    text=text,
                )
            ],
            response=make_response(),
        )

    monkeypatch.setattr(codex_lm.litellm, "responses", fake_responses)

    lm = dspy_codex_auth.LM(
        "openai/gpt-5.5",
        auth_provider="codex",
        api_key="dummy",
        chatgpt_account_id="acct_test",
        cache=False,
    )
    student = dspy.Predict("question -> answer")
    trainset = [dspy.Example(question="2+2?", answer="4").with_inputs("question")]

    def metric(example, pred, trace=None, pred_name=None, pred_trace=None):
        return 1.0 if pred.answer == example.answer else 0.0

    optimizer = GEPA(
        metric=metric,
        max_metric_calls=2,
        reflection_lm=lm,
        use_merge=False,
        skip_perfect_score=False,
        reflection_minibatch_size=1,
        add_format_failure_as_feedback=True,
        num_threads=1,
    )

    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        compiled = optimizer.compile(student, trainset=trainset, valset=trainset)

    assert compiled.signature.instructions
    assert captured_inputs
    assert all(
        block["type"] == "input_text"
        for input_messages in captured_inputs
        for message in input_messages
        for block in message["content"]
        if "text" in block
    )


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


def test_stream_recovers_after_completed_response_logging_usage_shape_error():
    stream = CompletedResponseLoggingErrorStream()

    response = codex_lm._consume_codex_response_stream(stream)

    assert response.output[0].content[0].text == "hello despite logging error"


def test_async_stream_recovers_after_completed_response_logging_usage_shape_error():
    stream = AsyncCompletedResponseLoggingErrorStream()

    response = asyncio.run(codex_lm._aconsume_codex_response_stream(stream))

    assert response.output[0].content[0].text == "hello despite async logging error"


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

    monkeypatch.setattr(codex_lm._DSPY_LM, "forward", fake_forward)

    lm = dspy_codex_auth.LM(
        "openai/test", api_key="dummy", api_base="http://example.invalid"
    )
    assert lm("hello") == ["ok"]
    assert called["prompt"] == "hello"


def make_model_not_found_error(
    model: str = "gpt-5.6-luna",
    *,
    status_code: int = 404,
    provider: str = "openai",
    error_message: str | None = None,
    error_type: str = "invalid_request_error",
    param: str = "model",
    code=None,
    structured: bool = True,
    message_prefix: str = "OpenAIException - ",
    top_level_extra: dict | None = None,
    error_extra: dict | None = None,
    attach_body: bool = False,
):
    if structured:
        payload = {
            "error": {
                "message": error_message or f"Model not found {model}",
                "type": error_type,
                "param": param,
                "code": code,
            }
        }
        payload["error"].update(error_extra or {})
        payload.update(top_level_extra or {})
        message = f"{message_prefix}{json.dumps(payload)}"
    else:
        message = error_message or f"Model not found {model}"
    error = codex_lm.litellm.BadRequestError(
        message=message,
        model=model,
        llm_provider=provider,
    )
    error.status_code = status_code
    if attach_body:
        error.body = payload
    return error


def make_text_response(text: str = "ok") -> SimpleNamespace:
    return make_response(
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(text=text)],
            )
        ]
    )


def dispatch_kwargs(transport: str = "auto") -> dict:
    return {
        "request": {
            "model": "openai/gpt-5.6-luna",
            "messages": [{"role": "user", "content": "hi"}],
        },
        "num_retries": 0,
        "codex_transport": transport,
        "codex_websocket_connect_timeout": 10.0,
        "codex_websocket_idle_timeout": 300.0,
    }


def test_codex_transport_constructor_defaults_and_validates():
    lm = dspy_codex_auth.LM(
        "openai/gpt-5.6-luna",
        auth_provider="codex",
        api_key="dummy",
        chatgpt_account_id="acct_test",
        cache=False,
    )

    assert lm.codex_transport == "auto"
    assert (
        lm.codex_websocket_connect_timeout
        == dspy_codex_auth.DEFAULT_CODEX_WEBSOCKET_CONNECT_TIMEOUT
    )
    assert (
        lm.codex_websocket_idle_timeout
        == dspy_codex_auth.DEFAULT_CODEX_WEBSOCKET_IDLE_TIMEOUT
    )

    explicit = dspy_codex_auth.LM(
        "openai/gpt-5.6-luna",
        auth_provider="codex",
        api_key="dummy",
        chatgpt_account_id="acct_test",
        codex_transport="websocket",
        cache=False,
    )
    assert explicit.codex_transport == "websocket"

    with pytest.raises(ValueError, match="auto.*http.*websocket"):
        dspy_codex_auth.LM(
            "openai/gpt-5.6-luna",
            auth_provider="codex",
            api_key="dummy",
            chatgpt_account_id="acct_test",
            codex_transport="invalid",
            cache=False,
        )


def test_non_codex_constructor_rejects_codex_only_transport_settings():
    with pytest.raises(ValueError, match="require a Codex LM route"):
        dspy_codex_auth.LM(
            "openai/test",
            api_key="dummy",
            api_base="http://example.invalid",
            codex_transport="websocket",
            cache=False,
        )


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_codex_transport_constructor_rejects_invalid_timeouts(timeout):
    with pytest.raises(ValueError, match="positive finite"):
        dspy_codex_auth.LM(
            "openai/gpt-5.6-luna",
            auth_provider="codex",
            api_key="dummy",
            chatgpt_account_id="acct_test",
            codex_websocket_connect_timeout=timeout,
            cache=False,
        )


def test_forward_per_call_transport_overrides_constructor_and_cache_key(monkeypatch):
    captured_calls = []
    lm = dspy_codex_auth.LM(
        "openai/gpt-5.6-luna",
        auth_provider="codex",
        api_key="dummy",
        chatgpt_account_id="acct_test",
        codex_transport="websocket",
        cache=False,
    )

    def fake_get_cached_completion_fn(fn, cache):
        assert fn is codex_lm._codex_completion

        def completion(**kwargs):
            captured_calls.append(kwargs)
            return make_text_response()

        return completion, {"no-cache": True}

    monkeypatch.setattr(lm, "_get_cached_completion_fn", fake_get_cached_completion_fn)

    lm.forward(
        prompt="first",
        codex_transport="http",
        codex_websocket_connect_timeout=21,
        codex_websocket_idle_timeout=22,
    )
    lm.forward(prompt="second", codex_transport="websocket")

    assert [call["codex_transport"] for call in captured_calls] == [
        "http",
        "websocket",
    ]
    assert captured_calls[0]["codex_websocket_connect_timeout"] == 21.0
    assert captured_calls[0]["codex_websocket_idle_timeout"] == 22.0
    assert "codex_transport" not in captured_calls[0]["request"]
    assert "codex_websocket_connect_timeout" not in captured_calls[0]["request"]
    assert "codex_websocket_idle_timeout" not in captured_calls[0]["request"]


def test_transport_selection_produces_distinct_real_cache_entries(monkeypatch):
    calls = []
    http_response = make_text_response("http")
    websocket_response = make_text_response("websocket")

    def fake_http(request, num_retries, cache=None):
        assert "_dspy_codex_transport_controls" not in request
        calls.append(("http", None, None))
        return http_response

    def fake_websocket(request, num_retries, connect_timeout, idle_timeout):
        assert "_dspy_codex_transport_controls" not in request
        calls.append(("websocket", connect_timeout, idle_timeout))
        return websocket_response

    monkeypatch.setattr(codex_lm, "_codex_responses_completion", fake_http)
    monkeypatch.setattr(codex_lm, "_codex_websocket_completion", fake_websocket)
    lm = dspy_codex_auth.LM(
        "openai/gpt-5.6-luna",
        auth_provider="codex",
        api_key="dummy",
        chatgpt_account_id="acct_test",
        cache=True,
    )
    prompt = f"cache transport probe {time.time_ns()}"

    first = lm.forward(prompt=prompt, codex_transport="http")
    second = lm.forward(prompt=prompt, codex_transport="websocket")
    lm.forward(
        prompt=prompt,
        codex_transport="websocket",
        codex_websocket_connect_timeout=11,
    )
    lm.forward(
        prompt=prompt,
        codex_transport="websocket",
        codex_websocket_idle_timeout=301,
    )

    assert first.output[0].content[0].text == "http"
    assert second.output[0].content[0].text == "websocket"
    assert calls == [
        ("http", None, None),
        ("websocket", 10.0, 300.0),
        ("websocket", 11.0, 300.0),
        ("websocket", 10.0, 301.0),
    ]


def test_forward_rejects_invalid_per_call_transport_before_completion(monkeypatch):
    lm = dspy_codex_auth.LM(
        "openai/gpt-5.6-luna",
        auth_provider="codex",
        api_key="dummy",
        chatgpt_account_id="acct_test",
        cache=False,
    )
    monkeypatch.setattr(
        lm,
        "_get_cached_completion_fn",
        lambda *_args: pytest.fail("completion must not be selected"),
    )

    with pytest.raises(ValueError, match="auto.*http.*websocket"):
        lm.forward(prompt="hi", codex_transport="invalid")


def test_exact_structured_model_not_found_is_detected():
    error = make_model_not_found_error()

    assert codex_lm._is_exact_codex_model_not_found(error, "openai/gpt-5.6-luna")


@pytest.mark.parametrize(
    "error",
    [
        make_model_not_found_error(status_code=400),
        make_model_not_found_error(provider="azure"),
        make_model_not_found_error(model="gpt-5.6-terra"),
        make_model_not_found_error(error_message="Model unavailable gpt-5.6-luna"),
        make_model_not_found_error(error_type="server_error"),
        make_model_not_found_error(param="deployment"),
        make_model_not_found_error(code="model_not_found"),
        make_model_not_found_error(structured=False),
        make_model_not_found_error(message_prefix="Proxy wrapper - "),
        make_model_not_found_error(top_level_extra={"request_id": "req_test"}),
        make_model_not_found_error(error_extra={"retryable": False}),
        make_model_not_found_error(attach_body=True),
    ],
)
def test_near_miss_model_errors_do_not_trigger_fallback(error):
    assert not codex_lm._is_exact_codex_model_not_found(error, "openai/gpt-5.6-luna")


def test_auto_transport_keeps_http_success(monkeypatch):
    response = make_text_response("http")
    calls = []

    def fake_http(request, num_retries, cache=None):
        calls.append("http")
        return response

    monkeypatch.setattr(codex_lm, "_codex_responses_completion", fake_http)
    monkeypatch.setattr(
        codex_lm,
        "_codex_websocket_completion",
        lambda *_args, **_kwargs: pytest.fail("websocket must not be called"),
    )

    assert codex_lm._codex_completion(**dispatch_kwargs()) is response
    assert calls == ["http"]


def test_auto_transport_falls_back_only_for_exact_model_not_found(monkeypatch):
    response = make_text_response("websocket")
    calls = []

    def fake_http(request, num_retries, cache=None):
        calls.append("http")
        raise make_model_not_found_error()

    def fake_websocket(request, num_retries, connect_timeout, idle_timeout):
        calls.append(("websocket", connect_timeout, idle_timeout))
        return response

    monkeypatch.setattr(codex_lm, "_codex_responses_completion", fake_http)
    monkeypatch.setattr(codex_lm, "_codex_websocket_completion", fake_websocket)

    assert codex_lm._codex_completion(**dispatch_kwargs()) is response
    assert calls == ["http", ("websocket", 10.0, 300.0)]


def test_auto_transport_propagates_near_miss_without_websocket(monkeypatch):
    error = make_model_not_found_error(param="deployment")

    def fake_http(request, num_retries, cache=None):
        raise error

    monkeypatch.setattr(codex_lm, "_codex_responses_completion", fake_http)
    monkeypatch.setattr(
        codex_lm,
        "_codex_websocket_completion",
        lambda *_args, **_kwargs: pytest.fail("websocket must not be called"),
    )

    with pytest.raises(codex_lm.litellm.BadRequestError) as caught:
        codex_lm._codex_completion(**dispatch_kwargs())
    assert caught.value is error


def test_explicit_http_never_falls_back(monkeypatch):
    error = make_model_not_found_error()

    def fake_http(request, num_retries, cache=None):
        raise error

    monkeypatch.setattr(codex_lm, "_codex_responses_completion", fake_http)
    monkeypatch.setattr(
        codex_lm,
        "_codex_websocket_completion",
        lambda *_args, **_kwargs: pytest.fail("websocket must not be called"),
    )

    with pytest.raises(codex_lm.litellm.BadRequestError) as caught:
        codex_lm._codex_completion(**dispatch_kwargs("http"))
    assert caught.value is error


def test_explicit_websocket_bypasses_http(monkeypatch):
    response = make_text_response("websocket")
    monkeypatch.setattr(
        codex_lm,
        "_codex_responses_completion",
        lambda *_args, **_kwargs: pytest.fail("http must not be called"),
    )
    monkeypatch.setattr(
        codex_lm,
        "_codex_websocket_completion",
        lambda *_args, **_kwargs: response,
    )

    assert codex_lm._codex_completion(**dispatch_kwargs("websocket")) is response


def test_async_auto_transport_falls_back_for_exact_model_not_found(monkeypatch):
    response = make_text_response("websocket")
    calls = []

    async def fake_http(request, num_retries, cache=None):
        calls.append("http")
        raise make_model_not_found_error()

    async def fake_websocket(request, num_retries, connect_timeout, idle_timeout):
        calls.append("websocket")
        return response

    monkeypatch.setattr(codex_lm, "_acodex_responses_completion", fake_http)
    monkeypatch.setattr(codex_lm, "_acodex_websocket_completion", fake_websocket)

    returned = asyncio.run(codex_lm._acodex_completion(**dispatch_kwargs()))
    assert returned is response
    assert calls == ["http", "websocket"]


def test_websocket_completion_builds_wire_request_and_reconstructs_events(monkeypatch):
    captured = {}

    def fake_websocket_response(request, **kwargs):
        captured["request"] = request
        captured.update(kwargs)
        return codex_lm.WebSocketResult(
            events=[
                {
                    "type": "response.output_text.done",
                    "output_index": 0,
                    "content_index": 0,
                    "text": "from websocket",
                },
                {
                    "type": "response.completed",
                    "response": {
                        "status": "completed",
                        "model": "gpt-5.6-luna",
                        "output": [],
                    },
                },
            ],
            response={
                "status": "completed",
                "model": "gpt-5.6-luna",
                "output": [],
                "usage": {},
            },
        )

    monkeypatch.setattr(codex_lm, "websocket_response", fake_websocket_response)

    response = codex_lm._codex_websocket_completion(
        {
            "model": "openai/gpt-5.6-luna",
            "messages": [{"role": "user", "content": "hi"}],
            "api_key": "secret-token",
            "api_base": "https://chatgpt.com/backend-api/codex",
            "headers": {"chatgpt-account-id": "acct_test"},
            "model_type": "responses",
            "use_developer_role": True,
            "temperature": None,
        },
        num_retries=0,
        connect_timeout=10.0,
        idle_timeout=300.0,
    )

    assert response.output[0].content[0].text == "from websocket"
    assert captured["api_key"] == "secret-token"
    assert captured["api_base"] == "https://chatgpt.com/backend-api/codex"
    assert captured["headers"]["User-Agent"].startswith("DSPy/")
    assert captured["request"]["model"] == "openai/gpt-5.6-luna"
    assert captured["request"]["stream"] is True
    assert captured["request"]["store"] is False
    assert all(value is not None for value in captured["request"].values())
    for client_key in (
        "api_key",
        "api_base",
        "headers",
        "model_type",
        "use_developer_role",
    ):
        assert client_key not in captured["request"]


@pytest.mark.parametrize("function_call_source", ["terminal", "event"])
def test_websocket_function_calls_are_compatible_with_dspy_processing(
    monkeypatch,
    function_call_source,
):
    function_call = {
        "type": "function_call",
        "id": "fc_test",
        "call_id": "call_test",
        "name": "lookup_weather",
        "arguments": '{"city":"Chicago"}',
        "status": "completed",
    }
    terminal_output = [function_call] if function_call_source == "terminal" else []
    events = []
    if function_call_source == "event":
        events.append(
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": function_call,
            }
        )
    events.append(
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "model": "gpt-5.6-luna",
                "output": terminal_output,
            },
        }
    )

    monkeypatch.setattr(
        codex_lm,
        "websocket_response",
        lambda *_args, **_kwargs: codex_lm.WebSocketResult(
            events=events,
            response={
                "status": "completed",
                "model": "gpt-5.6-luna",
                "output": terminal_output,
                "usage": {},
            },
        ),
    )
    response = codex_lm._codex_websocket_completion(
        {
            "model": "openai/gpt-5.6-luna",
            "messages": [{"role": "user", "content": "hi"}],
            "api_key": "secret-token",
            "api_base": "https://chatgpt.com/backend-api/codex",
            "headers": {"chatgpt-account-id": "acct_test"},
        },
        num_retries=0,
        connect_timeout=10.0,
        idle_timeout=300.0,
    )
    lm = dspy_codex_auth.LM(
        "openai/gpt-5.6-luna",
        auth_provider="codex",
        api_key="dummy",
        chatgpt_account_id="acct_test",
        cache=False,
    )

    processed = lm._process_response(response)

    assert processed[0]["tool_calls"][0]["call_id"] == "call_test"
    assert processed[0]["tool_calls"][0]["name"] == "lookup_weather"
