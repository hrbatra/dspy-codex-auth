from __future__ import annotations

import asyncio
import base64
import json
import time
from types import SimpleNamespace

import dspy
import httpx
import pytest
from openai_codex_auth import CodexAuth, CodexResponse
from pydantic import BaseModel

import dspy_codex_auth
import dspy_codex_auth.lm as codex_lm


def make_auth_storage(tmp_path) -> CodexAuth:
    claims = {
        "exp": int(time.time()) + 3600,
        "https://api.openai.com/auth": {"chatgpt_account_id": "acct_test"},
    }
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=")
    path = tmp_path / "auth.json"
    path.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": f"e30.{payload.decode()}.signature",
                    "refresh_token": "synthetic-refresh-token",
                    "account_id": "acct_test",
                },
            }
        )
    )
    return CodexAuth(path)


def make_response(text="ok", *, output=None, usage=None, transport="http"):
    return CodexResponse(
        response={
            "id": "resp_test",
            "model": "gpt-5.5-actual",
            "status": "completed",
            "output": output
            if output is not None
            else [
                {
                    "type": "message",
                    "id": "msg_test",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                }
            ],
            "usage": usage,
            "metadata": {"trace": "kept"},
        },
        transport=transport,
    )


def make_lm(**kwargs):
    return dspy_codex_auth.LM(
        "codex/gpt-5.5",
        **{
            "api_key": "synthetic-key",
            "chatgpt_account_id": "acct_test",
            "cache": False,
            **kwargs,
        },
    )


@pytest.fixture
def client_calls(monkeypatch):
    calls = SimpleNamespace(
        constructors=[], sync=[], async_=[], response=make_response()
    )

    class FakeClient:
        def __init__(self, **kwargs):
            calls.constructors.append(kwargs)

        def create(self, **kwargs):
            calls.sync.append(kwargs)
            return calls.response

        async def acreate(self, **kwargs):
            calls.async_.append(kwargs)
            return calls.response

    monkeypatch.setattr(codex_lm, "CodexClient", FakeClient)
    return calls


def test_install_routes_to_shared_client_without_eager_auth(tmp_path, client_calls):
    storage = make_auth_storage(tmp_path)
    original_lm = dspy.LM
    try:
        dspy_codex_auth.install(auth_storage=storage)
        assert dspy.LM is dspy_codex_auth.LM
        lm = dspy.LM("codex/gpt-5.5", cache=False)
        assert lm.model == "openai/gpt-5.5"
        assert "api_key" not in lm.kwargs
        assert "headers" not in lm.kwargs
        lm("hello")
        constructor = client_calls.constructors[0]
        assert constructor["auth"] is storage
        assert constructor["api_key"] is None
        assert constructor["account_id"] is None
        assert constructor["originator"] == "dspy_codex_auth"
        assert constructor["user_agent"] == f"DSPy/{dspy.__version__}"
    finally:
        dspy_codex_auth.uninstall()
        assert dspy.LM is original_lm


@pytest.mark.parametrize("alias", ["codex", "chatgpt", "openai-codex"])
def test_route_aliases_use_shared_client(alias, client_calls):
    lm = dspy_codex_auth.LM(f"{alias}/gpt-5.5", api_key="synthetic", cache=False)
    lm("hello")
    assert client_calls.sync[0]["model"] == "gpt-5.5"


def test_explicit_auth_provider_passes_overrides(client_calls):
    lm = dspy_codex_auth.LM(
        "openai/gpt-5.5",
        auth_provider="codex",
        api_key="override-key",
        chatgpt_account_id="override-account",
        originator="custom-originator",
        headers={"X-Test": "custom"},
        cache=False,
    )
    lm.forward(prompt="hello", api_key="per-call-key")
    constructor = client_calls.constructors[0]
    assert constructor["api_key"] == "per-call-key"
    assert constructor["account_id"] == "override-account"
    assert constructor["headers"] == {"X-Test": "custom"}
    assert constructor["originator"] == "custom-originator"
    assert (
        not {"api_key", "chatgpt_account_id", "headers", "originator", "api_base"}
        & client_calls.sync[0].keys()
    )


def test_sync_and_async_use_matching_shared_client_methods(client_calls):
    lm = make_lm()
    assert lm("sync") == [{"text": "ok"}]
    assert asyncio.run(lm.acall("async")) == [{"text": "ok"}]
    assert len(client_calls.sync) == len(client_calls.async_) == 1
    assert client_calls.sync[0]["input"][0]["content"][0]["text"] == "sync"
    assert client_calls.async_[0]["input"][0]["content"][0]["text"] == "async"
    assert all(
        call["model"] == "gpt-5.5" for call in client_calls.sync + client_calls.async_
    )


def test_shared_response_preserves_usage_model_metadata_and_history(client_calls):
    usage = {"input_tokens": 12, "output_tokens": 7, "total_tokens": 19}
    client_calls.response = make_response(usage=usage, transport="websocket")
    tracker = SimpleNamespace(calls=[])
    tracker.add_usage = lambda model, value: tracker.calls.append((model, value))
    lm = make_lm()
    with dspy.context(usage_tracker=tracker):
        assert lm("hello") == [{"text": "ok"}]
    assert tracker.calls == [("openai/gpt-5.5", usage)]
    entry = lm.history[-1]
    assert entry["usage"] == usage
    assert entry["response_model"] == "gpt-5.5-actual"
    assert entry["response"].metadata == {"trace": "kept"}
    assert entry["response"].codex_transport == "websocket"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_missing_usage_is_empty_for_dspy(client_calls, asynchronous):
    lm = make_lm()
    tracker = SimpleNamespace(add_usage=lambda *_: pytest.fail("no usage to record"))
    with dspy.context(usage_tracker=tracker):
        if asynchronous:
            result = asyncio.run(lm.aforward(prompt="hello"))
        else:
            result = lm.forward(prompt="hello")
    assert result.usage == {}


def test_request_options_translate_to_native_fields(client_calls):
    lm = make_lm(
        reasoning_effort="medium",
        reasoning_summary="detailed",
        service_tier="fast",
        max_tokens=18000,
        num_retries=2,
    )
    lm.forward(prompt="hello", rollout_id=7)
    request = client_calls.sync[0]
    assert request["reasoning"] == {"effort": "medium", "summary": "detailed"}
    assert request["max_output_tokens"] == 18000
    assert (
        request["service_tier"] == "fast"
    )  # Backend normalization belongs to the core.
    assert request["max_retries"] == 2
    assert request["transport"] == "auto"
    assert request["instructions"] == dspy_codex_auth.DEFAULT_CODEX_INSTRUCTIONS
    assert (
        not {
            "messages",
            "max_tokens",
            "max_completion_tokens",
            "rollout_id",
            "stream",
            "store",
            "model_type",
            "use_developer_role",
            "reasoning_effort",
            "reasoning_summary",
        }
        & request.keys()
    )


def test_codex_config_reasoning_options_are_translated(client_calls):
    make_lm(model_reasoning_effort="low", model_reasoning_summary="concise")("hello")
    request = client_calls.sync[0]
    assert request["reasoning"] == {"effort": "low", "summary": "concise"}
    assert "model_reasoning_effort" not in request
    assert "model_reasoning_summary" not in request


def test_native_reasoning_and_text_options_preserved(client_calls):
    make_lm(reasoning={"effort": "high"}, text={"verbosity": "low"})("hello")
    assert client_calls.sync[0]["reasoning"] == {"effort": "high"}
    assert client_calls.sync[0]["text"] == {"verbosity": "low"}


def test_pydantic_response_format_converts_to_native_json_schema(client_calls):
    class Answer(BaseModel):
        answer: str

    make_lm()("hello", response_format=Answer, text={"verbosity": "low"})
    assert client_calls.sync[0]["text"] == {
        "verbosity": "low",
        "format": {
            "name": "Answer",
            "type": "json_schema",
            "schema": Answer.model_json_schema(),
        },
    }
    assert "response_format" not in client_calls.sync[0]


def test_json_adapter_predict_uses_shared_client(client_calls):
    client_calls.response = make_response('{"answer":"four"}')
    with dspy.context(lm=make_lm(), adapter=dspy.JSONAdapter()):
        prediction = dspy.Predict("question -> answer")(question="2+2?")
    assert prediction.answer == "four"
    assert client_calls.sync
    assert client_calls.sync[0]["text"]["format"]["type"] in {
        "json_schema",
        "json_object",
    }


def test_codex_request_encodes_assistant_messages_and_media(client_calls):
    make_lm().forward(
        messages=[
            {"role": "system", "content": "Follow the schema."},
            {"role": "developer", "content": "Be concise."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "question"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.invalid/image.png"},
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "answer"},
                    {"type": "input_text", "text": "more answer"},
                ],
            },
            {"role": "user", "content": [{"type": "output_text", "text": "followup"}]},
        ]
    )
    request = client_calls.sync[0]
    assert request["instructions"] == "Follow the schema.\n\nBe concise."
    assert request["input"][0]["content"] == [
        {"type": "input_text", "text": "question"},
        {"type": "input_image", "image_url": "https://example.invalid/image.png"},
    ]
    assert request["input"][1]["content"] == [
        {"type": "output_text", "text": "answer"},
        {"type": "output_text", "text": "more answer"},
    ]
    assert request["input"][2]["content"] == [
        {"type": "input_text", "text": "followup"}
    ]


def test_function_calls_and_reasoning_are_compatible_with_dspy(client_calls):
    function_call = {
        "type": "function_call",
        "id": "fc_test",
        "call_id": "call_test",
        "name": "lookup_weather",
        "arguments": '{"city":"Chicago"}',
        "status": "completed",
    }
    client_calls.response = make_response(
        output=[
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Need weather."}],
            },
            function_call,
        ]
    )
    output = make_lm()("hello")[0]
    assert output["tool_calls"][0]["name"] == "lookup_weather"
    assert output["tool_calls"][0]["call_id"] == "call_test"
    assert output["tool_calls"][0]["arguments"] == '{"city":"Chicago"}'
    assert output["reasoning_content"] == "Need weather."


def test_transport_overrides_and_timeout_precedence(client_calls):
    lm = make_lm(codex_transport="websocket", timeout=600)
    lm.forward(
        prompt="one",
        codex_transport="http",
        codex_websocket_connect_timeout=21,
        codex_websocket_idle_timeout=22,
    )
    lm.forward(prompt="two", codex_websocket_idle_timeout=23, timeout=700)
    lm.forward(prompt="three")
    lm.forward(prompt="four", timeout=None, codex_websocket_idle_timeout=24)
    assert [
        (c["transport"], c["connect_timeout"], c["idle_timeout"], c.get("timeout"))
        for c in client_calls.sync
    ] == [
        ("http", 21.0, 22.0, 600),
        ("websocket", 10.0, 700.0, 700),
        ("websocket", 10.0, 600.0, 600),
        ("websocket", 10.0, 24.0, None),
    ]


def test_async_timeout_and_transport_overrides(client_calls):
    asyncio.run(
        make_lm().aforward(
            prompt="hello",
            codex_transport="websocket",
            codex_websocket_connect_timeout=12,
            timeout=620,
        )
    )
    call = client_calls.async_[0]
    assert call["transport"] == "websocket"
    assert call["connect_timeout"] == 12.0
    assert call["idle_timeout"] == call["timeout"] == 620.0


def test_http_preserves_structured_timeout(client_calls):
    timeout = httpx.Timeout(600, connect=8, pool=9)
    make_lm(codex_transport="http").forward(prompt="hello", timeout=timeout)
    assert client_calls.sync[0]["timeout"] is timeout


def test_real_cache_separates_transport_and_deadlines(client_calls):
    lm = make_lm(cache=True)
    prompt = f"cache transport {time.time_ns()}"
    lm(prompt, codex_transport="http")
    lm(prompt, codex_transport="websocket")
    lm(prompt, codex_transport="websocket", codex_websocket_connect_timeout=11)
    lm(prompt, codex_transport="websocket", codex_websocket_idle_timeout=301)
    lm(prompt, codex_transport="http")
    assert [
        (c["transport"], c["connect_timeout"], c["idle_timeout"])
        for c in client_calls.sync
    ] == [
        ("http", 10.0, 300.0),
        ("websocket", 10.0, 300.0),
        ("websocket", 11.0, 300.0),
        ("websocket", 10.0, 301.0),
    ]


def test_uncached_calls_reuse_auth_source_not_a_token_snapshot(tmp_path, client_calls):
    storage = make_auth_storage(tmp_path)
    lm = dspy_codex_auth.LM("codex/gpt-5.5", auth_storage=storage, cache=False)
    lm("first")
    lm("second")
    assert len(client_calls.constructors) == 2
    assert all(
        c["auth"] is storage and c["api_key"] is None for c in client_calls.constructors
    )


@pytest.mark.parametrize("transport", ["invalid", 1, None])
def test_constructor_rejects_invalid_transport(transport):
    with pytest.raises(ValueError, match="auto.*http.*websocket"):
        make_lm(codex_transport=transport)


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True])
def test_constructor_rejects_invalid_websocket_timeouts(timeout):
    with pytest.raises(ValueError, match="positive finite"):
        make_lm(codex_websocket_connect_timeout=timeout)


def test_invalid_per_call_transport_fails_before_client(client_calls):
    with pytest.raises(ValueError, match="auto.*http.*websocket"):
        make_lm().forward(prompt="hello", codex_transport="invalid")
    assert not client_calls.constructors


def test_non_codex_route_uses_standard_dspy_sync_and_async(monkeypatch, client_calls):
    calls = []
    result = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="ok", tool_calls=None), logprobs=None
            )
        ],
        model="fake",
        usage={},
    )

    def forward(self, prompt=None, messages=None, **kwargs):
        calls.append(("sync", prompt, kwargs))
        return result

    async def aforward(self, prompt=None, messages=None, **kwargs):
        calls.append(("async", prompt, kwargs))
        return result

    monkeypatch.setattr(codex_lm._DSPY_LM, "forward", forward)
    monkeypatch.setattr(codex_lm._DSPY_LM, "aforward", aforward)
    lm = dspy_codex_auth.LM(
        "openai/test", api_key="synthetic", api_base="http://example.invalid"
    )
    assert lm("hello", temperature=0.3) == ["ok"]
    assert asyncio.run(lm.acall("async", temperature=0.4)) == ["ok"]
    assert calls == [
        ("sync", "hello", {"temperature": 0.3}),
        ("async", "async", {"temperature": 0.4}),
    ]
    assert not client_calls.constructors


def test_non_codex_route_rejects_codex_transport_overrides():
    with pytest.raises(ValueError, match="require a Codex LM route"):
        dspy_codex_auth.LM(
            "openai/test", api_key="synthetic", codex_transport="websocket"
        )
    lm = dspy_codex_auth.LM("openai/test", api_key="synthetic")
    with pytest.raises(ValueError, match="require a Codex LM route"):
        lm.forward("hello", codex_transport="websocket")
    with pytest.raises(ValueError, match="require a Codex LM route"):
        asyncio.run(lm.aforward("hello", codex_websocket_idle_timeout=22))


def test_labeled_fewshot_demos_reach_shared_client_as_output_text(client_calls):
    from dspy.teleprompt import LabeledFewShot

    student = dspy.Predict("question -> answer")
    trainset = [dspy.Example(question="2+2?", answer="4").with_inputs("question")]
    compiled = LabeledFewShot(k=1).compile(student, trainset=trainset, sample=False)
    client_calls.response = make_response(
        "[[ ## answer ## ]]\n6\n\n[[ ## completed ## ]]"
    )
    with dspy.context(lm=make_lm(), adapter=dspy.ChatAdapter()):
        prediction = compiled(question="3+3?")
    assert prediction.answer == "6"
    assistant_messages = [
        item for item in client_calls.sync[0]["input"] if item["role"] == "assistant"
    ]
    assert assistant_messages
    assert all(
        block["type"] == "output_text"
        for item in assistant_messages
        for block in item["content"]
    )


def test_gepa_reflection_plain_prompt_uses_shared_client(client_calls):
    from dspy.teleprompt.gepa.gepa_utils import DspyAdapter

    client_calls.response = make_response("Use a tighter instruction.")
    adapter = DspyAdapter(
        student_module=dspy.Predict("question -> answer"),
        metric_fn=lambda *args: 1.0,
        feedback_map={},
        reflection_lm=make_lm(),
    )
    assert adapter.stripped_lm_call("Reflect on this trajectory.") == [
        "Use a tighter instruction."
    ]
    assert client_calls.sync[0]["input"] == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "Reflect on this trajectory."}],
        }
    ]


def test_gepa_compile_uses_shared_client_without_extra_patches(monkeypatch):
    from dspy.teleprompt import GEPA

    captured_inputs = []

    def create(self, **kwargs):
        captured_inputs.append(kwargs["input"])
        input_text = "\n".join(
            block.get("text", "")
            for item in kwargs["input"]
            for block in item["content"]
        )
        if "Your task is to write a new instruction" in input_text:
            text = "```Given the fields `question`, produce the fields `answer`. Return exactly the expected answer.```"
        else:
            text = "[[ ## answer ## ]]\n4\n\n[[ ## completed ## ]]"
        return make_response(text)

    monkeypatch.setattr(codex_lm.CodexClient, "create", create)
    lm = make_lm()
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


def test_chat_tool_continuation_and_declarations_become_native_responses(client_calls):
    function = {
        "name": "lookup_weather",
        "description": "Weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        "strict": True,
    }
    make_lm().forward(
        messages=[
            {"role": "user", "content": "Weather in Chicago?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_weather",
                        "type": "function",
                        "function": {
                            "name": "lookup_weather",
                            "arguments": '{"city":"Chicago"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_weather",
                "content": '{"temperature":72}',
            },
        ],
        tools=[{"type": "function", "function": function}, {"type": "web_search"}],
        tool_choice={"type": "function", "function": {"name": "lookup_weather"}},
    )
    request = client_calls.sync[0]
    assert request["input"][1:] == [
        {
            "type": "function_call",
            "call_id": "call_weather",
            "name": "lookup_weather",
            "arguments": '{"city":"Chicago"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_weather",
            "output": '{"temperature":72}',
        },
    ]
    assert request["tools"] == [
        {"type": "function", **function},
        {"type": "web_search"},
    ]
    assert request["tool_choice"] == {"type": "function", "name": "lookup_weather"}


def test_chat_json_schema_format_becomes_native_text_format(client_calls):
    schema = {
        "name": "Answer",
        "schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
        "strict": True,
    }
    make_lm()("hello", response_format={"type": "json_schema", "json_schema": schema})
    assert client_calls.sync[0]["text"]["format"] == {"type": "json_schema", **schema}


def test_provider_refusal_is_reported_instead_of_returning_empty_output(client_calls):
    from openai_codex_auth import CodexError

    client_calls.response = make_response(
        output=[
            {
                "type": "message",
                "content": [
                    {"type": "refusal", "refusal": "Cannot fulfill this request."}
                ],
            }
        ]
    )
    with pytest.raises(CodexError, match="Codex refused the request: Cannot fulfill"):
        make_lm()("hello")


@pytest.mark.parametrize(
    "key",
    [
        "model_type",
        "use_developer_role",
        "transport",
        "connect_timeout",
        "idle_timeout",
        "max_retries",
        "num_retries",
        "request_timeout",
        "stream_timeout",
        "read_timeout",
        "write_timeout",
        "pool_timeout",
        "retry_strategy",
    ],
)
def test_unsupported_client_options_are_rejected_before_client(client_calls, key):
    with pytest.raises(ValueError, match="Unsupported Codex LM option"):
        make_lm().forward("hello", **{key: 1})
    assert not client_calls.constructors


def test_real_cache_separates_explicit_credentials_and_api_bases(client_calls):
    prompt = f"cache credentials {time.time_ns()}"
    make_lm(cache=True, api_key="synthetic-one")(prompt)
    make_lm(cache=True, api_key="synthetic-two")(prompt)
    make_lm(
        cache=True, api_key="synthetic-one", api_base="https://example.invalid/codex"
    )(prompt)
    make_lm(cache=True, api_key="synthetic-one")(prompt)
    assert len(client_calls.sync) == 3


def test_real_cache_tracks_account_changes_in_the_same_auth_file(
    tmp_path, client_calls
):
    storage = make_auth_storage(tmp_path)
    lm = dspy_codex_auth.LM("codex/gpt-5.5", auth_storage=storage, cache=True)
    prompt = f"cache account {time.time_ns()}"
    lm(prompt)
    data = json.loads(storage.path.read_text())
    data["tokens"]["account_id"] = "acct_changed"
    storage.path.write_text(json.dumps(data))
    lm(prompt)
    lm(prompt)
    assert len(client_calls.sync) == 2


def test_explicit_codex_provider_preserves_bare_model_id(client_calls):
    dspy_codex_auth.LM(
        "gpt-5.5", auth_provider="codex", api_key="synthetic", cache=False
    )("hello")
    assert client_calls.sync[0]["model"] == "gpt-5.5"


def test_codex_api_base_selects_responses_interface(client_calls):
    lm = dspy_codex_auth.LM(
        "openai/gpt-5.5",
        api_base=dspy_codex_auth.DEFAULT_CODEX_API_BASE,
        api_key="synthetic",
        cache=False,
    )
    assert lm.model_type == "responses"
    assert lm("hello") == [{"text": "ok"}]


def test_codex_route_rejects_non_responses_model_type():
    with pytest.raises(ValueError, match="require model_type='responses'"):
        make_lm(model_type="chat")
