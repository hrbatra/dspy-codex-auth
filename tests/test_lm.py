"""Public DSPy 3.5 contract tests; provider boundaries are entirely offline."""

from __future__ import annotations

import asyncio
import base64
import json
import time
import warnings
from types import SimpleNamespace

import dspy
import httpx
import litellm
import pytest
from dspy import lm15
from litellm.types.llms.openai import ResponsesAPIResponse
from openai_codex_auth import (
    CodexAuth,
    CodexAuthError,
    CodexClient,
    CodexHTTPError,
    CodexProtocolError,
    CodexResponse,
)

import dspy_codex_auth


def make_auth_storage(tmp_path):
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
            "num_retries": 0,
            **kwargs,
        },
    )


def request_for(lm, text="hello", **kwargs):
    return lm15.Request(model=lm.model, messages=(lm15.Message.user(text),), **kwargs)


@pytest.fixture(autouse=True)
def block_builtin_provider(monkeypatch):
    """Wrong-route sentinel: a bypass returns observable incorrect output."""

    def complete(self, request):
        return lm15.Response(
            id="wrong-route",
            model=request.model,
            message=lm15.Message.assistant("BUILTIN_ROUTE_BYPASS"),
            finish_reason="stop",
            usage=lm15.Usage(),
        )

    async def acomplete(self, request):
        return complete(self, request)

    def responses(**kwargs):
        return ResponsesAPIResponse(
            id="wrong-route",
            object="response",
            created_at=1,
            model="gpt-5.5",
            status="completed",
            output=[
                {
                    "type": "message",
                    "id": "wrong-message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "BUILTIN_ROUTE_BYPASS",
                            "annotations": [],
                        }
                    ],
                }
            ],
        )

    async def aresponses(**kwargs):
        return responses(**kwargs)

    monkeypatch.setattr(litellm, "responses", responses)
    monkeypatch.setattr(litellm, "aresponses", aresponses)
    monkeypatch.setattr(lm15.OpenAILM, "complete", complete)
    monkeypatch.setattr(lm15.AsyncOpenAILM, "complete", acomplete)


@pytest.fixture
def client_calls(monkeypatch):
    calls = SimpleNamespace(
        sync=[], async_=[], clients=[], response=make_response(), effects=[]
    )

    def result(client, kwargs, destination):
        destination.append(kwargs)
        calls.clients.append(client)
        value = calls.effects.pop(0) if calls.effects else calls.response
        if isinstance(value, BaseException):
            raise value
        return value

    def create(self, **kwargs):
        return result(self, kwargs, calls.sync)

    async def acreate(self, **kwargs):
        await asyncio.sleep(0)
        return result(self, kwargs, calls.async_)

    monkeypatch.setattr(CodexClient, "create", create)
    monkeypatch.setattr(CodexClient, "acreate", acreate)
    return calls


# contract: C1
@pytest.mark.parametrize("asynchronous", [False, True])
def test_prompt_calls_use_codex_engine_without_deprecation(client_calls, asynchronous):
    lm = make_lm()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = asyncio.run(lm.acall("hello")) if asynchronous else lm("hello")
    assert result == [{"text": "ok"}]
    assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]
    calls = client_calls.async_ if asynchronous else client_calls.sync
    assert calls[0]["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "hello"}]}
    ]
    assert calls[0]["model"] == "gpt-5.5"


# contract: C1, C4
@pytest.mark.parametrize("asynchronous", [False, True])
def test_typed_response_preserves_text_usage_and_provider_metadata(
    client_calls, asynchronous
):
    client_calls.response = make_response(
        usage={"input_tokens": 12, "output_tokens": 7, "total_tokens": 19},
        transport="websocket",
    )
    lm = make_lm()
    request = request_for(lm)
    result = asyncio.run(lm.acall(request)) if asynchronous else lm(request)
    assert isinstance(result, lm15.Response)
    assert result.text == "ok"
    assert result.model == "gpt-5.5-actual"
    assert result.usage == lm15.Usage(input_tokens=12, output_tokens=7, total_tokens=19)
    assert result.provider_data["codex_transport"] == "websocket"
    assert result.provider_data["codex_response"] == client_calls.response.response
    assert len(lm.history) == 1


# contract: C1
@pytest.mark.parametrize("asynchronous", [False, True])
def test_typed_response_preserves_reasoning_and_function_call(
    client_calls, asynchronous
):
    client_calls.response = make_response(
        output=[
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Need weather."}],
            },
            {
                "type": "function_call",
                "id": "fc_weather",
                "call_id": "call_weather",
                "name": "lookup_weather",
                "arguments": '{"city":"Chicago"}',
            },
        ]
    )
    lm = make_lm()
    result = (
        asyncio.run(lm.acall(request_for(lm))) if asynchronous else lm(request_for(lm))
    )
    assert [p.text for p in result.message.parts_of(lm15.ThinkingPart)] == [
        "Need weather."
    ]
    assert result.tool_calls == [
        lm15.ToolCallPart(
            id="call_weather", name="lookup_weather", input={"city": "Chicago"}
        )
    ]
    assert result.usage == lm15.Usage()


# contract: C2
def test_typed_messages_media_tools_and_schema_map_to_responses(client_calls):
    lm = make_lm()
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    request = lm15.Request(
        model=lm.model,
        system="Be concise.",
        messages=(
            lm15.Message(
                role="user",
                parts=(
                    lm15.TextPart("question"),
                    lm15.ImagePart(url="https://example.invalid/image.png"),
                ),
            ),
            lm15.Message.assistant("example"),
            lm15.Message.user("followup"),
            lm15.Message(
                role="assistant",
                parts=(
                    lm15.ToolCallPart(
                        id="call_weather",
                        name="lookup_weather",
                        input={"city": "Chicago"},
                    ),
                ),
            ),
            lm15.Message(
                role="tool",
                parts=(
                    lm15.ToolResultPart(
                        id="call_weather",
                        content=(lm15.TextPart('{"temperature":72}'),),
                    ),
                ),
            ),
            lm15.Message(
                role="user",
                parts=(
                    lm15.DocumentPart(url="https://example.invalid/a.pdf"),
                    lm15.DocumentPart(file_id="file_distinct"),
                    lm15.DocumentPart(data="cGRm"),
                ),
            ),
        ),
        tools=(
            lm15.FunctionTool(
                name="lookup_weather",
                description="Weather",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            ),
        ),
        config=lm15.Config(
            response_format={
                "type": "json_schema",
                "name": "Answer",
                "schema": schema,
                "strict": True,
            },
            reasoning=lm15.Reasoning(effort="low", summary="detailed"),
        ),
    )
    assert lm(request).text == "ok"
    native = client_calls.sync[0]
    assert native["instructions"] == "Be concise."
    assert native["input"][:3] == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "question"},
                {
                    "type": "input_image",
                    "image_url": "https://example.invalid/image.png",
                },
            ],
        },
        {"role": "assistant", "content": [{"type": "output_text", "text": "example"}]},
        {"role": "user", "content": [{"type": "input_text", "text": "followup"}]},
    ]
    call, result = native["input"][3:5]
    assert (
        call["type"],
        call["call_id"],
        call["name"],
        json.loads(call["arguments"]),
    ) == ("function_call", "call_weather", "lookup_weather", {"city": "Chicago"})
    assert result == {
        "type": "function_call_output",
        "call_id": "call_weather",
        "output": '{"temperature":72}',
    }
    files = native["input"][5]["content"]
    assert [f["type"] for f in files] == ["input_file"] * 3
    assert files[0]["file_url"] == "https://example.invalid/a.pdf"
    assert files[1]["file_id"] == "file_distinct"
    assert files[2]["file_data"].endswith("cGRm")
    assert native["tools"][0]["name"] == "lookup_weather"
    assert native["tools"][0]["parameters"] == request.tools[0].parameters
    assert native["text"]["format"]["schema"] == schema
    assert native["reasoning"] == {"effort": "low", "summary": "detailed"}


# contract: C2
@pytest.mark.parametrize(
    "options",
    [
        {"reasoning_effort": "low", "reasoning_summary": "detailed"},
        {"model_reasoning_effort": "low", "model_reasoning_summary": "detailed"},
        {"reasoning": {"effort": "low", "summary": "detailed"}},
    ],
)
def test_plain_prompt_reasoning_aliases_and_native_options(client_calls, options):
    assert make_lm()(
        "Reflect.", service_tier="fast", text={"verbosity": "low"}, **options
    ) == [{"text": "ok"}]
    native = client_calls.sync[0]
    assert native["reasoning"] == {"effort": "low", "summary": "detailed"}
    assert native["service_tier"] == "fast"
    assert native["text"]["verbosity"] == "low"
    assert (
        not {
            "model_reasoning_effort",
            "model_reasoning_summary",
            "reasoning_effort",
            "reasoning_summary",
        }
        & native.keys()
    )


# contract: C2
def test_shared_http_client_normalizes_backend_payload(monkeypatch):
    requests = []

    def handler(request):
        requests.append(request)
        event = {"type": "response.completed", "response": make_response().response}
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=f"data: {json.dumps(event)}\n\n",
        )

    original = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original(**kwargs, transport=httpx.MockTransport(handler)),
    )
    assert make_lm(codex_transport="http")(
        "hello", max_tokens=2345, service_tier="fast"
    ) == [{"text": "ok"}]
    payload = json.loads(requests[0].content)
    assert payload["service_tier"] == "priority"
    assert payload["stream"] is True and payload["store"] is False
    assert (
        not {"max_tokens", "max_output_tokens", "api_key", "headers"} & payload.keys()
    )
    assert payload["instructions"] == dspy_codex_auth.DEFAULT_CODEX_INSTRUCTIONS


# contract: C2
def test_predict_json_and_chat_fewshot_use_codex_engine(client_calls):
    client_calls.response = make_response('{"answer":"four"}')
    with dspy.context(lm=make_lm(), adapter=dspy.JSONAdapter()):
        assert dspy.Predict("question -> answer")(question="2+2?").answer == "four"
    assert client_calls.sync[0]["text"]["format"]["type"] in {
        "json_schema",
        "json_object",
    }
    student = dspy.Predict("question -> answer")
    student.demos = [dspy.Example(question="2+2?", answer="four")]
    client_calls.response = make_response(
        "[[ ## answer ## ]]\nsix\n\n[[ ## completed ## ]]"
    )
    with dspy.context(lm=make_lm(), adapter=dspy.ChatAdapter()):
        assert student(question="3+3?").answer == "six"
    assistants = [
        m for m in client_calls.sync[-1]["input"] if m.get("role") == "assistant"
    ]
    assert assistants and "four" in assistants[0]["content"][0]["text"]
    assert all(p["type"] == "output_text" for m in assistants for p in m["content"])


# contract: C2
def test_gepa_reflection_plain_prompt_uses_codex_engine(client_calls):
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
    assert (
        client_calls.sync[0]["input"][0]["content"][0]["text"]
        == "Reflect on this trajectory."
    )


# contract: C3
def test_transport_timeout_precedence_and_no_default_mutation(client_calls):
    lm = make_lm(codex_transport="websocket", timeout=600)
    lm(
        "one",
        codex_transport="http",
        codex_websocket_connect_timeout=21,
        codex_websocket_idle_timeout=22,
    )
    lm("two", codex_websocket_idle_timeout=23, timeout=700)
    lm("three")
    lm("four", timeout=None, codex_websocket_idle_timeout=24)
    lm("five")
    assert [
        (c["transport"], c["connect_timeout"], c["idle_timeout"], c.get("timeout"))
        for c in client_calls.sync
    ] == [
        ("http", 21, 22, 600),
        ("websocket", 10, 700, 700),
        ("websocket", 10, 600, 600),
        ("websocket", 10, 24, None),
        ("websocket", 10, 600, 600),
    ]


# contract: C3
def test_concurrent_async_overrides_are_isolated(client_calls):
    lm = make_lm()

    async def run():
        return await asyncio.gather(
            lm.acall("a", api_key="key-a", codex_transport="http", timeout=31),
            lm.acall("b", api_key="key-b", codex_transport="websocket", timeout=47),
        )

    assert asyncio.run(run()) == [[{"text": "ok"}], [{"text": "ok"}]]
    assert [
        (c.api_key, r["transport"], r["idle_timeout"])
        for c, r in zip(client_calls.clients, client_calls.async_)
    ] == [("key-a", "http", 31), ("key-b", "websocket", 47)]
    assert lm("after") == [{"text": "ok"}]
    assert client_calls.clients[-1].api_key == "synthetic-key"
    assert client_calls.sync[0]["transport"] == "auto"


# contract: C3
def test_connection_overrides_and_structured_http_timeout_stay_out_of_payload_and_history(
    client_calls,
):
    timeout = httpx.Timeout(600, connect=8, pool=9)
    lm = make_lm(
        headers={"Authorization": "header-secret"}, originator="custom-originator"
    )
    assert lm(
        "hello",
        api_key="per-call-secret",
        chatgpt_account_id="acct_override",
        api_base="https://example.invalid/codex",
        timeout=timeout,
        codex_transport="http",
    ) == [{"text": "ok"}]
    client, request = client_calls.clients[0], client_calls.sync[0]
    assert (client.api_key, client.account_id, client.api_base, client.originator) == (
        "per-call-secret",
        "acct_override",
        "https://example.invalid/codex",
        "custom-originator",
    )
    assert client.headers == {"Authorization": "header-secret"}
    assert client.user_agent == f"DSPy/{dspy.__version__}"
    assert request["timeout"].as_dict() == timeout.as_dict()
    assert (
        not {"api_key", "chatgpt_account_id", "api_base", "headers", "originator"}
        & request.keys()
    )
    assert all(
        secret not in repr(lm.history)
        for secret in ("per-call-secret", "header-secret", "synthetic-key")
    )


# contract: C4
@pytest.mark.parametrize(
    "overrides",
    [
        {"api_key": "different-key"},
        {"chatgpt_account_id": "different-account"},
        {"api_base": "https://example.invalid/other"},
        {"codex_transport": "http"},
        {"timeout": 61},
        {"codex_websocket_connect_timeout": 17},
        {"codex_websocket_idle_timeout": 37},
        {"headers": {"X-Test": "other"}},
        {"rollout_id": 42},
    ],
)
def test_cache_partitions_connection_context_and_restores_previous_entry(
    client_calls, overrides
):
    lm = make_lm(cache=True)
    prompt = f"cache-{time.time_ns()}"
    assert lm(prompt) == lm(prompt) == [{"text": "ok"}]
    client_calls.response = make_response("changed")
    assert lm(prompt, **overrides) == [{"text": "changed"}]
    assert lm(prompt) == [{"text": "ok"}]
    assert len(client_calls.sync) == 2


# contract: C4
def test_cache_usage_is_once_per_miss_and_sync_async_are_distinct(client_calls):
    client_calls.response = make_response(
        usage={"input_tokens": 12, "output_tokens": 7, "total_tokens": 19}
    )
    tracker = SimpleNamespace(calls=[])
    tracker.add_usage = lambda model, usage: tracker.calls.append((model, usage))
    lm = make_lm(cache=True)
    prompt = f"usage-{time.time_ns()}"
    with dspy.context(usage_tracker=tracker):
        lm(prompt)
        lm(prompt)
        asyncio.run(lm.acall(prompt))
        asyncio.run(lm.acall(prompt))
    assert len(client_calls.sync) == len(client_calls.async_) == 1
    assert len(tracker.calls) == 2
    assert all(
        model == lm.model
        and usage["prompt_tokens"] == 12
        and usage["completion_tokens"] == 7
        for model, usage in tracker.calls
    )


# contract: C4
def test_cache_observes_account_changes_in_one_auth_file(tmp_path, client_calls):
    storage = make_auth_storage(tmp_path)
    lm = dspy_codex_auth.LM(
        "codex/gpt-5.5", auth_storage=storage, cache=True, num_retries=0
    )
    prompt = f"identity-{time.time_ns()}"
    assert lm(prompt) == [{"text": "ok"}]
    content = json.loads(storage.path.read_text())
    content["tokens"]["account_id"] = "acct_changed"
    storage.path.write_text(json.dumps(content))
    client_calls.response = make_response("new account")
    assert lm(prompt) == lm(prompt) == [{"text": "new account"}]
    assert len(client_calls.sync) == 2


# contract: C5
@pytest.mark.parametrize(
    "error",
    [
        CodexHTTPError(429, {"error": {"message": "slow down"}}),
        CodexHTTPError(503, "unavailable"),
        httpx.ConnectError("offline"),
        httpx.ReadTimeout("late"),
    ],
)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_retryable_errors_have_one_retry_owner_and_cache_only_success(
    client_calls, error, asynchronous
):
    client_calls.effects = [error, make_response("recovered")]
    lm = make_lm(num_retries=1, cache=True)
    prompt = f"retry-{time.time_ns()}"

    def call():
        return asyncio.run(lm.acall(prompt)) if asynchronous else lm(prompt)

    assert call() == call() == [{"text": "recovered"}]
    attempts = client_calls.async_ if asynchronous else client_calls.sync
    assert len(attempts) == 2
    assert all(a["max_retries"] == 0 for a in attempts)
    assert len(lm.history) == 2


# contract: C5
@pytest.mark.parametrize(
    "error, expected",
    [
        (CodexHTTPError(401, "unauthorized"), dspy.LMAuthError),
        (CodexAuthError("codex login required"), dspy.LMAuthError),
        (CodexHTTPError(400, "bad request"), dspy.LMInvalidRequestError),
        (CodexProtocolError("invalid event"), dspy.LMError),
    ],
)
def test_nonretryable_errors_remain_visible_and_do_not_poison_cache(
    client_calls, error, expected
):
    client_calls.effects = [error]
    lm = make_lm(num_retries=2, cache=True)
    prompt = f"failure-{time.time_ns()}"
    with pytest.raises(expected):
        lm(prompt)
    assert len(client_calls.sync) == 1
    assert lm.history == []
    assert lm(prompt) == [{"text": "ok"}]
    assert len(client_calls.sync) == 2


# contract: C5
def test_zero_retries_and_refusal_are_not_empty_successes(client_calls):
    client_calls.effects = [CodexHTTPError(503, "unavailable")]
    lm = make_lm(num_retries=0)
    with pytest.raises(dspy.LMError):
        lm("hello")
    assert len(client_calls.sync) == 1
    client_calls.response = make_response(
        output=[
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "refusal", "refusal": "Cannot fulfill this request."}
                ],
            }
        ]
    )
    with pytest.raises(dspy.LMError, match="Cannot fulfill"):
        make_lm(num_retries=2)("hello")
    assert len(client_calls.sync) == 2


# contract: C5
def test_async_cancellation_is_not_cached(client_calls):
    client_calls.effects = [asyncio.CancelledError()]
    lm = make_lm(cache=True)
    prompt = f"cancel-{time.time_ns()}"
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(lm.acall(prompt))
    assert lm.history == []
    assert asyncio.run(lm.acall(prompt)) == [{"text": "ok"}]
    assert len(client_calls.async_) == 2


# contract: C6
@pytest.mark.parametrize(
    "model, options",
    [
        ("codex/gpt-5.5", {}),
        ("chatgpt/gpt-5.5", {}),
        ("openai-codex/gpt-5.5", {}),
        ("gpt-5.5", {"auth_provider": "codex"}),
        ("openai/gpt-5.5", {"api_base": dspy_codex_auth.DEFAULT_CODEX_API_BASE}),
    ],
)
def test_install_routes_aliases_and_uninstall_restores_behavior(
    tmp_path, client_calls, model, options
):
    original = dspy.LM
    try:
        dspy_codex_auth.install(auth_storage=tmp_path / "missing.json")
        lm = dspy.LM(model, api_key="synthetic", cache=False, num_retries=0, **options)
        assert lm("hello") == [{"text": "ok"}]
        assert client_calls.sync[0]["model"] == "gpt-5.5"
    finally:
        dspy_codex_auth.uninstall()
    assert dspy.LM == original
    assert not hasattr(dspy, "getauthtoken")


# contract: C6
def test_missing_auth_is_lazy_and_copy_and_state_preserve_configuration(
    tmp_path, client_calls
):
    missing = tmp_path / "missing.json"
    lazy = dspy_codex_auth.LM("codex/gpt-5.5", auth_storage=missing)
    assert lazy.model == "openai/gpt-5.5" and not missing.exists()
    lm = make_lm(
        auth_storage=missing,
        codex_transport="http",
        headers={"Authorization": "secret-header"},
    )
    assert lm("first") == [{"text": "ok"}]
    copied = lm.copy(api_key="copy-secret", codex_transport="websocket")
    assert copied.history == []
    assert copied("second") == [{"text": "ok"}]
    assert len(lm.history) == len(copied.history) == 1
    assert client_calls.clients[-1].api_key == "copy-secret"
    assert client_calls.sync[-1]["transport"] == "websocket"
    state = copied.dump_state()
    assert all(
        secret not in repr(state)
        for secret in ("synthetic-key", "copy-secret", "secret-header", "Authorization")
    )
    restored = dspy_codex_auth.LM.load_state(state, allow_custom_lm_class=True)
    assert restored("third", api_key="restored-key") == [{"text": "ok"}]
    assert client_calls.clients[-1].auth.path == missing
    assert client_calls.sync[-1]["transport"] == "websocket"


# contract: C6
def test_non_codex_route_preserves_custom_engines(client_calls):
    class Echo:
        def complete(self, request):
            return lm15.Response(
                id=None,
                model=request.model,
                message=lm15.Message.assistant("echo:" + request.messages[0].text),
                finish_reason="stop",
                usage=lm15.Usage(),
            )

    class AsyncEcho:
        async def complete(self, request):
            return Echo().complete(request)

    lm = dspy_codex_auth.LM(
        "custom/echo", engine=Echo(), async_engine=AsyncEcho(), cache=False
    )
    assert lm("sync") == ["echo:sync"]
    assert asyncio.run(lm.acall("async")) == ["echo:async"]
    assert not client_calls.sync and not client_calls.async_


# contract: C7
def test_removed_messages_and_forward_shapes_fail_with_migration_guidance(client_calls):
    lm = make_lm()
    with pytest.raises(TypeError, match="Request|3.5"):
        lm(messages=[{"role": "user", "content": "hello"}])
    with pytest.raises(TypeError, match="Request|3.5"):
        asyncio.run(lm.acall(messages=[{"role": "user", "content": "hello"}]))
    assert not hasattr(lm, "forward") and not hasattr(lm, "aforward")
    assert not client_calls.sync and not client_calls.async_


# contract: C7
@pytest.mark.parametrize(
    "option, value",
    [
        ("codex_transport", "invalid"),
        ("codex_transport", None),
        ("codex_websocket_connect_timeout", 0),
        ("codex_websocket_idle_timeout", -1),
        ("codex_websocket_idle_timeout", float("nan")),
        ("codex_websocket_connect_timeout", float("inf")),
        ("codex_websocket_idle_timeout", True),
    ],
)
def test_invalid_constructor_transport_options_fail_before_io(
    client_calls, option, value
):
    with pytest.raises((ValueError, TypeError)):
        make_lm(**{option: value})
    assert not client_calls.sync


# contract: C7
@pytest.mark.parametrize(
    "options",
    [
        {"codex_transport": "invalid"},
        {"codex_websocket_connect_timeout": 0},
        {"codex_websocket_idle_timeout": float("nan")},
        {"max_retries": 1},
        {"request_timeout": 5},
        {"transport": "http"},
        {"engine": "auto"},
    ],
)
def test_invalid_call_options_fail_before_io(client_calls, options):
    with pytest.raises((ValueError, TypeError)):
        make_lm()("hello", **options)
    assert not client_calls.sync


# contract: C7
@pytest.mark.parametrize(
    "options", [{"codex_transport": "websocket"}, {"codex_websocket_idle_timeout": 22}]
)
def test_codex_options_are_rejected_on_non_codex_routes(client_calls, options):
    with pytest.raises((ValueError, TypeError)):
        dspy_codex_auth.LM("openai/test", **options)
    lm = dspy_codex_auth.LM("openai/test", api_key="synthetic", cache=False)
    with pytest.raises((ValueError, TypeError)):
        lm("hello", **options)
    with pytest.raises((ValueError, TypeError)):
        asyncio.run(lm.acall("hello", **options))
    assert not client_calls.sync and not client_calls.async_


# contract: C7
@pytest.mark.parametrize(
    "options",
    [
        {"engine": "auto"},
        {"async_engine": object()},
        {"max_retries": 1},
        {"request_timeout": 4},
        {"model_type": "chat"},
    ],
)
def test_codex_engine_and_client_only_constructor_options_are_rejected(
    client_calls, options
):
    with pytest.raises((ValueError, TypeError)):
        make_lm(**options)
    assert not client_calls.sync


# contract: C4
def test_uncached_http_calls_read_updated_file_credentials(tmp_path, monkeypatch):
    storage = make_auth_storage(tmp_path)
    requests = []

    def handler(request):
        requests.append(request)
        event = {"type": "response.completed", "response": make_response().response}
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=f"data: {json.dumps(event)}\n\n",
        )

    original = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original(**kwargs, transport=httpx.MockTransport(handler)),
    )
    lm = dspy_codex_auth.LM(
        "codex/gpt-5.5",
        auth_storage=storage,
        codex_transport="http",
        cache=False,
        num_retries=0,
    )
    assert lm("first") == [{"text": "ok"}]
    content = json.loads(storage.path.read_text())
    first_token = content["tokens"]["access_token"]
    second_token = first_token.removesuffix("signature") + "rotated-signature"
    content["tokens"]["access_token"] = second_token
    content["tokens"]["account_id"] = "acct_rotated"
    storage.path.write_text(json.dumps(content))
    assert lm("second") == [{"text": "ok"}]
    assert [r.headers["authorization"] for r in requests] == [
        f"Bearer {first_token}",
        f"Bearer {second_token}",
    ]
    assert [r.headers["chatgpt-account-id"] for r in requests] == [
        "acct_test",
        "acct_rotated",
    ]


# contract: C2
def test_gepa_compile_uses_codex_engine(monkeypatch):
    from dspy.teleprompt import GEPA

    captured = []

    def create(self, **kwargs):
        captured.append(kwargs["input"])
        text = "\n".join(
            part.get("text", "")
            for item in kwargs["input"]
            for part in item.get("content", [])
        )
        if "Your task is to write a new instruction" in text:
            answer = "```Given the fields `question`, produce the fields `answer`. Return exactly the expected answer.```"
        else:
            answer = "[[ ## answer ## ]]\n4\n\n[[ ## completed ## ]]"
        return make_response(answer)

    monkeypatch.setattr(CodexClient, "create", create)
    lm = make_lm()
    student = dspy.Predict("question -> answer")
    trainset = [dspy.Example(question="2+2?", answer="4").with_inputs("question")]

    def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
        return 1.0 if prediction.answer == example.answer else 0.0

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
    assert captured


# contract: C2
def test_pydantic_schema_on_string_call_reaches_native_format(client_calls):
    from pydantic import BaseModel

    class Answer(BaseModel):
        answer: str

    assert make_lm()("hello", response_format=Answer) == [{"text": "ok"}]
    output_format = client_calls.sync[0]["text"]["format"]
    assert output_format["type"] == "json_schema"
    assert output_format["schema"] == {
        **Answer.model_json_schema(),
        "additionalProperties": False,
    }
    assert output_format["strict"] is True


# contract: C6
@pytest.mark.parametrize("via_program", [False, True])
def test_deepcopy_isolates_lm_history_kwargs_and_callbacks(client_calls, via_program):
    import copy

    from dspy.utils.callback import BaseCallback

    lm = make_lm(text={"verbosity": "low"}, callbacks=[BaseCallback()])
    lm("before copy")
    if via_program:
        program = dspy.Predict("question -> answer")
        program.set_lm(lm)
        copied = program.deepcopy().get_lm()
    else:
        copied = copy.deepcopy(lm)
    copied.history.append({"test_copy_only": True})
    copied.kwargs["text"]["verbosity"] = "high"
    copied.callbacks.append(BaseCallback())
    assert len(lm.history) == 1
    assert lm.kwargs["text"] == {"verbosity": "low"}
    assert len(lm.callbacks) == 1
    assert copied("after copy") == [{"text": "ok"}]
    assert client_calls.sync[-1]["text"]["verbosity"] == "high"
    assert len(lm.history) == 1


# contract: C6
def test_attached_lm_full_program_save_load_preserves_execution(tmp_path, client_calls):
    client_calls.response = make_response('{"answer":"restored"}')
    storage = make_auth_storage(tmp_path)
    lm = make_lm(auth_storage=storage, codex_transport="http")
    program = dspy.Predict("question -> answer")
    program.set_lm(lm)
    destination = tmp_path / "saved_program"
    program.save(destination, save_program=True)
    loaded = dspy.load(destination, allow_pickle=True)
    with dspy.context(adapter=dspy.JSONAdapter()):
        assert loaded(question="What survived?").answer == "restored"
    assert loaded.get_lm().model == "openai/gpt-5.5"
    assert client_calls.sync[-1]["transport"] == "http"
    assert client_calls.clients[-1].auth.path == tmp_path / "auth.json"


# contract: C6
def test_prompt_cache_and_structured_timeout_state_are_json_reconstructible(
    tmp_path, client_calls
):
    timeout = httpx.Timeout(connect=None, read=51, write=13, pool=7)
    cache_config = lm15.CacheConfig(mode="off")
    lm = make_lm(
        auth_storage=tmp_path / "auth.json",
        codex_transport="http",
        timeout=timeout,
        prompt_cache=cache_config,
    )
    state = json.loads(json.dumps(lm.dump_state()))
    restored = dspy_codex_auth.LM.load_state(state, allow_custom_lm_class=True)
    assert restored("restored", api_key="restore-only-key") == [{"text": "ok"}]
    assert restored.dump_state()["prompt_cache"] == {"mode": "off"}
    assert client_calls.sync[-1]["timeout"].as_dict() == {
        "connect": None,
        "read": 51,
        "write": 13,
        "pool": 7,
    }
    assert client_calls.clients[-1].auth.path == tmp_path / "auth.json"


# contract: C6
def test_custom_alias_registration_and_removal_are_observable(tmp_path, client_calls):
    auth = CodexAuth(tmp_path / "missing.json")

    def resolver(model, kwargs, auth_storage):
        return "openai/gpt-5.5", {
            **kwargs,
            "api_base": dspy_codex_auth.DEFAULT_CODEX_API_BASE,
        }

    dspy_codex_auth.register_model_alias(["test-alias-one", "test-alias-two"], resolver)
    try:
        lm = dspy_codex_auth.LM(
            "test-alias-one/example",
            auth_storage=auth,
            api_key="synthetic",
            cache=False,
            num_retries=0,
        )
        assert lm("hello") == [{"text": "ok"}]
        assert client_calls.sync[-1]["model"] == "gpt-5.5"
        assert (
            dspy_codex_auth.resolve_lm_route(
                "test-alias-two/example", auth_storage=auth
            )[0]
            == "openai/gpt-5.5"
        )
    finally:
        dspy_codex_auth.unregister_model_alias("test-alias-one")
        dspy_codex_auth.unregister_model_alias("test-alias-two")
    assert dspy_codex_auth.resolve_lm_route(
        "test-alias-one/example", auth_storage=auth
    ) == ("test-alias-one/example", {})
    with pytest.raises(ValueError):
        dspy_codex_auth.LM(
            "gpt-5.5",
            auth_provider="unregistered-provider",
            auth_storage=auth,
            api_key="synthetic",
        )


# contract: C6
def test_non_codex_copy_and_state_roundtrip_preserve_dspy_behavior():
    lm = dspy_codex_auth.LM(
        "openai/gpt-4.1", api_key="synthetic", cache=False, num_retries=0
    )
    copied = lm.copy(temperature=0.4)
    state = json.loads(json.dumps(copied.dump_state()))
    restored = dspy_codex_auth.LM.load_state(state, allow_custom_lm_class=True)
    assert restored.model == lm.model
    result = restored(
        lm15.Request(model=restored.model, messages=(lm15.Message.user("hello"),)),
        api_key="synthetic",
    )
    assert result.text == "BUILTIN_ROUTE_BYPASS"
    assert lm.history == [] and copied.history == []
    assert restored.kwargs["temperature"] == 0.4


# contract: C6
def test_install_without_helpers_and_repeated_uninstall(tmp_path):
    original = dspy.LM
    try:
        dspy_codex_auth.install(
            auth_storage=tmp_path / "missing.json", attach_helpers=False
        )
        assert not hasattr(dspy, "getauthtoken")
        assert isinstance(dspy.LM("codex/gpt-5.5"), dspy_codex_auth.LM)
    finally:
        dspy_codex_auth.uninstall()
        dspy_codex_auth.uninstall()
    assert dspy.LM == original
    assert not hasattr(dspy, "getauthtoken")


# contract: C7
@pytest.mark.parametrize("prompt", [42, {}, ["hello"], object()])
def test_invalid_prompt_shapes_fail_before_provider_io(client_calls, prompt):
    with pytest.raises(TypeError):
        make_lm()(prompt)
    assert not client_calls.sync


# contract: C1, C7
def test_request_keyword_works_but_duplicate_request_is_rejected(client_calls):
    lm = make_lm()
    request = request_for(lm)
    assert lm(request=request).text == "ok"
    assert asyncio.run(lm.acall(request=request)).text == "ok"
    with pytest.raises(TypeError):
        lm("duplicate", request=request)
    with pytest.raises(TypeError):
        asyncio.run(lm.acall("duplicate", request=request))
    assert len(client_calls.sync) == len(client_calls.async_) == 1


# contract: C6
def test_alias_reregistration_does_not_change_existing_lm_copy_or_saved_state(
    tmp_path, client_calls
):
    storage = make_auth_storage(tmp_path)

    def resolver_a(model, kwargs, auth_storage):
        return "openai/gpt-alias-a", {
            **kwargs,
            "api_base": dspy_codex_auth.DEFAULT_CODEX_API_BASE,
        }

    def resolver_b(model, kwargs, auth_storage):
        return "openai/gpt-alias-b", {
            **kwargs,
            "api_base": dspy_codex_auth.DEFAULT_CODEX_API_BASE,
        }

    dspy_codex_auth.register_model_alias("frozen-alias", resolver_a)
    try:
        original = dspy_codex_auth.LM(
            "frozen-alias/example",
            auth_storage=storage,
            api_key="synthetic",
            cache=False,
            num_retries=0,
        )
        dspy_codex_auth.register_model_alias("frozen-alias", resolver_b)
        copied = original.copy()
        restored = dspy_codex_auth.LM.load_state(
            json.loads(json.dumps(original.dump_state())), allow_custom_lm_class=True
        )
        newer = dspy_codex_auth.LM(
            "frozen-alias/example",
            auth_storage=storage,
            api_key="synthetic",
            cache=False,
            num_retries=0,
        )
        assert (original.model, copied.model, restored.model, newer.model) == (
            "openai/gpt-alias-a",
            "openai/gpt-alias-a",
            "openai/gpt-alias-a",
            "openai/gpt-alias-b",
        )
        assert copied("copy") == [{"text": "ok"}]
        assert restored("restore", api_key="synthetic") == [{"text": "ok"}]
        assert [call["model"] for call in client_calls.sync] == [
            "gpt-alias-a",
            "gpt-alias-a",
        ]
    finally:
        dspy_codex_auth.unregister_model_alias("frozen-alias")
