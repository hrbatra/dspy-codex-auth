"""Canonical engines retain their connection when reused outside their LM."""

import asyncio

import dspy
import pytest
from openai_codex_auth import CodexClient
from test_lm import (
    block_builtin_provider,  # noqa: F401 -- imported pytest fixture
    client_calls,  # noqa: F401 -- imported pytest fixture
    make_lm,
    make_response,
    request_for,
)


# contract: C1, C3
@pytest.mark.parametrize("asynchronous", [False, True])
def test_borrowed_engines_keep_connection_defaults_and_effective_idle_timeout(
    client_calls, asynchronous  # noqa: F811 -- imported pytest fixture
):
    original = make_lm(
        api_key="borrowed-key",
        chatgpt_account_id="borrowed-account",
        codex_transport="websocket",
        codex_websocket_connect_timeout=17,
        codex_websocket_idle_timeout=23,
        timeout=61,
    )
    borrower = dspy.LM(
        original.model,
        engine=original.engine,
        async_engine=original.async_engine,
        cache=False,
        num_retries=0,
    )
    request = request_for(borrower, "borrowed request")
    result = asyncio.run(borrower.acall(request)) if asynchronous else borrower(request)
    assert result.text == "ok"
    calls = client_calls.async_ if asynchronous else client_calls.sync
    assert len(calls) == 1
    assert (
        calls[0]["transport"],
        calls[0]["connect_timeout"],
        calls[0]["idle_timeout"],
        calls[0]["timeout"],
    ) == ("websocket", 17, 61, 61)
    assert client_calls.clients[0].api_key == "borrowed-key"
    assert client_calls.clients[0].account_id == "borrowed-account"


# contract: C1, C3
def test_nested_direct_engine_call_keeps_each_lms_connection(monkeypatch):
    outer = make_lm(api_key="outer-default", codex_transport="auto", timeout=41)
    inner = make_lm(
        api_key="inner-default",
        codex_transport="websocket",
        codex_websocket_connect_timeout=19,
        timeout=97,
    )
    captures = []
    inner_results = []

    def create(client, **request):
        captures.append(
            (
                client.api_key,
                request["transport"],
                request["connect_timeout"],
                request["idle_timeout"],
                request["timeout"],
            )
        )
        if len(captures) == 1:
            inner_results.append(inner.engine.complete(request_for(inner, "inner")))
        return make_response()

    monkeypatch.setattr(CodexClient, "create", create)
    assert outer(
        "outer",
        api_key="outer-override",
        codex_transport="http",
        timeout=73,
    ) == [{"text": "ok"}]
    assert [result.text for result in inner_results] == ["ok"]
    assert captures == [
        ("outer-override", "http", 10, 73, 73),
        ("inner-default", "websocket", 19, 97, 97),
    ]
    assert outer("outer defaults") == [{"text": "ok"}]
    assert captures[-1] == ("outer-default", "auto", 10, 41, 41)
