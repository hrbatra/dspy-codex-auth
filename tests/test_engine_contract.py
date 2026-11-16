"""Supplemental public-boundary cases from the DSPy 3.5 engine contract."""

from __future__ import annotations

import asyncio
import time

import dspy
import httpx
import pytest
from dspy import lm15
from openai_codex_auth import CodexEmptyResponseError, CodexError, CodexHTTPError
from openai_codex_auth.responses_websocket import (
    CodexWebSocketConnectionError,
    CodexWebSocketProtocolError,
    CodexWebSocketResponseError,
    CodexWebSocketTimeoutError,
)

from test_lm import (
    block_builtin_provider,  # noqa: F401 -- imported autouse fixture
    client_calls,  # noqa: F401 -- imported provider boundary fixture
    make_lm,
    make_response,
    request_for,
)


class UnknownCodexError(CodexError):
    """An unclassified future public provider error."""


def websocket_error(status):
    return CodexWebSocketResponseError(
        status_code=status,
        message="provider failure",
        error_type=None,
        param=None,
        code=None,
    )


# contract: C2
def test_native_options_merge_with_canonical_schema_and_system(client_calls):
    lm = make_lm()
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    request = request_for(
        lm,
        system="Use the supplied facts.",
        config=lm15.Config(
            response_format={
                "type": "json_schema",
                "name": "Answer",
                "schema": schema,
                "strict": True,
            },
            extensions={
                "text": {"verbosity": "low"},
                "instructions": "Keep the answer concise.",
            },
        ),
    )
    assert lm(request).text == "ok"
    native = client_calls.sync[0]
    assert native["text"]["verbosity"] == "low"
    assert native["text"]["format"]["schema"] == schema
    assert native["text"]["format"]["strict"] is True
    assert "Use the supplied facts." in native["instructions"]
    assert "Keep the answer concise." in native["instructions"]


# contract: C2
def test_reasoning_summary_alone_reaches_native_request(client_calls):
    assert make_lm()("Reflect.", reasoning_summary="detailed") == [{"text": "ok"}]
    assert client_calls.sync[0]["reasoning"]["summary"] == "detailed"
    assert "reasoning_summary" not in client_calls.sync[0]


# contract: C2
def test_removed_output_cap_and_codec_seed_adaptations_are_visible(client_calls):
    lm = make_lm()
    response = lm(request_for(lm, config=lm15.Config(max_tokens=1234, seed=7)))
    assert response.text == "ok"
    cap = [a for a in response.adaptations if a.asked == 1234]
    seed = [a for a in response.adaptations if a.asked == 7]
    assert len(cap) == len(seed) == 1
    assert cap[0].reason and cap[0].applied is None
    assert seed[0].asked == 7 and seed[0].applied is None and seed[0].reason


# contract: C7
@pytest.mark.parametrize("asynchronous", [False, True])
def test_stop_is_rejected_before_provider_io(client_calls, asynchronous):
    lm = make_lm()
    request = request_for(lm, config=lm15.Config(stop=("END",)))
    with pytest.raises(dspy.LMUnsupportedFeatureError):
        if asynchronous:
            asyncio.run(lm.acall(request))
        else:
            lm(request)
    assert client_calls.sync == client_calls.async_ == []


# contract: C7
@pytest.mark.parametrize("options", [{"text": 3}, {"instructions": []}])
def test_invalid_native_shapes_are_visible_before_io(client_calls, options):
    with pytest.raises((TypeError, ValueError, dspy.LMInvalidRequestError)):
        make_lm()("hello", **options)
    assert client_calls.sync == []


# contract: C3, C7
@pytest.mark.parametrize("value", ["30", {"read": 30}])
def test_nonnumeric_timeout_is_rejected_at_construction_and_call(client_calls, value):
    with pytest.raises((TypeError, ValueError)):
        make_lm(timeout=value)
    with pytest.raises((TypeError, ValueError)):
        make_lm()("hello", timeout=value)
    assert client_calls.sync == []


# contract: C3
def test_structured_http_timeout_preserves_disabled_phases(client_calls):
    timeout = httpx.Timeout(connect=None, read=41, write=43, pool=None)
    lm = make_lm(codex_transport="http", timeout=timeout)
    assert lm("first") == [{"text": "ok"}]
    assert lm("override", timeout=53) == [{"text": "ok"}]
    assert lm("original") == [{"text": "ok"}]
    assert client_calls.sync[0]["timeout"].as_dict() == {
        "connect": None,
        "read": 41,
        "write": 43,
        "pool": None,
    }
    assert client_calls.sync[1]["timeout"] == 53
    assert client_calls.sync[2]["timeout"].as_dict() == timeout.as_dict()
    assert timeout.as_dict() == {"connect": None, "read": 41, "write": 43, "pool": None}


# contract: C5
@pytest.mark.parametrize(
    "error, expected, retryable",
    [
        (websocket_error(403), dspy.LMAuthError, False),
        (websocket_error(408), dspy.LMTimeoutError, True),
        (websocket_error(504), dspy.LMTimeoutError, True),
        (websocket_error(None), dspy.LMError, False),
        (CodexWebSocketProtocolError("invalid frame"), dspy.LMError, False),
        (CodexWebSocketConnectionError("disconnected"), dspy.LMTransportError, True),
        (CodexWebSocketTimeoutError("late"), dspy.LMTimeoutError, True),
        (CodexHTTPError(403, "forbidden"), dspy.LMAuthError, False),
        (CodexHTTPError(408, "late"), dspy.LMTimeoutError, True),
        (CodexHTTPError(504, "unavailable"), dspy.LMTimeoutError, True),
        (CodexError("HTTP attempts exhausted"), dspy.LMTransportError, True),
        (CodexEmptyResponseError("no response"), dspy.LMError, True),
        (UnknownCodexError("unclassified"), dspy.LMError, False),
    ],
)
def test_typed_errors_are_public_and_retry_only_when_classified(
    client_calls, error, expected, retryable
):
    client_calls.effects = [error]
    with pytest.raises(expected):
        make_lm(num_retries=0)("one attempt")
    assert len(client_calls.sync) == 1

    client_calls.effects = [error, make_response("recovered")]
    lm = make_lm(num_retries=1, cache=True)
    prompt = f"classified-error-{time.time_ns()}"
    if retryable:
        assert lm(prompt) == [{"text": "recovered"}]
    else:
        with pytest.raises(expected):
            lm(prompt)
        assert len(client_calls.sync) == 2
        assert lm.history == []
        assert lm(prompt) == [{"text": "recovered"}]
    assert lm(prompt) == [{"text": "recovered"}]
    assert len(client_calls.sync) == 3
    assert all(call["max_retries"] == 0 for call in client_calls.sync)


# contract: C5
def test_refusal_without_explanation_is_still_visible(client_calls):
    client_calls.response = make_response(
        output=[
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "refusal", "refusal": ""}],
            }
        ]
    )
    lm = make_lm(num_retries=2)
    with pytest.raises(dspy.LMError):
        lm("hello")
    assert len(client_calls.sync) == 1
    assert lm.history == []
