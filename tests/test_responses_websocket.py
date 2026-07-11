from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import ssl
from collections.abc import Iterator
from copy import deepcopy
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import dspy


MODULE_NAME = "dspy_codex_auth.responses_websocket"


def _module_spec():
    return importlib.util.find_spec(MODULE_NAME)


@pytest.fixture
def websocket_module() -> ModuleType:
    if _module_spec() is None:
        pytest.skip(f"{MODULE_NAME} is not implemented yet")
    return importlib.import_module(MODULE_NAME)


def _completed_event(text: str = "OK") -> dict[str, Any]:
    return {
        "type": "response.completed",
        "response": {
            "id": "resp_test",
            "created_at": 1,
            "model": "gpt-5.6-luna",
            "object": "response",
            "status": "completed",
            "output": [
                {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": text,
                            "annotations": [],
                        }
                    ],
                }
            ],
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
            },
        },
    }


class FakeConnectionClosed(Exception):
    pass


class FakeSyncConnection:
    def __init__(self, frames: list[str | bytes | BaseException]):
        self.frames: Iterator[str | bytes | BaseException] = iter(frames)
        self.sent: list[str | bytes] = []
        self.recv_timeouts: list[float | None] = []
        self.response = SimpleNamespace(status_code=101, headers={})

    def __enter__(self) -> FakeSyncConnection:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def send(self, message: str | bytes) -> None:
        self.sent.append(message)

    def recv(self, timeout: float | None = None) -> str | bytes:
        self.recv_timeouts.append(timeout)
        try:
            frame = next(self.frames)
        except StopIteration as exc:
            raise FakeConnectionClosed from exc
        if isinstance(frame, BaseException):
            raise frame
        return frame


class FakeAsyncConnection:
    def __init__(self, frames: list[str | bytes | BaseException]):
        self.frames: Iterator[str | bytes | BaseException] = iter(frames)
        self.sent: list[str | bytes] = []
        self.response = SimpleNamespace(status_code=101, headers={})

    async def __aenter__(self) -> FakeAsyncConnection:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    async def send(self, message: str | bytes) -> None:
        self.sent.append(message)

    async def recv(self) -> str | bytes:
        try:
            frame = next(self.frames)
        except StopIteration as exc:
            raise FakeConnectionClosed from exc
        if isinstance(frame, BaseException):
            raise frame
        return frame


def _request() -> dict[str, Any]:
    return {
        "model": "openai/gpt-5.6-luna",
        "instructions": "You are a helpful assistant.",
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Reply with OK."}],
            }
        ],
        "reasoning": {"effort": "high", "summary": "auto"},
        "store": False,
        "stream": True,
    }


def test_responses_websocket_module_is_discoverable():
    assert _module_spec() is not None


def test_sync_client_sends_secure_response_create_and_collects_events(
    websocket_module: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    connection = FakeSyncConnection(
        [
            json.dumps({"type": "response.created", "response": {}}),
            json.dumps(_completed_event()),
        ]
    )
    connect_call: dict[str, Any] = {}

    def fake_connect(uri: str, **kwargs: Any) -> FakeSyncConnection:
        connect_call["uri"] = uri
        connect_call.update(kwargs)
        return connection

    monkeypatch.setattr(websocket_module, "_sync_connect", fake_connect)
    monkeypatch.setattr(websocket_module, "ConnectionClosed", FakeConnectionClosed)
    headers = {
        "Authorization": "Bearer caller-value",
        "aUtHoRiZaTiOn": "Bearer mixed-case-caller-value",
        "chatgpt-account-id": "acct_test",
        "OpenAI-Beta": "responses=experimental",
        "oPeNaI-bEtA": "mixed-case-beta",
        "originator": "dspy_codex_auth",
        "ORIGINATOR": "mixed-case-originator",
        "User-Agent": "DSPy/3.2.0",
        "user-agent": "spoofed-agent",
        "x-client-request-id": "caller-request-id",
        "X-CLIENT-REQUEST-ID": "mixed-case-request-id",
        "X-Test": "preserved",
    }
    original_headers = dict(headers)
    request = _request()
    original_request = deepcopy(request)

    result = websocket_module.websocket_response(
        request,
        api_base="https://chatgpt.com/backend-api/codex",
        api_key="secret-token",
        headers=headers,
        connect_timeout=12.0,
        idle_timeout=34.0,
    )

    assert connect_call["uri"] == ("wss://chatgpt.com/backend-api/codex/responses")
    assert isinstance(connect_call["ssl"], ssl.SSLContext)
    assert connect_call["open_timeout"] == 12.0
    assert connect_call["user_agent_header"] is None
    handshake_headers = connect_call["additional_headers"]
    assert handshake_headers["Authorization"] == "Bearer secret-token"
    assert handshake_headers["chatgpt-account-id"] == "acct_test"
    assert (
        handshake_headers["OpenAI-Beta"]
        == websocket_module.DEFAULT_CODEX_WEBSOCKET_BETA
    )
    assert (
        handshake_headers["originator"]
        == websocket_module.DEFAULT_CODEX_WEBSOCKET_ORIGINATOR
    )
    assert handshake_headers["User-Agent"] == f"DSPy/{dspy.__version__}"
    assert sum(key.lower() == "user-agent" for key in handshake_headers) == 1
    assert sum(key.lower() == "authorization" for key in handshake_headers) == 1
    assert sum(key.lower() == "openai-beta" for key in handshake_headers) == 1
    assert sum(key.lower() == "originator" for key in handshake_headers) == 1
    assert sum(key.lower() == "x-client-request-id" for key in handshake_headers) == 1
    assert handshake_headers["X-Test"] == "preserved"
    assert handshake_headers["x-client-request-id"] != "caller-request-id"
    assert handshake_headers["session-id"]
    assert handshake_headers["thread-id"]

    assert len(connection.sent) == 1
    assert isinstance(connection.sent[0], str)
    wire_request = json.loads(connection.sent[0])
    assert wire_request["type"] == "response.create"
    assert wire_request["model"] == "gpt-5.6-luna"
    assert "secret-token" not in connection.sent[0]
    assert "api_key" not in wire_request
    assert connection.recv_timeouts == [34.0, 34.0]
    assert request == original_request
    assert headers == original_headers
    assert [event["type"] for event in result.events] == [
        "response.created",
        "response.completed",
    ]
    assert result.response["output"][0]["content"][0]["text"] == "OK"


def test_async_client_uses_matching_protocol(
    websocket_module: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    connection = FakeAsyncConnection([json.dumps(_completed_event("async OK"))])
    connect_call: dict[str, Any] = {}

    def fake_connect(uri: str, **kwargs: Any) -> FakeAsyncConnection:
        connect_call["uri"] = uri
        connect_call.update(kwargs)
        return connection

    monkeypatch.setattr(websocket_module, "_async_connect", fake_connect)
    monkeypatch.setattr(websocket_module, "ConnectionClosed", FakeConnectionClosed)

    result = asyncio.run(
        websocket_module.awebsocket_response(
            _request(),
            api_base="https://chatgpt.com/backend-api/codex/",
            api_key="secret-token",
            headers={
                "chatgpt-account-id": "acct_test",
                "User-Agent": "DSPy/3.2.0",
            },
            connect_timeout=13.0,
            idle_timeout=35.0,
        )
    )

    assert connect_call["uri"] == ("wss://chatgpt.com/backend-api/codex/responses")
    assert connect_call["open_timeout"] == 13.0
    assert json.loads(connection.sent[0])["type"] == "response.create"
    assert result.response["output"][0]["content"][0]["text"] == "async OK"


@pytest.mark.parametrize(
    ("api_base", "expected"),
    [
        (
            "https://chatgpt.com/backend-api/codex",
            "wss://chatgpt.com/backend-api/codex/responses",
        ),
        ("http://localhost:8000/codex/", "ws://localhost:8000/codex/responses"),
    ],
)
def test_codex_websocket_url_converts_http_schemes(
    websocket_module: ModuleType, api_base: str, expected: str
):
    assert websocket_module.codex_websocket_url(api_base) == expected


def test_codex_websocket_url_rejects_non_http_scheme(
    websocket_module: ModuleType,
):
    with pytest.raises(ValueError, match="HTTP or HTTPS"):
        websocket_module.codex_websocket_url("ftp://example.com/codex")


def test_client_rejects_api_key_in_data_frame(
    websocket_module: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    called = False

    def fake_connect(*_args: Any, **_kwargs: Any) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(websocket_module, "_sync_connect", fake_connect)
    request = {**_request(), "api_key": "must-not-be-sent"}

    with pytest.raises(ValueError, match="api_key"):
        websocket_module.websocket_response(
            request,
            api_base="https://chatgpt.com/backend-api/codex",
            api_key="secret-token",
            headers={"User-Agent": "DSPy/3.2.0"},
        )

    assert called is False


def test_client_rejects_nested_bearer_token_in_data_frame(
    websocket_module: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    called = False

    def fake_connect(*_args: Any, **_kwargs: Any) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(websocket_module, "_sync_connect", fake_connect)
    request = _request()
    request["input"][0]["content"][0]["text"] = "leaked secret-token"

    with pytest.raises(ValueError, match="bearer credential") as caught:
        websocket_module.websocket_response(
            request,
            api_base="https://chatgpt.com/backend-api/codex",
            api_key="secret-token",
            headers={"User-Agent": "DSPy/3.2.0"},
        )

    assert "secret-token" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert called is False


def test_server_error_is_typed_structured_and_token_redacted(
    websocket_module: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    event = {
        "type": "error",
        "status": 404,
        "error": {
            "message": "Model not found secret-token",
            "type": "invalid_request_error",
            "param": "model",
            "code": {"unsafe": "secret-token-code"},
        },
    }
    connection = FakeSyncConnection([json.dumps(event)])
    monkeypatch.setattr(websocket_module, "_sync_connect", lambda *_a, **_k: connection)

    with pytest.raises(websocket_module.CodexWebSocketResponseError) as caught:
        websocket_module.websocket_response(
            _request(),
            api_base="https://chatgpt.com/backend-api/codex",
            api_key="secret-token",
            headers={"User-Agent": "DSPy/3.2.0"},
        )

    error = caught.value
    assert error.status_code == 404
    assert error.error_type == "invalid_request_error"
    assert error.param == "model"
    assert error.code == "{'unsafe': '<redacted>-code'}"
    assert "secret-token" not in str(error)
    assert error.__cause__ is None
    assert error.__context__ is None


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        ("not json", "valid JSON"),
        (b"binary", "binary"),
        (json.dumps({"type": "response.completed", "response": None}), "response"),
        (json.dumps({"type": "response.completed", "response": {}}), "status"),
        (
            json.dumps(
                {
                    "type": "response.completed",
                    "response": {
                        "status": "failed",
                        "model": "gpt-5.6-luna",
                        "output": [],
                    },
                }
            ),
            "status",
        ),
        (
            json.dumps(
                {
                    "type": "response.completed",
                    "response": {
                        "status": "completed",
                        "model": "gpt-5.6-luna",
                        "output": {},
                    },
                }
            ),
            "output",
        ),
    ],
)
def test_invalid_frames_raise_protocol_error(
    websocket_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    frame: str | bytes,
    message: str,
):
    connection = FakeSyncConnection([frame])
    monkeypatch.setattr(websocket_module, "_sync_connect", lambda *_a, **_k: connection)

    with pytest.raises(websocket_module.CodexWebSocketProtocolError, match=message):
        websocket_module.websocket_response(
            _request(),
            api_base="https://chatgpt.com/backend-api/codex",
            api_key="secret-token",
            headers={"User-Agent": "DSPy/3.2.0"},
        )


def test_malformed_frame_does_not_survive_in_exception_graph(
    websocket_module: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    connection = FakeSyncConnection(["not-json secret-token"])
    monkeypatch.setattr(websocket_module, "_sync_connect", lambda *_a, **_k: connection)

    with pytest.raises(websocket_module.CodexWebSocketProtocolError) as caught:
        websocket_module.websocket_response(
            _request(),
            api_base="https://chatgpt.com/backend-api/codex",
            api_key="secret-token",
            headers={"User-Agent": "DSPy/3.2.0"},
        )

    assert "secret-token" not in repr(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_close_before_completed_raises_protocol_error(
    websocket_module: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    connection = FakeSyncConnection([])
    monkeypatch.setattr(websocket_module, "_sync_connect", lambda *_a, **_k: connection)
    monkeypatch.setattr(websocket_module, "ConnectionClosed", FakeConnectionClosed)

    with pytest.raises(
        websocket_module.CodexWebSocketProtocolError,
        match="before response.completed",
    ):
        websocket_module.websocket_response(
            _request(),
            api_base="https://chatgpt.com/backend-api/codex",
            api_key="secret-token",
            headers={"User-Agent": "DSPy/3.2.0"},
        )


def test_sync_idle_timeout_raises_typed_timeout(
    websocket_module: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    connection = FakeSyncConnection([TimeoutError()])
    monkeypatch.setattr(websocket_module, "_sync_connect", lambda *_a, **_k: connection)

    with pytest.raises(
        websocket_module.CodexWebSocketTimeoutError,
        match="30.0 seconds",
    ) as caught:
        websocket_module.websocket_response(
            _request(),
            api_base="https://chatgpt.com/backend-api/codex",
            api_key="secret-token",
            headers={"User-Agent": "DSPy/3.2.0"},
            idle_timeout=30.0,
        )

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_connection_failure_does_not_retain_secret_exception(
    websocket_module: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    def fake_connect(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("connection failed with secret-token")

    monkeypatch.setattr(websocket_module, "_sync_connect", fake_connect)

    with pytest.raises(websocket_module.CodexWebSocketError) as caught:
        websocket_module.websocket_response(
            _request(),
            api_base="https://chatgpt.com/backend-api/codex",
            api_key="secret-token",
            headers={"User-Agent": "DSPy/3.2.0"},
        )

    assert str(caught.value) == "Codex WebSocket connection failed"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_async_server_error_uses_same_typed_error(
    websocket_module: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    connection = FakeAsyncConnection(
        [
            json.dumps(
                {
                    "type": "error",
                    "status": 429,
                    "error": {
                        "message": "rate limited",
                        "type": "rate_limit_error",
                        "param": None,
                        "code": "rate_limit_exceeded",
                    },
                }
            )
        ]
    )
    monkeypatch.setattr(
        websocket_module, "_async_connect", lambda *_a, **_k: connection
    )

    with pytest.raises(websocket_module.CodexWebSocketResponseError) as caught:
        asyncio.run(
            websocket_module.awebsocket_response(
                _request(),
                api_base="https://chatgpt.com/backend-api/codex",
                api_key="secret-token",
                headers={"User-Agent": "DSPy/3.2.0"},
            )
        )

    assert caught.value.status_code == 429
    assert caught.value.code == "rate_limit_exceeded"


@pytest.mark.parametrize("frame", ["not json", b"binary"])
def test_async_invalid_frames_raise_protocol_error(
    websocket_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    frame: str | bytes,
):
    connection = FakeAsyncConnection([frame])
    monkeypatch.setattr(
        websocket_module, "_async_connect", lambda *_a, **_k: connection
    )

    with pytest.raises(websocket_module.CodexWebSocketProtocolError):
        asyncio.run(
            websocket_module.awebsocket_response(
                _request(),
                api_base="https://chatgpt.com/backend-api/codex",
                api_key="secret-token",
                headers={"User-Agent": "DSPy/3.2.0"},
            )
        )


def test_async_close_before_completed_raises_protocol_error(
    websocket_module: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    connection = FakeAsyncConnection([])
    monkeypatch.setattr(
        websocket_module, "_async_connect", lambda *_a, **_k: connection
    )
    monkeypatch.setattr(websocket_module, "ConnectionClosed", FakeConnectionClosed)

    with pytest.raises(
        websocket_module.CodexWebSocketProtocolError,
        match="before response.completed",
    ):
        asyncio.run(
            websocket_module.awebsocket_response(
                _request(),
                api_base="https://chatgpt.com/backend-api/codex",
                api_key="secret-token",
                headers={"User-Agent": "DSPy/3.2.0"},
            )
        )


def test_async_idle_timeout_raises_typed_timeout(
    websocket_module: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    class SlowAsyncConnection(FakeAsyncConnection):
        async def recv(self) -> str | bytes:
            await asyncio.sleep(0.05)
            return json.dumps(_completed_event())

    connection = SlowAsyncConnection([])
    monkeypatch.setattr(
        websocket_module, "_async_connect", lambda *_a, **_k: connection
    )

    with pytest.raises(websocket_module.CodexWebSocketTimeoutError) as caught:
        asyncio.run(
            websocket_module.awebsocket_response(
                _request(),
                api_base="https://chatgpt.com/backend-api/codex",
                api_key="secret-token",
                headers={"User-Agent": "DSPy/3.2.0"},
                idle_timeout=0.001,
            )
        )

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_non_openai_provider_prefix_is_preserved(
    websocket_module: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    connection = FakeSyncConnection([json.dumps(_completed_event())])
    monkeypatch.setattr(websocket_module, "_sync_connect", lambda *_a, **_k: connection)
    request = _request()
    request["model"] = "other/openai/gpt-test"

    websocket_module.websocket_response(
        request,
        api_base="https://chatgpt.com/backend-api/codex",
        api_key="secret-token",
        headers={"User-Agent": "DSPy/3.2.0"},
    )

    assert json.loads(connection.sent[0])["model"] == "other/openai/gpt-test"


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_timeouts_must_be_positive_and_finite(
    websocket_module: ModuleType, timeout: float
):
    with pytest.raises(ValueError, match="positive finite"):
        websocket_module.websocket_response(
            _request(),
            api_base="https://chatgpt.com/backend-api/codex",
            api_key="secret-token",
            headers={"User-Agent": "DSPy/3.2.0"},
            connect_timeout=timeout,
        )
