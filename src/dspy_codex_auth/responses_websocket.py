"""Codex Responses WebSocket protocol client."""

from __future__ import annotations

import asyncio
import json
import math
import ssl
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests
from websockets.asyncio.client import connect as _async_connect
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect as _sync_connect

DEFAULT_CODEX_WEBSOCKET_BETA = "responses_websockets=2026-02-06"
DEFAULT_CODEX_WEBSOCKET_ORIGINATOR = "codex_cli_rs"
DEFAULT_CODEX_WEBSOCKET_CONNECT_TIMEOUT = 10.0
DEFAULT_CODEX_WEBSOCKET_IDLE_TIMEOUT = 300.0

_CLOSE_TIMEOUT = 5.0
_MAX_MESSAGE_SIZE = 16 * 1024 * 1024
_PING_INTERVAL = 20.0
_PING_TIMEOUT = 20.0
_PROTECTED_HEADER_NAMES = {
    "authorization",
    "openai-beta",
    "originator",
    "session-id",
    "thread-id",
    "x-client-request-id",
}


class CodexWebSocketError(RuntimeError):
    """Base error for the Codex Responses WebSocket transport."""


class CodexWebSocketProtocolError(CodexWebSocketError):
    """The peer sent a frame that violates the expected protocol."""


class CodexWebSocketTimeoutError(TimeoutError, CodexWebSocketError):
    """The WebSocket connection or event stream exceeded its timeout."""


class CodexWebSocketResponseError(CodexWebSocketError):
    """The Codex backend returned a structured WebSocket error event."""

    def __init__(
        self,
        *,
        status_code: int | None,
        message: str,
        error_type: str | None,
        param: str | None,
        code: Any,
    ) -> None:
        self.status_code = status_code
        self.error_type = error_type
        self.param = param
        self.code = code
        details = [
            f"status={status_code}" if status_code is not None else None,
            f"type={error_type}" if error_type else None,
            f"param={param}" if param else None,
            f"code={code}" if code is not None else None,
        ]
        rendered_details = ", ".join(detail for detail in details if detail)
        prefix = "Codex WebSocket response error"
        if rendered_details:
            prefix = f"{prefix} ({rendered_details})"
        super().__init__(f"{prefix}: {message}")


@dataclass(frozen=True, slots=True)
class WebSocketResult:
    """Responses events and the terminal response from one WebSocket call."""

    events: list[dict[str, Any]]
    response: dict[str, Any]


def codex_websocket_url(api_base: str) -> str:
    """Convert a Codex HTTP API base URL to its Responses WebSocket URL."""
    parsed = urlsplit(api_base)
    scheme = {"http": "ws", "https": "wss"}.get(parsed.scheme.lower())
    if scheme is None:
        raise ValueError("Codex api_base must use HTTP or HTTPS")
    if not parsed.netloc:
        raise ValueError("Codex api_base must include a host")
    if parsed.fragment:
        raise ValueError("Codex api_base must not include a URL fragment")

    path = f"{parsed.path.rstrip('/')}/responses"
    return urlunsplit((scheme, parsed.netloc, path, parsed.query, ""))


def _validate_timeout(name: str, value: float) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return timeout


def _handshake_headers(
    headers: Mapping[str, Any] | None,
    *,
    api_key: str,
) -> dict[str, str]:
    request_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    handshake_headers = {
        str(key): str(value)
        for key, value in (headers or {}).items()
        if str(key).lower() not in _PROTECTED_HEADER_NAMES
    }
    handshake_headers.update(
        {
            "Authorization": f"Bearer {api_key}",
            "OpenAI-Beta": DEFAULT_CODEX_WEBSOCKET_BETA,
            "originator": DEFAULT_CODEX_WEBSOCKET_ORIGINATOR,
            "x-client-request-id": request_id,
            "session-id": session_id,
            "thread-id": request_id,
        }
    )
    return handshake_headers


def _serialize_request(request: dict[str, Any]) -> str:
    if "api_key" in request:
        raise ValueError("api_key must not appear in a WebSocket request frame")

    payload = dict(request)
    model = payload.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("Codex WebSocket request model must be a non-empty string")
    if model.startswith("openai/"):
        payload["model"] = model.removeprefix("openai/")
    payload["type"] = "response.create"
    return json.dumps(payload, separators=(",", ":"))


def _decode_event(raw_message: str | bytes) -> dict[str, Any]:
    if not isinstance(raw_message, str):
        raise CodexWebSocketProtocolError(
            "Codex WebSocket returned an unexpected binary frame"
        )
    try:
        event = json.loads(raw_message)
    except json.JSONDecodeError as exc:
        raise CodexWebSocketProtocolError(
            "Codex WebSocket frame must contain valid JSON"
        ) from exc
    if not isinstance(event, dict):
        raise CodexWebSocketProtocolError("Codex WebSocket event must be a JSON object")
    if not isinstance(event.get("type"), str):
        raise CodexWebSocketProtocolError(
            "Codex WebSocket event must include a string type"
        )
    return event


def _redact(value: Any, secret: str) -> str:
    rendered = str(value)
    if secret:
        rendered = rendered.replace(secret, "<redacted>")
    return rendered


def _status_code(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _raise_response_error(event: dict[str, Any], *, api_key: str) -> None:
    raw_error = event.get("error")
    error = raw_error if isinstance(raw_error, dict) else {}
    raw_message = error.get("message", "unknown backend error")
    error_type = error.get("type")
    param = error.get("param")
    code = error.get("code")
    raise CodexWebSocketResponseError(
        status_code=_status_code(event.get("status", event.get("status_code"))),
        message=_redact(raw_message, api_key),
        error_type=_redact(error_type, api_key)
        if isinstance(error_type, str)
        else None,
        param=_redact(param, api_key) if isinstance(param, str) else None,
        code=_redact(code, api_key) if isinstance(code, str) else code,
    )


def _terminal_response(
    event: dict[str, Any],
    *,
    api_key: str,
) -> dict[str, Any] | None:
    event_type = event["type"]
    if event_type == "error":
        _raise_response_error(event, api_key=api_key)
    if event_type != "response.completed":
        return None

    response = event.get("response")
    if not isinstance(response, dict):
        raise CodexWebSocketProtocolError(
            "response.completed must include a response object"
        )
    return response


def _connection_options(
    websocket_url: str,
    *,
    headers: dict[str, str],
    connect_timeout: float,
) -> dict[str, Any]:
    ssl_context = None
    if urlsplit(websocket_url).scheme == "wss":
        ssl_context = ssl.create_default_context(cafile=requests.certs.where())
    return {
        "ssl": ssl_context,
        "additional_headers": headers,
        "user_agent_header": None,
        "open_timeout": connect_timeout,
        "close_timeout": _CLOSE_TIMEOUT,
        "ping_interval": _PING_INTERVAL,
        "ping_timeout": _PING_TIMEOUT,
        "max_size": _MAX_MESSAGE_SIZE,
    }


def websocket_response(
    request: dict[str, Any],
    *,
    api_base: str,
    api_key: str,
    headers: Mapping[str, Any] | None,
    connect_timeout: float = DEFAULT_CODEX_WEBSOCKET_CONNECT_TIMEOUT,
    idle_timeout: float = DEFAULT_CODEX_WEBSOCKET_IDLE_TIMEOUT,
) -> WebSocketResult:
    """Send one synchronous Codex Responses request over WebSocket."""
    connect_timeout = _validate_timeout("connect_timeout", connect_timeout)
    idle_timeout = _validate_timeout("idle_timeout", idle_timeout)
    websocket_url = codex_websocket_url(api_base)
    frame = _serialize_request(request)
    handshake_headers = _handshake_headers(headers, api_key=api_key)
    options = _connection_options(
        websocket_url,
        headers=handshake_headers,
        connect_timeout=connect_timeout,
    )

    events: list[dict[str, Any]] = []
    try:
        with _sync_connect(websocket_url, **options) as connection:
            connection.send(frame)
            while True:
                try:
                    raw_message = connection.recv(timeout=idle_timeout)
                except TimeoutError as exc:
                    raise CodexWebSocketTimeoutError(
                        f"Codex WebSocket received no event for {idle_timeout} seconds"
                    ) from exc
                except ConnectionClosed as exc:
                    raise CodexWebSocketProtocolError(
                        "Codex WebSocket closed before response.completed"
                    ) from exc

                event = _decode_event(raw_message)
                events.append(event)
                response = _terminal_response(event, api_key=api_key)
                if response is not None:
                    return WebSocketResult(events=events, response=response)
    except CodexWebSocketError:
        raise
    except TimeoutError as exc:
        raise CodexWebSocketTimeoutError(
            f"Codex WebSocket connection exceeded {connect_timeout} seconds"
        ) from exc
    except Exception as exc:
        raise CodexWebSocketError("Codex WebSocket connection failed") from exc


async def awebsocket_response(
    request: dict[str, Any],
    *,
    api_base: str,
    api_key: str,
    headers: Mapping[str, Any] | None,
    connect_timeout: float = DEFAULT_CODEX_WEBSOCKET_CONNECT_TIMEOUT,
    idle_timeout: float = DEFAULT_CODEX_WEBSOCKET_IDLE_TIMEOUT,
) -> WebSocketResult:
    """Send one asynchronous Codex Responses request over WebSocket."""
    connect_timeout = _validate_timeout("connect_timeout", connect_timeout)
    idle_timeout = _validate_timeout("idle_timeout", idle_timeout)
    websocket_url = codex_websocket_url(api_base)
    frame = _serialize_request(request)
    handshake_headers = _handshake_headers(headers, api_key=api_key)
    options = _connection_options(
        websocket_url,
        headers=handshake_headers,
        connect_timeout=connect_timeout,
    )

    events: list[dict[str, Any]] = []
    try:
        async with _async_connect(websocket_url, **options) as connection:
            await connection.send(frame)
            while True:
                try:
                    async with asyncio.timeout(idle_timeout):
                        raw_message = await connection.recv()
                except TimeoutError as exc:
                    raise CodexWebSocketTimeoutError(
                        f"Codex WebSocket received no event for {idle_timeout} seconds"
                    ) from exc
                except ConnectionClosed as exc:
                    raise CodexWebSocketProtocolError(
                        "Codex WebSocket closed before response.completed"
                    ) from exc

                event = _decode_event(raw_message)
                events.append(event)
                response = _terminal_response(event, api_key=api_key)
                if response is not None:
                    return WebSocketResult(events=events, response=response)
    except CodexWebSocketError:
        raise
    except TimeoutError as exc:
        raise CodexWebSocketTimeoutError(
            f"Codex WebSocket connection exceeded {connect_timeout} seconds"
        ) from exc
    except Exception as exc:
        raise CodexWebSocketError("Codex WebSocket connection failed") from exc


__all__ = [
    "DEFAULT_CODEX_WEBSOCKET_BETA",
    "DEFAULT_CODEX_WEBSOCKET_CONNECT_TIMEOUT",
    "DEFAULT_CODEX_WEBSOCKET_IDLE_TIMEOUT",
    "DEFAULT_CODEX_WEBSOCKET_ORIGINATOR",
    "CodexWebSocketError",
    "CodexWebSocketProtocolError",
    "CodexWebSocketResponseError",
    "CodexWebSocketTimeoutError",
    "WebSocketResult",
    "awebsocket_response",
    "codex_websocket_url",
    "websocket_response",
]
