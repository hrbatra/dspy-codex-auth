"""Immutable Codex connection settings and response-cache identity."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from hashlib import sha256
from typing import Any

import dspy
import httpx
from openai_codex_auth import (
    DEFAULT_CODEX_API_BASE,
    CodexAuth,
    CodexClient,
    CodexTransport,
)
from openai_codex_auth.responses_websocket import (
    DEFAULT_CODEX_WEBSOCKET_CONNECT_TIMEOUT,
    DEFAULT_CODEX_WEBSOCKET_IDLE_TIMEOUT,
)

DEFAULT_CODEX_ORIGINATOR = "dspy_codex_auth"
CACHE_IDENTITY_KEY = "_dspy_codex_cache_identity"
CONNECTION_KEYS = frozenset(
    {
        "api_base",
        "api_key",
        "chatgpt_account_id",
        "headers",
        "originator",
        "timeout",
        "codex_transport",
        "codex_websocket_connect_timeout",
        "codex_websocket_idle_timeout",
    }
)
TRANSPORT_KEYS = frozenset(key for key in CONNECTION_KEYS if key.startswith("codex_"))
_FORBIDDEN_GENERATION_OPTIONS = frozenset(
    {
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
        "engine",
        "async_engine",
    }
)


def reject_generation_options(options: dict[str, Any]) -> None:
    unsupported = _FORBIDDEN_GENERATION_OPTIONS.intersection(options)
    if unsupported:
        raise ValueError(
            f"Unsupported Codex LM option(s): {', '.join(sorted(unsupported))}. "
            "Use num_retries on LM, timeout, and codex_websocket_* timeout options."
        )
    for name, expected in (("text", dict), ("instructions", str)):
        value = options.get(name)
        if value is not None and not isinstance(value, expected):
            raise TypeError(f"Codex {name} must be a {expected.__name__}")


def positive_timeout(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive finite number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


@dataclass(frozen=True)
class CodexConnection:
    auth_storage: CodexAuth
    api_base: str = DEFAULT_CODEX_API_BASE
    api_key: str | None = field(default=None, repr=False)
    chatgpt_account_id: str | None = None
    headers: dict[str, str] | None = field(default=None, repr=False)
    originator: str = DEFAULT_CODEX_ORIGINATOR
    timeout: float | httpx.Timeout | None = None
    codex_transport: CodexTransport = "auto"
    codex_websocket_connect_timeout: float = DEFAULT_CODEX_WEBSOCKET_CONNECT_TIMEOUT
    codex_websocket_idle_timeout: float = DEFAULT_CODEX_WEBSOCKET_IDLE_TIMEOUT

    def __post_init__(self) -> None:
        if not isinstance(self.codex_transport, str) or self.codex_transport not in {
            "auto",
            "http",
            "websocket",
        }:
            raise ValueError(
                "codex_transport must be one of 'auto', 'http', or 'websocket'"
            )
        for name in ("codex_websocket_connect_timeout", "codex_websocket_idle_timeout"):
            object.__setattr__(self, name, positive_timeout(name, getattr(self, name)))
        if isinstance(self.timeout, httpx.Timeout):
            for phase, value in self.timeout.as_dict().items():
                if value is not None:
                    positive_timeout(f"timeout.{phase}", value)
        elif self.timeout is not None:
            object.__setattr__(
                self, "timeout", positive_timeout("timeout", self.timeout)
            )
        # The shared client validates connection fields without reading auth or
        # opening a network connection. Keep that contract in one place.
        client = self.create_client()
        object.__setattr__(self, "headers", dict(client.headers))

    def create_client(self) -> CodexClient:
        return CodexClient(
            auth=self.auth_storage,
            api_base=self.api_base,
            api_key=self.api_key,
            account_id=self.chatgpt_account_id,
            headers=self.headers,
            originator=self.originator,
            user_agent=f"DSPy/{dspy.__version__}",
        )

    def for_call(self, overrides: dict[str, Any]) -> CodexConnection:
        selected = replace(self, **overrides)
        # An explicit call timeout wins; otherwise an explicit call idle
        # deadline wins over the constructor's request timeout.
        if isinstance(selected.timeout, (int, float)) and (
            "timeout" in overrides or "codex_websocket_idle_timeout" not in overrides
        ):
            selected = replace(selected, codex_websocket_idle_timeout=selected.timeout)
        return selected

    def request_controls(self) -> dict[str, Any]:
        controls: dict[str, Any] = {
            "transport": self.codex_transport,
            "connect_timeout": self.codex_websocket_connect_timeout,
            "idle_timeout": self.codex_websocket_idle_timeout,
            "max_retries": 0,
        }
        if self.timeout is not None:
            controls["timeout"] = self.timeout
        return controls

    def options(self, *, secrets: bool) -> dict[str, Any]:
        values = {name: getattr(self, name) for name in CONNECTION_KEYS}
        if not secrets:
            values.pop("api_key")
            values.pop("headers")
        return values

    def cache_identity(self) -> str:
        identity = self.options(secrets=True)
        identity["auth_path"] = str(self.auth_storage.path)
        if self.api_key is None:
            identity["file_account_id"] = self.auth_storage.account_id()
        if isinstance(self.timeout, httpx.Timeout):
            identity["timeout"] = self.timeout.as_dict()
        # Only this digest reaches DSPy's request/cache/history. In particular,
        # neither a bearer token nor a user-supplied header becomes request data.
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return sha256(("codex-lm15-v1:" + encoded).encode()).hexdigest()
