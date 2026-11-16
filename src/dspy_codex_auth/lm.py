"""DSPy 3.5 engines backed by the shared Codex subscription client.

Portions are adapted from dspy-lm-auth under the MIT License. See
THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace
from typing import Any

import dspy
import httpx
from dspy.lm15 import Request, Response
from openai_codex_auth import (
    DEFAULT_CODEX_API_BASE,
    DEFAULT_CODEX_INSTRUCTIONS,
    CodexAuth,
    CodexAuthError,
    CodexTransport,
    getauthtoken,
)

from .connection import (
    CACHE_IDENTITY_KEY,
    CONNECTION_KEYS,
    DEFAULT_CODEX_ORIGINATOR,
    DEFAULT_CODEX_WEBSOCKET_CONNECT_TIMEOUT,
    DEFAULT_CODEX_WEBSOCKET_IDLE_TIMEOUT,
    TRANSPORT_KEYS,
    CodexConnection,
    reject_generation_options,
)
from .engines import AsyncCodexEngine, CodexEngine

DEFAULT_CODEX_MODEL = "gpt-5.4"
OPENAI_CODEX_PROVIDER = "openai-codex"
_CODEX_ROUTE_ALIASES = ("codex", "chatgpt", OPENAI_CODEX_PROVIDER)
_DEFAULT_AUTH: CodexAuth | None = None
_DSPY_LM = dspy.LM
_ORIGINAL_DSPY_LM = dspy.LM
# Dynamic call state stays outside LM instances so DSPy can deepcopy/pickle
# programs. The owner check also isolates directly borrowed engine calls.
_CALL_CONNECTION: ContextVar[tuple[object, CodexConnection] | None] = ContextVar(
    "codex_connection", default=None
)

RouteResolver = Callable[[str, dict[str, Any], CodexAuth], tuple[str, dict[str, Any]]]
_ROUTE_RESOLVERS: dict[str, RouteResolver] = {}


@dataclass(frozen=True, slots=True)
class RouteRegistration:
    aliases: tuple[str, ...]
    resolver: RouteResolver


def _coerce_auth_storage(
    auth_storage: CodexAuth | str | os.PathLike[str] | None,
) -> CodexAuth:
    if auth_storage is None:
        return _DEFAULT_AUTH or CodexAuth()
    return (
        auth_storage if isinstance(auth_storage, CodexAuth) else CodexAuth(auth_storage)
    )


def _normalize_route(name: str) -> str:
    return OPENAI_CODEX_PROVIDER if name in _CODEX_ROUTE_ALIASES else name


def register_model_alias(
    aliases: str | tuple[str, ...] | list[str],
    resolver: RouteResolver,
) -> None:
    for alias in (aliases,) if isinstance(aliases, str) else aliases:
        _ROUTE_RESOLVERS[alias] = resolver


def unregister_model_alias(alias: str) -> None:
    _ROUTE_RESOLVERS.pop(alias, None)


def _resolve_codex_route(
    model: str,
    kwargs: dict[str, Any],
    auth_storage: CodexAuth,
) -> tuple[str, dict[str, Any]]:
    model_id = (
        model.split("/", 1)[1]
        if "/" in model
        else (DEFAULT_CODEX_MODEL if model in _CODEX_ROUTE_ALIASES else model)
    )
    return f"openai/{model_id}", {
        "originator": DEFAULT_CODEX_ORIGINATOR,
        "api_base": DEFAULT_CODEX_API_BASE,
        "model_type": "responses",
        **kwargs,
    }


def resolve_lm_route(
    model: str,
    *,
    auth_storage: CodexAuth,
    auth_provider: str | None = None,
    kwargs: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    options = dict(kwargs or {})
    alias = _normalize_route(auth_provider) if auth_provider else model.split("/", 1)[0]
    resolver = _ROUTE_RESOLVERS.get(alias)
    if auth_provider and resolver is None:
        raise ValueError(
            f"No DSPy LM auth route registered for auth_provider={auth_provider!r}"
        )
    return resolver(model, options, auth_storage) if resolver else (model, options)


def _reasoning_options(options: dict[str, Any]) -> dict[str, Any]:
    options = dict(options)
    for name in ("reasoning_effort", "reasoning_summary"):
        alias = f"model_{name}"
        if alias in options:
            options.setdefault(name, options.pop(alias))
    return options


class LM(_DSPY_LM):
    """Codex routing and per-call connections around DSPy's canonical engines."""

    def __init__(
        self,
        model: str,
        *,
        auth_storage: CodexAuth | str | os.PathLike[str] | None = None,
        auth_provider: str | None = None,
        codex_transport: CodexTransport = "auto",
        codex_websocket_connect_timeout: float = DEFAULT_CODEX_WEBSOCKET_CONNECT_TIMEOUT,
        codex_websocket_idle_timeout: float = DEFAULT_CODEX_WEBSOCKET_IDLE_TIMEOUT,
        **kwargs: Any,
    ) -> None:
        storage = _coerce_auth_storage(auth_storage)
        resolved_model, options = resolve_lm_route(
            model,
            auth_storage=storage,
            auth_provider=auth_provider,
            kwargs=kwargs,
        )
        route = _normalize_route(auth_provider or model.split("/", 1)[0])
        self._uses_codex_route = (
            route == OPENAI_CODEX_PROVIDER
            or options.get("api_base") == DEFAULT_CODEX_API_BASE
        )
        self.original_model_string = model
        self.resolved_model_string = resolved_model
        self.auth_provider = auth_provider
        self.auth_storage = storage
        self.codex_transport = codex_transport
        self.codex_websocket_connect_timeout = codex_websocket_connect_timeout
        self.codex_websocket_idle_timeout = codex_websocket_idle_timeout
        if not self._uses_codex_route:
            if (
                codex_transport != "auto"
                or codex_websocket_connect_timeout
                != DEFAULT_CODEX_WEBSOCKET_CONNECT_TIMEOUT
                or codex_websocket_idle_timeout != DEFAULT_CODEX_WEBSOCKET_IDLE_TIMEOUT
            ):
                raise ValueError("Codex transport settings require a Codex LM route")
            super().__init__(resolved_model, **options)
            return

        if options.pop("model_type", "responses") != "responses":
            raise ValueError("Codex routes require model_type='responses'")
        # num_retries is a constructor execution setting, never a generation
        # option sent to Codex. DSPy owns all attempts.
        retries = options.pop("num_retries", 3)
        reject_generation_options(options)
        connection_options = {
            key: options.pop(key) for key in CONNECTION_KEYS if key in options
        }
        self._connection = CodexConnection(
            auth_storage=storage,
            codex_transport=codex_transport,
            codex_websocket_connect_timeout=codex_websocket_connect_timeout,
            codex_websocket_idle_timeout=codex_websocket_idle_timeout,
            **connection_options,
        )
        super().__init__(
            resolved_model,
            model_type="responses",
            num_retries=retries,
            engine=CodexEngine(self._connection_for_engine),
            async_engine=AsyncCodexEngine(self._connection_for_engine),
            **_reasoning_options(options),
        )

    def _connection_for_engine(self) -> CodexConnection:
        current = _CALL_CONNECTION.get()
        if current is not None and current[0] is self:
            return current[1]
        return self._connection.for_call({})

    def _prepare_call(
        self,
        prompt: Request | str | None,
        options: dict[str, Any],
    ) -> tuple[Request | str | None, dict[str, Any], CodexConnection]:
        if "messages" in options:
            raise TypeError(
                "messages= was removed in DSPy 3.5. Use lm(Request(...)) with "
                "dspy.lm15.Message objects and Request.system for instructions."
            )
        if "request" in options:
            if prompt is not None:
                raise TypeError("Pass a Request once, positionally or by keyword")
            prompt = options.pop("request")
        if not isinstance(prompt, (str, Request)):
            raise TypeError("An LM call takes a dspy.lm15.Request or a prompt string")
        reject_generation_options(options)
        overrides = {key: options.pop(key) for key in CONNECTION_KEYS if key in options}
        if isinstance(prompt, Request):
            reject_generation_options(prompt.config.extensions or {})
        connection = self._connection.for_call(overrides)
        try:
            identity = connection.cache_identity()
        except CodexAuthError as error:
            raise dspy.LMAuthError(
                str(error), model=self.model, provider=OPENAI_CODEX_PROVIDER
            ) from error
        if isinstance(prompt, Request):
            config = replace(
                prompt.config,
                extensions={
                    **(prompt.config.extensions or {}),
                    CACHE_IDENTITY_KEY: identity,
                },
            )
            prompt = replace(prompt, config=config)
        else:
            options = {**_reasoning_options(options), CACHE_IDENTITY_KEY: identity}
        return prompt, options, connection

    def __call__(
        self,
        prompt: Request | str | None = None,
        *,
        cache: bool | None = None,
        rollout_id: Any = None,
        **options: Any,
    ) -> Response | list[str | dict[str, Any] | None]:
        if not self._uses_codex_route:
            self._reject_transport_overrides(options)
            return super().__call__(
                prompt, cache=cache, rollout_id=rollout_id, **options
            )
        prompt, options, connection = self._prepare_call(prompt, options)
        token = _CALL_CONNECTION.set((self, connection))
        try:
            return super().__call__(
                prompt, cache=cache, rollout_id=rollout_id, **options
            )
        finally:
            _CALL_CONNECTION.reset(token)

    async def acall(
        self,
        prompt: Request | str | None = None,
        *,
        cache: bool | None = None,
        rollout_id: Any = None,
        **options: Any,
    ) -> Response | list[str | dict[str, Any] | None]:
        if not self._uses_codex_route:
            self._reject_transport_overrides(options)
            return await super().acall(
                prompt, cache=cache, rollout_id=rollout_id, **options
            )
        prompt, options, connection = self._prepare_call(prompt, options)
        token = _CALL_CONNECTION.set((self, connection))
        try:
            return await super().acall(
                prompt, cache=cache, rollout_id=rollout_id, **options
            )
        finally:
            _CALL_CONNECTION.reset(token)

    @staticmethod
    def _reject_transport_overrides(options: dict[str, Any]) -> None:
        if TRANSPORT_KEYS.intersection(options):
            raise ValueError("Codex transport overrides require a Codex LM route")

    def _constructor_options(self, *, secrets: bool) -> dict[str, Any]:
        generation = dict(self.kwargs)
        if "max_completion_tokens" in generation:
            generation["max_tokens"] = generation.pop("max_completion_tokens")
        return {
            **generation,
            **self._connection.options(secrets=secrets),
            "model": self.model,
            "model_type": self.model_type,
            "auth_storage": str(self.auth_storage.path),
            "auth_provider": OPENAI_CODEX_PROVIDER,
            "cache": self.cache,
            "num_retries": self.num_retries,
        }

    def copy(self, **kwargs: Any) -> LM:
        if not self._uses_codex_route:
            self._reject_transport_overrides(kwargs)
            return super().copy(**kwargs)
        return type(self)(
            **{
                **self._constructor_options(secrets=True),
                "auth_storage": self.auth_storage,
                "callbacks": list(self.callbacks),
                **kwargs,
            }
        )

    def dump_state(self) -> dict[str, Any]:
        if not self._uses_codex_route:
            return super().dump_state()
        state = self._constructor_options(secrets=False)
        state["_dspy_lm_class"] = f"{type(self).__module__}.{type(self).__qualname__}"
        if isinstance(state["timeout"], httpx.Timeout):
            state["timeout"] = state["timeout"].as_dict()
        if state.get("prompt_cache") is not None:
            state["prompt_cache"] = {
                key: value
                for key, value in asdict(state["prompt_cache"]).items()
                if value is not None
            }
        return state

    @classmethod
    def load_state(
        cls, state: dict[str, Any], *, allow_custom_lm_class: bool = False
    ) -> LM:
        state = dict(state)
        if isinstance(state.get("timeout"), dict):
            state["timeout"] = httpx.Timeout(**state["timeout"])
        return super().load_state(state, allow_custom_lm_class=allow_custom_lm_class)


def install(
    *,
    auth_storage: CodexAuth | str | os.PathLike[str] | None = None,
    attach_helpers: bool = True,
) -> type[LM]:
    global _DEFAULT_AUTH
    _DEFAULT_AUTH = _coerce_auth_storage(auth_storage)
    dspy.LM = LM
    dspy.clients.LM = LM
    if attach_helpers:
        dspy.getauthtoken = getauthtoken
    return LM


def uninstall() -> None:
    dspy.LM = _ORIGINAL_DSPY_LM
    dspy.clients.LM = _ORIGINAL_DSPY_LM
    if hasattr(dspy, "getauthtoken"):
        delattr(dspy, "getauthtoken")


register_model_alias(_CODEX_ROUTE_ALIASES, _resolve_codex_route)

__all__ = [
    "DEFAULT_CODEX_API_BASE",
    "DEFAULT_CODEX_INSTRUCTIONS",
    "DEFAULT_CODEX_MODEL",
    "DEFAULT_CODEX_ORIGINATOR",
    "DEFAULT_CODEX_WEBSOCKET_CONNECT_TIMEOUT",
    "DEFAULT_CODEX_WEBSOCKET_IDLE_TIMEOUT",
    "LM",
    "CodexTransport",
    "RouteRegistration",
    "install",
    "register_model_alias",
    "resolve_lm_route",
    "uninstall",
    "unregister_model_alias",
]
