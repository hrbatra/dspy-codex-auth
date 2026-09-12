"""DSPy LM integration for ChatGPT/Codex subscription credentials.

Portions are adapted from dspy-lm-auth under the MIT License. See
THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from types import SimpleNamespace
from typing import Any, Literal, cast

import dspy
from litellm.types.responses.main import OutputFunctionToolCall
from openai_codex_auth import (
    DEFAULT_CODEX_API_BASE,
    DEFAULT_CODEX_INSTRUCTIONS,
    CodexAuth,
    CodexClient,
    CodexError,
    CodexResponse,
    CodexTransport,
    getauthtoken,
)

from openai_codex_auth.responses_websocket import (
    DEFAULT_CODEX_WEBSOCKET_CONNECT_TIMEOUT,
    DEFAULT_CODEX_WEBSOCKET_IDLE_TIMEOUT,
)

DEFAULT_CODEX_MODEL = "gpt-5.4"
DEFAULT_CODEX_ORIGINATOR = "dspy_codex_auth"

OPENAI_CODEX_PROVIDER = "openai-codex"
_CODEX_ROUTE_ALIASES = ("codex", "chatgpt", OPENAI_CODEX_PROVIDER)
_DEFAULT_AUTH: CodexAuth | None = None

_CODEX_TRANSPORTS = {"auto", "http", "websocket"}
_CODEX_CACHE_CONTROL_KEY = "_dspy_codex_transport_controls"
_UNSUPPORTED_CODEX_OPTIONS = frozenset(
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
    }
)
_DSPY_LM = dspy.LM
_ORIGINAL_DSPY_LM = dspy.LM

RouteResolver = Callable[[str, dict[str, Any], CodexAuth], tuple[str, dict[str, Any]]]
_ROUTE_RESOLVERS: dict[str, RouteResolver] = {}


def _validate_codex_transport(value: Any) -> CodexTransport:
    if not isinstance(value, str) or value not in _CODEX_TRANSPORTS:
        raise ValueError(
            "codex_transport must be one of 'auto', 'http', or 'websocket'"
        )
    return cast(CodexTransport, value)


def _validate_codex_websocket_timeout(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite number")
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return timeout


def _request_timeout_overrides_codex_websocket_idle_timeout(
    *,
    per_call_idle_timeout: float | None,
    constructor_request_timeout: Any,
    per_call_kwargs: dict[str, Any],
) -> bool:
    if per_call_kwargs.get("timeout") is not None:
        return True
    if per_call_idle_timeout is not None:
        return False
    return "timeout" not in per_call_kwargs and constructor_request_timeout is not None


def _codex_cache_request(
    request: dict[str, Any],
    *,
    codex_transport: CodexTransport,
    codex_websocket_connect_timeout: float,
    codex_websocket_idle_timeout: float,
    request_timeout_overrides_websocket_idle_timeout: bool,
) -> dict[str, Any]:
    return {
        **request,
        _CODEX_CACHE_CONTROL_KEY: {
            "transport": codex_transport,
            "connect_timeout": codex_websocket_connect_timeout,
            "idle_timeout": codex_websocket_idle_timeout,
            "request_timeout_overrides_idle_timeout": (
                request_timeout_overrides_websocket_idle_timeout
            ),
        },
    }


@dataclass(frozen=True, slots=True)
class RouteRegistration:
    aliases: tuple[str, ...]
    resolver: RouteResolver


def _coerce_auth_storage(
    auth_storage: CodexAuth | str | os.PathLike[str] | None,
) -> CodexAuth:
    if auth_storage is None:
        return _DEFAULT_AUTH or CodexAuth()
    if isinstance(auth_storage, CodexAuth):
        return auth_storage
    return CodexAuth(auth_storage)


def _normalize_route(name: str) -> str:
    return OPENAI_CODEX_PROVIDER if name in _CODEX_ROUTE_ALIASES else name


def register_model_alias(
    aliases: str | tuple[str, ...] | list[str],
    resolver: RouteResolver,
) -> None:
    if isinstance(aliases, str):
        aliases = (aliases,)
    for alias in aliases:
        _ROUTE_RESOLVERS[alias] = resolver


def unregister_model_alias(alias: str) -> None:
    _ROUTE_RESOLVERS.pop(alias, None)


def _resolve_codex_route(
    model: str,
    kwargs: dict[str, Any],
    auth_storage: CodexAuth,
) -> tuple[str, dict[str, Any]]:
    if "/" in model:
        _, model_id = model.split("/", 1)
    else:
        model_id = DEFAULT_CODEX_MODEL if model in _CODEX_ROUTE_ALIASES else model

    resolved_kwargs = dict(kwargs)
    resolved_kwargs.setdefault("originator", DEFAULT_CODEX_ORIGINATOR)
    resolved_kwargs.setdefault("api_base", DEFAULT_CODEX_API_BASE)
    resolved_kwargs.setdefault("model_type", "responses")
    resolved_kwargs.setdefault("use_developer_role", True)
    return f"openai/{model_id}", resolved_kwargs


def resolve_lm_route(
    model: str,
    *,
    auth_storage: CodexAuth,
    auth_provider: str | None = None,
    kwargs: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    resolved_kwargs = dict(kwargs or {})

    if auth_provider:
        provider = _normalize_route(auth_provider)
        resolver = _ROUTE_RESOLVERS.get(provider)
        if resolver is None:
            raise ValueError(
                f"No DSPy LM auth route registered for auth_provider={auth_provider!r}"
            )
        return resolver(model, resolved_kwargs, auth_storage)

    alias = model.split("/", 1)[0]
    resolver = _ROUTE_RESOLVERS.get(alias)
    if resolver is None and model in _ROUTE_RESOLVERS:
        resolver = _ROUTE_RESOLVERS[model]

    if resolver is None:
        return model, resolved_kwargs
    return resolver(model, resolved_kwargs, auth_storage)


def _stringify_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type in {"text", "input_text", "output_text"}:
                    text = item.get("text")
                    if isinstance(text, str) and text:
                        parts.append(text)
                elif item_type == "image_url":
                    image_url = item.get("image_url")
                    if isinstance(image_url, dict):
                        url = image_url.get("url")
                        if isinstance(url, str) and url:
                            parts.append(url)
                elif item_type == "input_image":
                    image_url = item.get("image_url")
                    if isinstance(image_url, str) and image_url:
                        parts.append(image_url)
            elif item is not None:
                parts.append(str(item))
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _convert_content_item_to_responses_format(item: dict[str, Any]) -> dict[str, Any]:
    item_type = item.get("type")
    if item_type == "image_url":
        image_url = item.get("image_url", {})
        if isinstance(image_url, dict):
            image_url = image_url.get("url", "")
        return {
            "type": "input_image",
            "image_url": image_url,
        }
    if item_type in {"text", "input_text", "output_text"}:
        return {
            "type": "input_text",
            "text": item.get("text", ""),
        }
    if item_type == "file":
        file = item.get("file", {})
        return {
            "type": "input_file",
            "file_data": file.get("file_data"),
            "filename": file.get("filename"),
            "file_id": file.get("file_id"),
        }
    return item


def _convert_text_blocks_for_role(
    blocks: list[dict[str, Any]],
    *,
    role: str,
) -> list[dict[str, Any]]:
    text_type = "output_text" if role == "assistant" else "input_text"
    converted: list[dict[str, Any]] = []
    for block in blocks:
        if block.get("type") in {"input_text", "output_text"}:
            converted.append({"type": text_type, "text": block.get("text", "")})
        else:
            converted.append(block)
    return converted


def _convert_message_content_to_responses_format(
    content: Any,
    *,
    role: str = "user",
) -> list[dict[str, Any]]:
    if isinstance(content, str):
        text_type = "output_text" if role == "assistant" else "input_text"
        return [{"type": text_type, "text": content}]
    if isinstance(content, list):
        blocks: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, dict):
                blocks.append(_convert_content_item_to_responses_format(item))
            elif item is not None:
                blocks.append({"type": "input_text", "text": str(item)})
        return _convert_text_blocks_for_role(blocks, role=role)
    if content is None:
        return []
    text_type = "output_text" if role == "assistant" else "input_text"
    return [{"type": text_type, "text": str(content)}]


def _coerce_response_format(response_format: Any) -> Any:
    if hasattr(response_format, "model_json_schema") and hasattr(
        response_format, "__name__"
    ):
        return {
            "name": response_format.__name__,
            "type": "json_schema",
            "schema": response_format.model_json_schema(),
        }
    if isinstance(response_format, dict) and "json_schema" in response_format:
        return {"type": "json_schema", **response_format["json_schema"]}
    return response_format


def _merge_codex_instructions(
    explicit_instructions: Any,
    instruction_messages: list[str],
) -> str:
    parts: list[str] = []
    if explicit_instructions is not None:
        explicit_text = str(explicit_instructions).strip()
        if explicit_text:
            parts.append(explicit_text)

    for instruction in instruction_messages:
        cleaned = instruction.strip()
        if cleaned and cleaned not in parts:
            parts.append(cleaned)

    if not parts:
        parts.append(DEFAULT_CODEX_INSTRUCTIONS)
    return "\n\n".join(parts)


def _build_codex_responses_request(request: dict[str, Any]) -> dict[str, Any]:
    request = dict(request)
    raw_messages = request.pop("messages", [])
    messages = raw_messages if isinstance(raw_messages, list) else []

    instructions_from_messages: list[str] = []
    input_messages: list[dict[str, Any]] = []
    for raw_message in messages:
        if not isinstance(raw_message, dict):
            continue
        message = raw_message
        role = str(message.get("role", "user"))
        content = message.get("content")
        if role in {"system", "developer"}:
            instruction_text = _stringify_message_content(content).strip()
            if instruction_text:
                instructions_from_messages.append(instruction_text)
            continue

        if role == "tool":
            input_messages.append(
                {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": _stringify_message_content(content),
                }
            )
            continue

        tool_calls = message.get("tool_calls") or []
        if content or not tool_calls:
            input_messages.append(
                {
                    "role": role,
                    "content": _convert_message_content_to_responses_format(
                        content,
                        role=role,
                    ),
                }
            )
        for tool_call in tool_calls:
            function = tool_call["function"]
            input_messages.append(
                {
                    "type": "function_call",
                    "call_id": tool_call["id"],
                    "name": function["name"],
                    "arguments": function["arguments"],
                }
            )

    request["input"] = input_messages
    request["instructions"] = _merge_codex_instructions(
        request.pop("instructions", None),
        instructions_from_messages,
    )

    if request.get("max_output_tokens") is None:
        max_tokens = request.pop("max_tokens", None)
        if max_tokens is not None:
            request["max_output_tokens"] = max_tokens
    else:
        request.pop("max_tokens", None)

    if "reasoning_effort" in request:
        effort = request.pop("reasoning_effort")
        request["reasoning"] = {"effort": effort, "summary": "auto"}

    if "response_format" in request:
        response_format = _coerce_response_format(request.pop("response_format"))
        text = request.pop("text", {}) or {}
        request["text"] = {**text, "format": response_format}

    if "tools" in request:
        request["tools"] = [
            {"type": "function", **tool["function"]}
            if tool.get("type") == "function" and "function" in tool
            else tool
            for tool in request["tools"]
        ]
    tool_choice = request.get("tool_choice")
    if isinstance(tool_choice, dict) and "function" in tool_choice:
        request["tool_choice"] = {"type": "function", **tool_choice["function"]}

    return request


def _build_codex_request(request: dict[str, Any]) -> dict[str, Any]:
    request = dict(request)
    if "reasoning_effort" not in request and "model_reasoning_effort" in request:
        request["reasoning_effort"] = request.pop("model_reasoning_effort")
    else:
        request.pop("model_reasoning_effort", None)

    if "reasoning_summary" not in request and "model_reasoning_summary" in request:
        request["reasoning_summary"] = request.pop("model_reasoning_summary")
    else:
        request.pop("model_reasoning_summary", None)

    reasoning_summary = request.pop("reasoning_summary", None)
    request = _build_codex_responses_request(request)

    max_completion_tokens = request.pop("max_completion_tokens", None)
    if request.get("max_output_tokens") is None and max_completion_tokens is not None:
        request["max_output_tokens"] = max_completion_tokens

    if reasoning_summary is not None:
        reasoning = dict(request.pop("reasoning", {}) or {})
        reasoning["summary"] = reasoning_summary
        request["reasoning"] = reasoning
    request.pop("reasoning_summary", None)
    return request


def _response_for_dspy(result: CodexResponse) -> SimpleNamespace:
    """Adapt the shared client's final JSON into DSPy's Responses objects."""
    output = []
    for item in result.output:
        item_type = item.get("type")
        if item_type == "function_call":
            output.append(OutputFunctionToolCall(**item))
        elif item_type == "message":
            for part in item.get("content", []):
                if part.get("type") == "refusal":
                    raise CodexError(
                        f"Codex refused the request: {part.get('refusal', '')}"
                    )
            output.append(
                SimpleNamespace(
                    **{
                        **item,
                        "content": [
                            SimpleNamespace(**part)
                            for part in item.get("content", [])
                            if isinstance(part.get("text"), str)
                        ],
                    }
                )
            )
        elif item_type == "reasoning":
            output.append(
                SimpleNamespace(
                    **{
                        **item,
                        "summary": [
                            SimpleNamespace(**part)
                            for part in item.get("summary", [])
                            if isinstance(part.get("text"), str)
                        ],
                        "content": [
                            SimpleNamespace(**part)
                            for part in item.get("content", [])
                            if isinstance(part.get("text"), str)
                        ],
                    }
                )
            )
    return SimpleNamespace(
        **{
            **result.response,
            "output": output,
            "model": result.model,
            "usage": result.usage,
            "codex_transport": result.transport,
        }
    )


def _prepare_client_call(
    request: dict[str, Any],
    auth_storage: CodexAuth,
    num_retries: int,
) -> tuple[CodexClient, dict[str, Any]]:
    request = dict(request)
    controls = request.pop(_CODEX_CACHE_CONTROL_KEY)
    client = CodexClient(
        auth=auth_storage,
        api_base=request.pop("api_base", DEFAULT_CODEX_API_BASE),
        api_key=request.pop("api_key", None),
        account_id=request.pop("chatgpt_account_id", None),
        headers=request.pop("headers", None),
        originator=request.pop("originator", DEFAULT_CODEX_ORIGINATOR),
        user_agent=f"DSPy/{dspy.__version__}",
    )
    request.pop("rollout_id", None)
    timeout = request.pop("timeout", None)
    idle_timeout = controls["idle_timeout"]
    if controls["request_timeout_overrides_idle_timeout"]:
        # Structured HTTP timeouts retain their per-phase deadlines. Numeric
        # request timeouts also override the WebSocket idle timeout.
        if isinstance(timeout, (int, float)):
            idle_timeout = _validate_codex_websocket_timeout("timeout", timeout)
    request["model"] = request["model"].removeprefix("openai/")
    parameters = {
        key: value
        for key, value in _build_codex_request(request).items()
        if value is not None
    }
    parameters.update(
        transport=controls["transport"],
        connect_timeout=controls["connect_timeout"],
        idle_timeout=idle_timeout,
        max_retries=num_retries,
    )
    if timeout is not None:
        parameters["timeout"] = timeout
    return client, parameters


def _codex_completion(
    request: dict[str, Any],
    auth_storage: CodexAuth,
    num_retries: int,
    cache: dict[str, Any] | None = None,
) -> SimpleNamespace:
    client, parameters = _prepare_client_call(request, auth_storage, num_retries)
    return _response_for_dspy(client.create(**parameters))


async def _acodex_completion(
    request: dict[str, Any],
    auth_storage: CodexAuth,
    num_retries: int,
    cache: dict[str, Any] | None = None,
) -> SimpleNamespace:
    client, parameters = _prepare_client_call(request, auth_storage, num_retries)
    return _response_for_dspy(await client.acreate(**parameters))


class LM(_DSPY_LM):
    """DSPy LM backed by the shared, framework-independent Codex client."""

    def __init__(
        self,
        model: str,
        *args: Any,
        auth_storage: CodexAuth | str | os.PathLike[str] | None = None,
        auth_provider: str | None = None,
        codex_transport: Literal["auto", "http", "websocket"] = "auto",
        codex_websocket_connect_timeout: float = (
            DEFAULT_CODEX_WEBSOCKET_CONNECT_TIMEOUT
        ),
        codex_websocket_idle_timeout: float = DEFAULT_CODEX_WEBSOCKET_IDLE_TIMEOUT,
        **kwargs: Any,
    ) -> None:
        resolved_transport = _validate_codex_transport(codex_transport)
        resolved_connect_timeout = _validate_codex_websocket_timeout(
            "codex_websocket_connect_timeout",
            codex_websocket_connect_timeout,
        )
        resolved_idle_timeout = _validate_codex_websocket_timeout(
            "codex_websocket_idle_timeout",
            codex_websocket_idle_timeout,
        )
        requested_route = _normalize_route(
            auth_provider if auth_provider else model.split("/", 1)[0]
        )
        if requested_route == OPENAI_CODEX_PROVIDER:
            kwargs.setdefault("originator", DEFAULT_CODEX_ORIGINATOR)

        storage = _coerce_auth_storage(auth_storage)
        resolved_model, resolved_kwargs = resolve_lm_route(
            model,
            auth_storage=storage,
            auth_provider=auth_provider,
            kwargs=kwargs,
        )

        self.auth_storage = storage
        self.original_model_string = model
        self.auth_provider = auth_provider
        self.resolved_model_string = resolved_model
        self.codex_transport = resolved_transport
        self.codex_websocket_connect_timeout = resolved_connect_timeout
        self.codex_websocket_idle_timeout = resolved_idle_timeout
        uses_codex_route = (
            requested_route == OPENAI_CODEX_PROVIDER
            or resolved_kwargs.get("api_base") == DEFAULT_CODEX_API_BASE
        )
        if not uses_codex_route and (
            resolved_transport != "auto"
            or resolved_connect_timeout != DEFAULT_CODEX_WEBSOCKET_CONNECT_TIMEOUT
            or resolved_idle_timeout != DEFAULT_CODEX_WEBSOCKET_IDLE_TIMEOUT
        ):
            raise ValueError("Codex transport settings require a Codex LM route")
        self._uses_codex_route = uses_codex_route
        if uses_codex_route:
            resolved_kwargs.setdefault("model_type", "responses")
            if resolved_kwargs["model_type"] != "responses":
                raise ValueError("Codex routes require model_type='responses'")
        super().__init__(resolved_model, *args, **resolved_kwargs)

    def _prepare_completion(
        self,
        prompt: str | None,
        messages: list[dict[str, Any]] | None,
        codex_transport: CodexTransport | None,
        codex_websocket_connect_timeout: float | None,
        codex_websocket_idle_timeout: float | None,
        kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        selected_transport = _validate_codex_transport(
            self.codex_transport if codex_transport is None else codex_transport
        )
        selected_connect_timeout = _validate_codex_websocket_timeout(
            "codex_websocket_connect_timeout",
            self.codex_websocket_connect_timeout
            if codex_websocket_connect_timeout is None
            else codex_websocket_connect_timeout,
        )
        selected_idle_timeout = _validate_codex_websocket_timeout(
            "codex_websocket_idle_timeout",
            self.codex_websocket_idle_timeout
            if codex_websocket_idle_timeout is None
            else codex_websocket_idle_timeout,
        )
        request_timeout_overrides_idle_timeout = (
            _request_timeout_overrides_codex_websocket_idle_timeout(
                per_call_idle_timeout=codex_websocket_idle_timeout,
                constructor_request_timeout=self.kwargs.get("timeout"),
                per_call_kwargs=kwargs,
            )
        )
        kwargs = dict(kwargs)
        cache = kwargs.pop("cache", self.cache)

        messages = messages or [{"role": "user", "content": prompt}]
        kwargs = {**self.kwargs, **kwargs}
        unsupported = _UNSUPPORTED_CODEX_OPTIONS.intersection(kwargs)
        if unsupported:
            raise ValueError(
                f"Unsupported Codex LM option(s): {', '.join(sorted(unsupported))}. "
                "Use num_retries on LM, timeout, and codex_websocket_* timeout options."
            )
        self._warn_zero_temp_rollout(
            kwargs.get("temperature"), kwargs.get("rollout_id")
        )
        if kwargs.get("rollout_id") is None:
            kwargs.pop("rollout_id", None)

        request = _codex_cache_request(
            dict(model=self.model, messages=messages, **kwargs),
            codex_transport=selected_transport,
            codex_websocket_connect_timeout=selected_connect_timeout,
            codex_websocket_idle_timeout=selected_idle_timeout,
            request_timeout_overrides_websocket_idle_timeout=(
                request_timeout_overrides_idle_timeout
            ),
        )
        request[_CODEX_CACHE_CONTROL_KEY]["auth_path"] = str(self.auth_storage.path)
        controls = request[_CODEX_CACHE_CONTROL_KEY]
        controls["api_base"] = kwargs.get("api_base", DEFAULT_CODEX_API_BASE)
        api_key = kwargs.get("api_key")
        if api_key is not None:
            if not isinstance(api_key, str):
                raise ValueError("api_key must be a string")
            controls["credential_digest"] = sha256(api_key.encode()).hexdigest()
        else:
            # Read identity at cache lookup so signing into another account in
            # the same file cannot reuse its predecessor's cached responses.
            # The shared client reads and refreshes tokens only on cache misses.
            controls["account_id"] = self.auth_storage.account_id()
        return request, cache

    def _record_usage(self, results: SimpleNamespace) -> SimpleNamespace:
        self._check_truncation(results)

        usage = getattr(results, "usage", None)
        if (
            not getattr(results, "cache_hit", False)
            and dspy.settings.usage_tracker
            and usage
        ):
            dspy.settings.usage_tracker.add_usage(self.model, dict(usage))
        return results

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        *,
        codex_transport: CodexTransport | None = None,
        codex_websocket_connect_timeout: float | None = None,
        codex_websocket_idle_timeout: float | None = None,
        **kwargs: Any,
    ) -> Any:
        if not self._uses_codex_route:
            if (
                codex_transport is not None
                or codex_websocket_connect_timeout is not None
                or codex_websocket_idle_timeout is not None
            ):
                raise ValueError("Codex transport overrides require a Codex LM route")
            return super().forward(prompt=prompt, messages=messages, **kwargs)

        request, cache = self._prepare_completion(
            prompt,
            messages,
            codex_transport,
            codex_websocket_connect_timeout,
            codex_websocket_idle_timeout,
            kwargs,
        )
        completion, litellm_cache_args = self._get_cached_completion_fn(
            _codex_completion, cache
        )
        results = completion(
            request=request,
            auth_storage=self.auth_storage,
            num_retries=self.num_retries,
            cache=litellm_cache_args,
        )
        return self._record_usage(results)

    async def aforward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        *,
        codex_transport: CodexTransport | None = None,
        codex_websocket_connect_timeout: float | None = None,
        codex_websocket_idle_timeout: float | None = None,
        **kwargs: Any,
    ) -> Any:
        if not self._uses_codex_route:
            if (
                codex_transport is not None
                or codex_websocket_connect_timeout is not None
                or codex_websocket_idle_timeout is not None
            ):
                raise ValueError("Codex transport overrides require a Codex LM route")
            return await super().aforward(prompt=prompt, messages=messages, **kwargs)

        request, cache = self._prepare_completion(
            prompt,
            messages,
            codex_transport,
            codex_websocket_connect_timeout,
            codex_websocket_idle_timeout,
            kwargs,
        )
        completion, litellm_cache_args = self._get_cached_completion_fn(
            _acodex_completion, cache
        )
        results = await completion(
            request=request,
            auth_storage=self.auth_storage,
            num_retries=self.num_retries,
            cache=litellm_cache_args,
        )
        return self._record_usage(results)


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


register_model_alias(("codex", "chatgpt", OPENAI_CODEX_PROVIDER), _resolve_codex_route)


__all__ = [
    "CodexTransport",
    "DEFAULT_CODEX_API_BASE",
    "DEFAULT_CODEX_INSTRUCTIONS",
    "DEFAULT_CODEX_MODEL",
    "DEFAULT_CODEX_ORIGINATOR",
    "DEFAULT_CODEX_WEBSOCKET_CONNECT_TIMEOUT",
    "DEFAULT_CODEX_WEBSOCKET_IDLE_TIMEOUT",
    "LM",
    "RouteRegistration",
    "install",
    "register_model_alias",
    "resolve_lm_route",
    "uninstall",
    "unregister_model_alias",
]
