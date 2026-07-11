"""DSPy LM integration for ChatGPT/Codex subscription credentials.

Portions are adapted from dspy-lm-auth under the MIT License. See
THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal, cast

import dspy
import litellm
from litellm.types.responses.main import OutputFunctionToolCall

from dspy_codex_auth.auth import (
    OPENAI_CODEX_PROVIDER,
    AuthStorage,
    extract_chatgpt_account_id,
    get_default_auth_storage,
    getauthtoken,
    normalize_provider_id,
    set_default_auth_storage,
)
from dspy_codex_auth.responses_websocket import (
    DEFAULT_CODEX_WEBSOCKET_CONNECT_TIMEOUT,
    DEFAULT_CODEX_WEBSOCKET_IDLE_TIMEOUT,
    CodexWebSocketError,
    CodexWebSocketProtocolError,
    CodexWebSocketResponseError,
    WebSocketResult,
    awebsocket_response,
    websocket_response,
)

DEFAULT_CODEX_MODEL = "gpt-5.4"
DEFAULT_CODEX_API_BASE = "https://chatgpt.com/backend-api/codex"
DEFAULT_CODEX_ORIGINATOR = "dspy_codex_auth"
DEFAULT_CODEX_INSTRUCTIONS = "You are a helpful assistant."

type CodexTransport = Literal["auto", "http", "websocket"]

_CODEX_TRANSPORTS = {"auto", "http", "websocket"}
_CODEX_CACHE_CONTROL_KEY = "_dspy_codex_transport_controls"
_CODEX_MODEL_NOT_FOUND_PREFIX = "litellm.BadRequestError: OpenAIException - "

_DSPY_LM = dspy.LM
_ORIGINAL_DSPY_LM = dspy.LM

RouteResolver = Callable[[str, dict[str, Any], AuthStorage], tuple[str, dict[str, Any]]]
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


def _codex_cache_request(
    request: dict[str, Any],
    *,
    codex_transport: CodexTransport,
    codex_websocket_connect_timeout: float,
    codex_websocket_idle_timeout: float,
) -> dict[str, Any]:
    return {
        **request,
        _CODEX_CACHE_CONTROL_KEY: {
            "transport": codex_transport,
            "connect_timeout": codex_websocket_connect_timeout,
            "idle_timeout": codex_websocket_idle_timeout,
        },
    }


def _codex_provider_request(request: dict[str, Any]) -> dict[str, Any]:
    provider_request = dict(request)
    provider_request.pop(_CODEX_CACHE_CONTROL_KEY, None)
    return provider_request


@dataclass(frozen=True, slots=True)
class RouteRegistration:
    aliases: tuple[str, ...]
    resolver: RouteResolver


def _coerce_auth_storage(
    auth_storage: AuthStorage | str | os.PathLike[str] | None,
) -> AuthStorage:
    if auth_storage is None:
        return get_default_auth_storage()
    if isinstance(auth_storage, AuthStorage):
        return auth_storage
    return AuthStorage(auth_storage)


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


def codex_headers(
    token: str,
    *,
    account_id: str | None = None,
    originator: str = DEFAULT_CODEX_ORIGINATOR,
    extra_headers: dict[str, Any] | None = None,
) -> dict[str, str]:
    resolved_account_id = account_id or extract_chatgpt_account_id(token)
    headers = {
        "chatgpt-account-id": resolved_account_id,
        "OpenAI-Beta": "responses=experimental",
        "originator": originator,
    }
    if extra_headers:
        headers.update({str(key): str(value) for key, value in extra_headers.items()})
    return headers


def _resolve_codex_route(
    model: str,
    kwargs: dict[str, Any],
    auth_storage: AuthStorage,
) -> tuple[str, dict[str, Any]]:
    if "/" in model:
        _, model_id = model.split("/", 1)
    else:
        model_id = DEFAULT_CODEX_MODEL

    resolved_kwargs = dict(kwargs)
    token = resolved_kwargs.get("api_key") or auth_storage.get_api_key(
        OPENAI_CODEX_PROVIDER
    )
    if not token:
        raise ValueError(
            "No OpenAI Codex credential found. Run `dspy_codex_auth.login()`, "
            "reuse Pi's auth.json, or pass `api_key=` explicitly."
        )

    credential = auth_storage.get(OPENAI_CODEX_PROVIDER)
    account_id = resolved_kwargs.pop("chatgpt_account_id", None)
    if account_id is None and isinstance(credential, dict):
        raw_account_id = credential.get("accountId")
        if isinstance(raw_account_id, str) and raw_account_id:
            account_id = raw_account_id

    originator = str(resolved_kwargs.pop("originator", DEFAULT_CODEX_ORIGINATOR))
    headers = codex_headers(
        token,
        account_id=account_id,
        originator=originator,
        extra_headers=resolved_kwargs.get("headers"),
    )

    resolved_kwargs["headers"] = headers
    resolved_kwargs.setdefault("api_key", token)
    resolved_kwargs.setdefault("api_base", DEFAULT_CODEX_API_BASE)
    resolved_kwargs.setdefault("model_type", "responses")
    resolved_kwargs.setdefault("use_developer_role", True)
    return f"openai/{model_id}", resolved_kwargs


def resolve_lm_route(
    model: str,
    *,
    auth_storage: AuthStorage,
    auth_provider: str | None = None,
    kwargs: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    resolved_kwargs = dict(kwargs or {})

    if auth_provider:
        provider = normalize_provider_id(auth_provider)
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


def _add_dspy_identifier_to_headers(
    headers: dict[str, Any] | None = None,
) -> dict[str, Any]:
    headers = headers or {}
    return {
        "User-Agent": f"DSPy/{dspy.__version__}",
        **headers,
    }


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
    return response_format


def _normalize_codex_service_tier(service_tier: Any) -> Any:
    if not isinstance(service_tier, str):
        return service_tier

    normalized = service_tier.lower()
    if normalized == "fast":
        return "priority"
    if normalized in {"priority", "flex"}:
        return normalized
    return service_tier


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

        input_messages.append(
            {
                "role": role,
                "content": _convert_message_content_to_responses_format(
                    content,
                    role=role,
                ),
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

    if "service_tier" in request:
        service_tier = request.pop("service_tier")
        if service_tier is not None:
            request["service_tier"] = _normalize_codex_service_tier(service_tier)

    if "response_format" in request:
        response_format = _coerce_response_format(request.pop("response_format"))
        text = request.pop("text", {}) or {}
        request["text"] = {**text, "format": response_format}

    request["store"] = False
    request["stream"] = True
    return request


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _event_name(event: Any) -> str:
    event_type = _field(event, "type", "")
    return getattr(event_type, "value", str(event_type))


def _message_item(texts: list[str]) -> SimpleNamespace | None:
    content = [SimpleNamespace(text=text) for text in texts if text]
    if not content:
        return None
    return SimpleNamespace(type="message", content=content)


def _reasoning_item(texts: list[str]) -> SimpleNamespace | None:
    summary = [SimpleNamespace(text=text) for text in texts if text]
    if not summary:
        return None
    return SimpleNamespace(type="reasoning", summary=summary)


def _coerce_output_item_for_dspy(output_item: Any) -> Any:
    output_item_type = getattr(
        _field(output_item, "type"), "value", _field(output_item, "type")
    )
    if output_item_type == "function_call":
        if isinstance(output_item, dict):
            return OutputFunctionToolCall(**output_item)
        return output_item
    if output_item_type == "reasoning":
        texts: list[str] = []
        for summary_item in _field(output_item, "summary", []) or []:
            text = _field(summary_item, "text")
            if isinstance(text, str) and text:
                texts.append(text)
        return _reasoning_item(texts)
    if output_item_type != "message":
        return None

    texts: list[str] = []
    for content_item in _field(output_item, "content", []) or []:
        text = _field(content_item, "text")
        if isinstance(text, str) and text:
            texts.append(text)
    return _message_item(texts)


class _StreamOutputBuilder:
    def __init__(self) -> None:
        self._done_items_by_index: dict[int, Any] = {}
        self._done_text_by_index: dict[tuple[int, int], str] = {}
        self._delta_text_by_index: defaultdict[tuple[int, int], list[str]] = (
            defaultdict(list)
        )
        self._done_reasoning_summary_by_index: dict[tuple[int, int], str] = {}
        self._delta_reasoning_summary_by_index: defaultdict[
            tuple[int, int], list[str]
        ] = defaultdict(list)

    def add(self, event: Any) -> None:
        event_name = _event_name(event)

        if event_name == "response.output_item.done":
            output_index = int(
                _field(event, "output_index", len(self._done_items_by_index))
            )
            item = _coerce_output_item_for_dspy(_field(event, "item"))
            if item is not None:
                self._done_items_by_index[output_index] = item
            return

        if event_name == "response.output_text.done":
            output_index = int(_field(event, "output_index", 0))
            content_index = int(_field(event, "content_index", 0))
            text = _field(event, "text")
            if isinstance(text, str):
                self._done_text_by_index[(output_index, content_index)] = text
            return

        if event_name == "response.output_text.delta":
            output_index = int(_field(event, "output_index", 0))
            content_index = int(_field(event, "content_index", 0))
            delta = _field(event, "delta")
            if isinstance(delta, str):
                self._delta_text_by_index[(output_index, content_index)].append(delta)
            return

        if event_name == "response.reasoning_summary_text.done":
            output_index = int(_field(event, "output_index", 0))
            summary_index = int(_field(event, "summary_index", 0))
            text = _field(event, "text")
            if isinstance(text, str):
                self._done_reasoning_summary_by_index[(output_index, summary_index)] = (
                    text
                )
            return

        if event_name == "response.reasoning_summary_text.delta":
            output_index = int(_field(event, "output_index", 0))
            summary_index = int(_field(event, "summary_index", 0))
            delta = _field(event, "delta")
            if isinstance(delta, str):
                self._delta_reasoning_summary_by_index[
                    (output_index, summary_index)
                ].append(delta)

    def output_items(self) -> list[Any]:
        output_items_by_index = dict(self._done_items_by_index)

        text_by_index = self._done_text_by_index or {
            key: "".join(parts) for key, parts in self._delta_text_by_index.items()
        }
        texts_by_output_index: defaultdict[int, list[tuple[int, str]]] = defaultdict(
            list
        )
        for (output_index, content_index), text in text_by_index.items():
            texts_by_output_index[output_index].append((content_index, text))

        for output_index in sorted(texts_by_output_index):
            texts = [text for _, text in sorted(texts_by_output_index[output_index])]
            item = _message_item(texts)
            if item is not None:
                output_items_by_index.setdefault(output_index, item)

        reasoning_by_index = self._done_reasoning_summary_by_index or {
            key: "".join(parts)
            for key, parts in self._delta_reasoning_summary_by_index.items()
        }
        reasoning_texts_by_output_index: defaultdict[int, list[tuple[int, str]]] = (
            defaultdict(list)
        )
        for (output_index, summary_index), text in reasoning_by_index.items():
            reasoning_texts_by_output_index[output_index].append((summary_index, text))

        for output_index in sorted(reasoning_texts_by_output_index):
            texts = [
                text
                for _, text in sorted(reasoning_texts_by_output_index[output_index])
            ]
            item = _reasoning_item(texts)
            if item is not None:
                output_items_by_index.setdefault(output_index, item)

        return [item for _, item in sorted(output_items_by_index.items())]


def _set_response_output(response: Any, output_items: list[Any]) -> Any:
    if not output_items:
        return response

    if isinstance(response, dict):
        response = SimpleNamespace(**response)

    try:
        response.output = output_items
    except Exception:
        if hasattr(response, "model_copy"):
            return response.model_copy(update={"output": output_items})
        response.__dict__["output"] = output_items
    return response


def _ensure_response_usage(response: Any) -> Any:
    if _field(response, "usage") is not None:
        return response

    if isinstance(response, dict):
        response = SimpleNamespace(**response)

    try:
        response.usage = {}
    except Exception:
        if hasattr(response, "model_copy"):
            return response.model_copy(update={"usage": {}})
        response.__dict__["usage"] = {}
    return response


def _normalise_existing_output(response: Any) -> list[Any]:
    output_items: list[Any] = []
    for output_item in _field(response, "output", []) or []:
        item = _coerce_output_item_for_dspy(output_item)
        if item is not None:
            output_items.append(item)
    return output_items


def _reconstruct_stream_output(response: Any, builder: _StreamOutputBuilder) -> Any:
    existing_output = _normalise_existing_output(response)
    if existing_output:
        return _set_response_output(response, existing_output)
    return _set_response_output(response, builder.output_items())


def _response_has_dspy_output(response: Any) -> bool:
    for output_item in _field(response, "output", []) or []:
        output_item_type = getattr(
            _field(output_item, "type"), "value", _field(output_item, "type")
        )
        if output_item_type == "function_call":
            return True
        if output_item_type != "message":
            continue
        for content_item in _field(output_item, "content", []) or []:
            text = _field(content_item, "text")
            if isinstance(text, str) and text:
                return True
    return False


def _is_retryable_stream_error(exc: Exception) -> bool:
    text = f"{type(exc).__module__}.{type(exc).__name__}: {exc}"
    return any(
        marker in text
        for marker in (
            "RemoteProtocolError",
            "ReadError",
            "ConnectError",
            "incomplete chunked read",
            "peer closed connection",
        )
    )


def _structured_litellm_error(exc: Exception) -> dict[str, Any] | None:
    if getattr(exc, "body", None) is not None:
        return None
    message = getattr(exc, "message", None)
    if not isinstance(message, str) or not message.startswith(
        _CODEX_MODEL_NOT_FOUND_PREFIX
    ):
        return None
    try:
        payload = json.loads(message[len(_CODEX_MODEL_NOT_FOUND_PREFIX) :])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or set(payload) != {"error"}:
        return None
    return payload


def _is_exact_codex_model_not_found(exc: Exception, model: str) -> bool:
    wire_model = model.removeprefix("openai/")
    if not isinstance(exc, litellm.BadRequestError):
        return False
    if getattr(exc, "status_code", None) != 404:
        return False
    if getattr(exc, "llm_provider", None) != "openai":
        return False
    if getattr(exc, "model", None) != wire_model:
        return False

    payload = _structured_litellm_error(exc)
    if payload is None:
        return False
    error = payload.get("error")
    if not isinstance(error, dict) or set(error) != {
        "message",
        "type",
        "param",
        "code",
    }:
        return False
    return (
        error.get("message") == f"Model not found {wire_model}"
        and error.get("type") == "invalid_request_error"
        and error.get("param") == "model"
        and "code" in error
        and error["code"] is None
    )


def _completed_response_from_stream(response_stream: Any) -> Any:
    completed_event = getattr(response_stream, "completed_response", None)
    return getattr(completed_event, "response", None)


def _is_recoverable_completed_response_logging_error(
    exc: Exception, response_stream: Any
) -> bool:
    if _completed_response_from_stream(response_stream) is None:
        return False
    text = f"{type(exc).__module__}.{type(exc).__name__}: {exc}"
    return "'dict' object has no attribute 'usage'" in text


def _consume_codex_response_stream(response_stream: Any) -> Any:
    if not hasattr(response_stream, "completed_response"):
        return response_stream

    builder = _StreamOutputBuilder()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Pydantic serializer warnings:*", category=UserWarning
        )
        try:
            for event in response_stream:
                builder.add(event)
        except Exception as exc:
            if not _is_recoverable_completed_response_logging_error(
                exc, response_stream
            ):
                raise

    completed_response = _completed_response_from_stream(response_stream)
    if completed_response is None:
        raise RuntimeError("Codex response stream ended without a completed response")
    return _reconstruct_stream_output(completed_response, builder)


async def _aconsume_codex_response_stream(response_stream: Any) -> Any:
    if not hasattr(response_stream, "completed_response"):
        return response_stream

    builder = _StreamOutputBuilder()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Pydantic serializer warnings:*", category=UserWarning
        )
        try:
            async for event in response_stream:
                builder.add(event)
        except Exception as exc:
            if not _is_recoverable_completed_response_logging_error(
                exc, response_stream
            ):
                raise

    completed_response = _completed_response_from_stream(response_stream)
    if completed_response is None:
        raise RuntimeError("Codex response stream ended without a completed response")
    return _reconstruct_stream_output(completed_response, builder)


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

    request.pop("max_tokens", None)
    request.pop("max_output_tokens", None)
    request.pop("max_completion_tokens", None)

    if reasoning_summary is not None:
        reasoning = dict(request.pop("reasoning", {}) or {})
        reasoning["summary"] = reasoning_summary
        request["reasoning"] = reasoning
    request.pop("reasoning_summary", None)
    return request


def _codex_responses_completion(
    request: dict[str, Any],
    num_retries: int,
    cache: dict[str, Any] | None = None,
) -> Any:
    cache = cache or {"no-cache": True, "no-store": True}
    request = dict(request)
    request.pop("rollout_id", None)
    headers = request.pop("headers", None)
    request = _build_codex_request(request)

    max_attempts = max(1, num_retries + 1)
    last_empty_response: Any = None
    for attempt in range(max_attempts):
        try:
            response_stream = litellm.responses(
                cache=cache,
                num_retries=num_retries,
                retry_strategy="exponential_backoff_retry",
                headers=_add_dspy_identifier_to_headers(headers),
                **request,
            )
            response = _consume_codex_response_stream(response_stream)
        except Exception as exc:
            if attempt < max_attempts - 1 and _is_retryable_stream_error(exc):
                continue
            raise

        if _response_has_dspy_output(response):
            return response
        last_empty_response = response

    raise RuntimeError(
        "Codex response completed without message text or tool call "
        f"after {max_attempts} attempt(s): {last_empty_response!r}"
    )


async def _acodex_responses_completion(
    request: dict[str, Any],
    num_retries: int,
    cache: dict[str, Any] | None = None,
) -> Any:
    cache = cache or {"no-cache": True, "no-store": True}
    request = dict(request)
    request.pop("rollout_id", None)
    headers = request.pop("headers", None)
    request = _build_codex_request(request)

    max_attempts = max(1, num_retries + 1)
    last_empty_response: Any = None
    for attempt in range(max_attempts):
        try:
            response_stream = await litellm.aresponses(
                cache=cache,
                num_retries=num_retries,
                retry_strategy="exponential_backoff_retry",
                headers=_add_dspy_identifier_to_headers(headers),
                **request,
            )
            response = await _aconsume_codex_response_stream(response_stream)
        except Exception as exc:
            if attempt < max_attempts - 1 and _is_retryable_stream_error(exc):
                continue
            raise

        if _response_has_dspy_output(response):
            return response
        last_empty_response = response

    raise RuntimeError(
        "Codex response completed without message text or tool call "
        f"after {max_attempts} attempt(s): {last_empty_response!r}"
    )


def _prepare_codex_websocket_request(
    request: dict[str, Any],
) -> tuple[dict[str, Any], str, str, dict[str, Any]]:
    request = dict(request)
    request.pop("rollout_id", None)
    headers = request.pop("headers", None)
    api_key = request.pop("api_key", None)
    if not isinstance(api_key, str) or not api_key:
        raise ValueError("Codex WebSocket transport requires a non-empty api_key")
    api_base = request.pop("api_base", DEFAULT_CODEX_API_BASE)
    if not isinstance(api_base, str) or not api_base:
        raise ValueError("Codex WebSocket transport requires a non-empty api_base")
    request.pop("model_type", None)
    request.pop("use_developer_role", None)

    wire_request = {
        key: value
        for key, value in _build_codex_request(request).items()
        if value is not None
    }
    return (
        wire_request,
        api_key,
        api_base,
        _add_dspy_identifier_to_headers(headers),
    )


def _consume_codex_websocket_result(result: WebSocketResult) -> Any:
    builder = _StreamOutputBuilder()
    for event in result.events:
        builder.add(event)
    return _reconstruct_stream_output(result.response, builder)


def _is_retryable_codex_websocket_error(exc: Exception) -> bool:
    return isinstance(exc, CodexWebSocketError) and not isinstance(
        exc,
        (CodexWebSocketProtocolError, CodexWebSocketResponseError),
    )


def _codex_websocket_completion(
    request: dict[str, Any],
    num_retries: int,
    connect_timeout: float,
    idle_timeout: float,
) -> Any:
    wire_request, api_key, api_base, headers = _prepare_codex_websocket_request(request)
    max_attempts = max(1, num_retries + 1)
    last_empty_response: Any = None
    for attempt in range(max_attempts):
        try:
            result = websocket_response(
                wire_request,
                api_base=api_base,
                api_key=api_key,
                headers=headers,
                connect_timeout=connect_timeout,
                idle_timeout=idle_timeout,
            )
            response = _consume_codex_websocket_result(result)
        except Exception as exc:
            if attempt < max_attempts - 1 and _is_retryable_codex_websocket_error(exc):
                continue
            raise

        if _response_has_dspy_output(response):
            return response
        last_empty_response = response

    raise RuntimeError(
        "Codex WebSocket response completed without message text or tool call "
        f"after {max_attempts} attempt(s): {last_empty_response!r}"
    )


async def _acodex_websocket_completion(
    request: dict[str, Any],
    num_retries: int,
    connect_timeout: float,
    idle_timeout: float,
) -> Any:
    wire_request, api_key, api_base, headers = _prepare_codex_websocket_request(request)
    max_attempts = max(1, num_retries + 1)
    last_empty_response: Any = None
    for attempt in range(max_attempts):
        try:
            result = await awebsocket_response(
                wire_request,
                api_base=api_base,
                api_key=api_key,
                headers=headers,
                connect_timeout=connect_timeout,
                idle_timeout=idle_timeout,
            )
            response = _consume_codex_websocket_result(result)
        except Exception as exc:
            if attempt < max_attempts - 1 and _is_retryable_codex_websocket_error(exc):
                continue
            raise

        if _response_has_dspy_output(response):
            return response
        last_empty_response = response

    raise RuntimeError(
        "Codex WebSocket response completed without message text or tool call "
        f"after {max_attempts} attempt(s): {last_empty_response!r}"
    )


def _codex_completion(
    request: dict[str, Any],
    num_retries: int,
    codex_transport: CodexTransport,
    codex_websocket_connect_timeout: float,
    codex_websocket_idle_timeout: float,
    cache: dict[str, Any] | None = None,
) -> Any:
    request = _codex_provider_request(request)
    if codex_transport == "websocket":
        return _codex_websocket_completion(
            request,
            num_retries,
            codex_websocket_connect_timeout,
            codex_websocket_idle_timeout,
        )
    if codex_transport == "http":
        return _codex_responses_completion(request, num_retries, cache)

    try:
        return _codex_responses_completion(request, num_retries, cache)
    except Exception as exc:
        if not _is_exact_codex_model_not_found(exc, str(request.get("model", ""))):
            raise
    return _codex_websocket_completion(
        request,
        num_retries,
        codex_websocket_connect_timeout,
        codex_websocket_idle_timeout,
    )


async def _acodex_completion(
    request: dict[str, Any],
    num_retries: int,
    codex_transport: CodexTransport,
    codex_websocket_connect_timeout: float,
    codex_websocket_idle_timeout: float,
    cache: dict[str, Any] | None = None,
) -> Any:
    request = _codex_provider_request(request)
    if codex_transport == "websocket":
        return await _acodex_websocket_completion(
            request,
            num_retries,
            codex_websocket_connect_timeout,
            codex_websocket_idle_timeout,
        )
    if codex_transport == "http":
        return await _acodex_responses_completion(request, num_retries, cache)

    try:
        return await _acodex_responses_completion(request, num_retries, cache)
    except Exception as exc:
        if not _is_exact_codex_model_not_found(exc, str(request.get("model", ""))):
            raise
    return await _acodex_websocket_completion(
        request,
        num_retries,
        codex_websocket_connect_timeout,
        codex_websocket_idle_timeout,
    )


class LM(_DSPY_LM):
    """DSPy LM with Codex subscription auth and stream compatibility fixes."""

    def __init__(
        self,
        model: str,
        *args: Any,
        auth_storage: AuthStorage | str | os.PathLike[str] | None = None,
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
        requested_route = normalize_provider_id(
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
        super().__init__(resolved_model, *args, **resolved_kwargs)

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
        kwargs = dict(kwargs)
        cache = kwargs.pop("cache", self.cache)

        messages = messages or [{"role": "user", "content": prompt}]
        kwargs = {**self.kwargs, **kwargs}
        self._warn_zero_temp_rollout(
            kwargs.get("temperature"), kwargs.get("rollout_id")
        )
        if kwargs.get("rollout_id") is None:
            kwargs.pop("rollout_id", None)

        completion, litellm_cache_args = self._get_cached_completion_fn(
            _codex_completion, cache
        )
        results = completion(
            request=_codex_cache_request(
                dict(model=self.model, messages=messages, **kwargs),
                codex_transport=selected_transport,
                codex_websocket_connect_timeout=selected_connect_timeout,
                codex_websocket_idle_timeout=selected_idle_timeout,
            ),
            num_retries=self.num_retries,
            codex_transport=selected_transport,
            codex_websocket_connect_timeout=selected_connect_timeout,
            codex_websocket_idle_timeout=selected_idle_timeout,
            cache=litellm_cache_args,
        )
        results = _ensure_response_usage(results)

        self._check_truncation(results)

        usage = getattr(results, "usage", None)
        if (
            not getattr(results, "cache_hit", False)
            and dspy.settings.usage_tracker
            and usage
        ):
            dspy.settings.usage_tracker.add_usage(self.model, dict(usage))
        return results

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
        kwargs = dict(kwargs)
        cache = kwargs.pop("cache", self.cache)

        messages = messages or [{"role": "user", "content": prompt}]
        kwargs = {**self.kwargs, **kwargs}
        self._warn_zero_temp_rollout(
            kwargs.get("temperature"), kwargs.get("rollout_id")
        )
        if kwargs.get("rollout_id") is None:
            kwargs.pop("rollout_id", None)

        completion, litellm_cache_args = self._get_cached_completion_fn(
            _acodex_completion, cache
        )
        results = await completion(
            request=_codex_cache_request(
                dict(model=self.model, messages=messages, **kwargs),
                codex_transport=selected_transport,
                codex_websocket_connect_timeout=selected_connect_timeout,
                codex_websocket_idle_timeout=selected_idle_timeout,
            ),
            num_retries=self.num_retries,
            codex_transport=selected_transport,
            codex_websocket_connect_timeout=selected_connect_timeout,
            codex_websocket_idle_timeout=selected_idle_timeout,
            cache=litellm_cache_args,
        )
        results = _ensure_response_usage(results)

        self._check_truncation(results)

        usage = getattr(results, "usage", None)
        if (
            not getattr(results, "cache_hit", False)
            and dspy.settings.usage_tracker
            and usage
        ):
            dspy.settings.usage_tracker.add_usage(self.model, dict(usage))
        return results


def install(
    *,
    auth_storage: AuthStorage | str | os.PathLike[str] | None = None,
    attach_helpers: bool = True,
) -> type[LM]:
    storage = _coerce_auth_storage(auth_storage)
    set_default_auth_storage(storage)

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
    "codex_headers",
    "install",
    "register_model_alias",
    "resolve_lm_route",
    "uninstall",
    "unregister_model_alias",
]
