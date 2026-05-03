from __future__ import annotations

import warnings
from collections import defaultdict
from types import SimpleNamespace
from typing import Any

import dspy
import dspy_lm_auth
import litellm

from dspy_lm_auth.auth import AuthStorage
from dspy_lm_auth.lm import (
    DEFAULT_CODEX_API_BASE,
    DEFAULT_CODEX_MODEL,
    LM as AuthLM,
    _add_dspy_identifier_to_headers,
    _build_codex_responses_request,
    getauthtoken,
)

DEFAULT_CODEX_ORIGINATOR = "dspy_codex_auth"


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

    try:
        response.output = output_items
    except Exception:
        if hasattr(response, "model_copy"):
            return response.model_copy(update={"output": output_items})
        response.__dict__["output"] = output_items
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


def _consume_codex_response_stream(response_stream: Any) -> Any:
    if not hasattr(response_stream, "completed_response"):
        return response_stream

    builder = _StreamOutputBuilder()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Pydantic serializer warnings:*", category=UserWarning
        )
        for event in response_stream:
            builder.add(event)

    completed_event = getattr(response_stream, "completed_response", None)
    completed_response = getattr(completed_event, "response", None)
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
        async for event in response_stream:
            builder.add(event)

    completed_event = getattr(response_stream, "completed_response", None)
    completed_response = getattr(completed_event, "response", None)
    if completed_response is None:
        raise RuntimeError("Codex response stream ended without a completed response")
    return _reconstruct_stream_output(completed_response, builder)


def _build_codex_request(request: dict[str, Any]) -> dict[str, Any]:
    request = dict(request)
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

    response_stream = litellm.responses(
        cache=cache,
        num_retries=num_retries,
        retry_strategy="exponential_backoff_retry",
        headers=_add_dspy_identifier_to_headers(headers),
        **request,
    )
    return _consume_codex_response_stream(response_stream)


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

    response_stream = await litellm.aresponses(
        cache=cache,
        num_retries=num_retries,
        retry_strategy="exponential_backoff_retry",
        headers=_add_dspy_identifier_to_headers(headers),
        **request,
    )
    return await _aconsume_codex_response_stream(response_stream)


class LM(AuthLM):
    """DSPy LM with Codex subscription stream compatibility fixes."""

    def __init__(self, model: str, *args: Any, **kwargs: Any) -> None:
        auth_provider = kwargs.get("auth_provider")
        if model.split("/", 1)[0] in {
            "codex",
            "chatgpt",
            "openai-codex",
        } or auth_provider in {
            "codex",
            "chatgpt",
            "openai-codex",
        }:
            kwargs.setdefault("originator", DEFAULT_CODEX_ORIGINATOR)
        super().__init__(model, *args, **kwargs)

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        if not self._uses_codex_route:
            return super().forward(prompt=prompt, messages=messages, **kwargs)

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
            _codex_responses_completion, cache
        )
        results = completion(
            request=dict(model=self.model, messages=messages, **kwargs),
            num_retries=self.num_retries,
            cache=litellm_cache_args,
        )

        self._check_truncation(results)

        if (
            not getattr(results, "cache_hit", False)
            and dspy.settings.usage_tracker
            and hasattr(results, "usage")
        ):
            dspy.settings.usage_tracker.add_usage(self.model, dict(results.usage))
        return results

    async def aforward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        if not self._uses_codex_route:
            return await super().aforward(prompt=prompt, messages=messages, **kwargs)

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
            _acodex_responses_completion, cache
        )
        results = await completion(
            request=dict(model=self.model, messages=messages, **kwargs),
            num_retries=self.num_retries,
            cache=litellm_cache_args,
        )

        self._check_truncation(results)

        if (
            not getattr(results, "cache_hit", False)
            and dspy.settings.usage_tracker
            and hasattr(results, "usage")
        ):
            dspy.settings.usage_tracker.add_usage(self.model, dict(results.usage))
        return results


def install(
    *, auth_storage: AuthStorage | str | None = None, attach_helpers: bool = True
) -> type[LM]:
    dspy_lm_auth.install(auth_storage=auth_storage, attach_helpers=attach_helpers)
    dspy.LM = LM
    dspy.clients.LM = LM
    if attach_helpers:
        dspy.getauthtoken = getauthtoken
    return LM


def uninstall() -> None:
    dspy_lm_auth.uninstall()


__all__ = [
    "DEFAULT_CODEX_API_BASE",
    "DEFAULT_CODEX_MODEL",
    "DEFAULT_CODEX_ORIGINATOR",
    "LM",
    "install",
    "uninstall",
]
