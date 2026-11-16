"""Single-attempt Codex engines for DSPy's canonical LM interface.

DSPy's public Responses dialect supplies the wire codec. Authentication and
transport belong to the shared Codex client; execution policy belongs to DSPy.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from dataclasses import replace
from typing import Any

import httpx
from dspy.lm15 import (
    Adaptation,
    AuthError,
    InvalidRequestError,
    OpenAILM,
    ProviderError,
    RateLimitError,
    Request,
    Response,
    ServerError,
    StreamAssemblyError,
    TimeoutError,
    TransportError,
    UnsupportedFeatureError,
    providers,
)
from openai_codex_auth import (
    DEFAULT_CODEX_INSTRUCTIONS,
    CodexAuthError,
    CodexEmptyResponseError,
    CodexError,
    CodexHTTPError,
    CodexProtocolError,
    CodexResponse,
)
from openai_codex_auth.responses_websocket import (
    CodexWebSocketConnectionError,
    CodexWebSocketProtocolError,
    CodexWebSocketResponseError,
    CodexWebSocketTimeoutError,
)

from .connection import CACHE_IDENTITY_KEY, CodexConnection, reject_generation_options


def _provider_error(
    error: CodexHTTPError | CodexWebSocketResponseError,
) -> ProviderError:
    status = error.status_code
    kind: type[ProviderError]
    if status in {401, 403}:
        kind = AuthError
    elif status == 429:
        kind = RateLimitError
    elif status in {408, 504}:
        kind = TimeoutError
    elif status is not None and 500 <= status <= 599:
        kind = ServerError
    elif status is not None and 400 <= status <= 499:
        kind = InvalidRequestError
    else:
        kind = ProviderError
    return kind(str(error), provider="openai-codex", status=status)


@contextmanager
def _codex_errors() -> Iterator[None]:
    """Translate typed client failures without interpreting exception prose."""
    try:
        yield
    except CodexAuthError as error:
        raise AuthError(str(error), provider="openai-codex") from error
    except (CodexHTTPError, CodexWebSocketResponseError) as error:
        raise _provider_error(error) from error
    except (CodexProtocolError, CodexWebSocketProtocolError) as error:
        raise StreamAssemblyError(str(error), provider="openai-codex") from error
    except (httpx.TimeoutException, CodexWebSocketTimeoutError) as error:
        raise TimeoutError(str(error), provider="openai-codex") from error
    except (httpx.TransportError, CodexWebSocketConnectionError) as error:
        raise TransportError(str(error), provider="openai-codex") from error
    except CodexEmptyResponseError as error:
        raise ServerError(str(error), provider="openai-codex") from error
    except CodexError as error:
        # openai-codex-auth's HTTP transport emits this exact base class after
        # exhausting its attempt budget; protocol/provider failures have types.
        if type(error) is CodexError:
            raise TransportError(str(error), provider="openai-codex") from error
        raise


def _encode(
    codec: OpenAILM, request: Request
) -> tuple[Request, dict[str, Any], tuple[Adaptation, ...]]:
    extensions = dict(request.config.extensions or {})
    extensions.pop(CACHE_IDENTITY_KEY, None)
    reject_generation_options(extensions)
    if request.config.stop:
        raise UnsupportedFeatureError(
            "Codex completion engines do not support stop sequences; "
            "they collect the complete response before returning it.",
            provider="openai-codex",
            feature="config.stop",
        )
    summary = extensions.pop("reasoning_summary", None)
    text = extensions.pop("text", None) or {}
    instructions = extensions.pop("instructions", None)
    reasoning = request.config.reasoning
    if reasoning is not None and summary is not None:
        reasoning = replace(reasoning, summary=summary)
    wire_request = replace(
        request,
        model=request.model.removeprefix("openai/"),
        config=replace(
            request.config, reasoning=reasoning, extensions=extensions or None
        ),
    )
    adaptations = codec.plan(wire_request)
    payload = json.loads(codec.build_request(wire_request, stream=True).body)
    if text:
        payload["text"] = {**text, **payload.get("text", {})}
    if summary is not None and reasoning is None:
        payload["reasoning"] = {"summary": summary}
    parts = [part for part in (instructions, payload.get("instructions")) if part]
    payload["instructions"] = (
        "\n\n".join(parts) if parts else DEFAULT_CODEX_INSTRUCTIONS
    )
    if request.config.max_tokens is not None:
        adaptations += (
            Adaptation(
                field="config.max_tokens",
                action="dropped",
                asked=request.config.max_tokens,
                reason="The Codex backend rejects output-token caps; the shared client removes them.",
            ),
        )
    return wire_request, payload, adaptations


def _decode(
    codec: OpenAILM,
    request: Request,
    result: CodexResponse,
    adaptations: tuple[Adaptation, ...],
) -> Response:
    # The general codec renders an empty refusal as empty text. Preserve the
    # provider's refusal discriminant even when it supplies no explanation.
    refusals = [
        part.get("refusal", "")
        for item in result.output
        if item.get("type") == "message"
        for part in item.get("content", [])
        if part.get("type") == "refusal"
    ]
    if refusals:
        raise ProviderError(
            "Codex refused the request: " + "\n".join(refusals),
            provider="openai-codex",
        )
    response = codec.parse_response(
        request,
        providers.HttpResponse(
            status=200,
            reason="OK",
            headers=[],
            body=json.dumps(result.response).encode(),
        ),
    )
    return replace(
        response,
        provider_data={
            **(response.provider_data or {}),
            "codex_response": result.response,
            "codex_transport": result.transport,
        },
        adaptations=(*response.adaptations, *adaptations),
    )


class _CodexEngine:
    """Shared capabilities and per-call connection access for the engine pair."""

    supports_function_calling = True
    supports_reasoning = True
    supports_response_schema = True
    supported_tool_result_media = frozenset({"image", "document"})
    supported_params = frozenset(
        {
            "temperature",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "response_format",
            "reasoning",
            "reasoning_effort",
            "reasoning_summary",
            "service_tier",
            "logprobs",
            "top_logprobs",
        }
    )

    def __init__(self, get_connection: Callable[[], CodexConnection]) -> None:
        self.get_connection = get_connection


class CodexEngine(_CodexEngine):
    """Execute one canonical request through the synchronous shared client."""

    def complete(self, request: Request) -> Response:
        connection = self.get_connection()
        with closing(OpenAILM(api_key="serialization-only")) as codec, _codex_errors():
            wire, payload, adaptations = _encode(codec, request)
            result = connection.create_client().create(
                **payload, **connection.request_controls()
            )
            return _decode(codec, wire, result, adaptations)


class AsyncCodexEngine(_CodexEngine):
    """Execute one canonical request through the asynchronous shared client."""

    async def complete(self, request: Request) -> Response:
        connection = self.get_connection()
        with closing(OpenAILM(api_key="serialization-only")) as codec, _codex_errors():
            wire, payload, adaptations = _encode(codec, request)
            result = await connection.create_client().acreate(
                **payload, **connection.request_controls()
            )
            return _decode(codec, wire, result, adaptations)
