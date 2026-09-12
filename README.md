# dspy-codex-auth

DSPy integration for using ChatGPT/Codex subscription credentials as a DSPy
language model.

- It uses [`openai-codex-auth`](https://github.com/hrbatra/openai-codex-auth)
  for authentication, HTTP/WebSocket requests, and streamed-response handling.
- It installs a DSPy `LM` wrapper for `codex/...` model strings.
- It converts DSPy messages and options into native Codex requests, and converts
  shared client responses into DSPy's output, history, and usage formats.

## Install

```bash
uv add dspy-codex-auth
```

The two packages have separate roles: `openai-codex-auth` provides the reusable
Codex client; `dspy-codex-auth` provides its DSPy adapter. Installing the adapter
brings in both packages. For ordinary Python without DSPy, install
`openai-codex-auth` and use `CodexClient` directly. No Pi plugin is needed.

## Login

Sign in once with the Codex CLI and choose "Sign in with ChatGPT":

```bash
codex login
```

The adapter reads the account identity from `~/.codex/auth.json` on each call,
including cache lookup, to keep cached results separate across accounts. The
shared client refreshes the token when an uncached request is sent. Constructing
an LM does not read credentials; credential errors surface on its first call.
Cached calls using file-backed auth still require a readable credential file.
`CodexAuth` and `getauthtoken` are re-exports from `openai-codex-auth`; pass
`auth_storage=` (a `CodexAuth` or a path) to `install()` or `LM(...)` to use a
different auth file.

This integration requires a file-backed ChatGPT login. Codex can also store
credentials in the OS credential store, which this package does not read; see
the [Codex authentication documentation](https://learn.chatgpt.com/docs/auth#login-caching).

## Basic Usage

```python
import dspy
import dspy_codex_auth

dspy_codex_auth.install()

lm = dspy.LM("codex/gpt-5.6-luna", cache=False)
dspy.configure(lm=lm, adapter=dspy.JSONAdapter())
```

Use `codex/<model>` for the ChatGPT Codex subscription route. This is not the
OpenAI API key route and does not require `OPENAI_API_KEY`.

`cache=False` is recommended for Codex while iterating because stale DSPy cache
entries can preserve old empty-output responses across package upgrades.

## Codex Transport Selection

Codex LMs accept `codex_transport` at construction time. Its accepted values
are exactly `"auto"`, `"http"`, and `"websocket"`:

```python
lm = dspy.LM(
    "codex/gpt-5.6-luna",
    codex_transport="auto",
    codex_websocket_connect_timeout=10.0,
    codex_websocket_idle_timeout=300.0,
    cache=False,
)
```

- `"auto"` is the default. It tries HTTP first and falls back to WebSocket only
  when HTTP returns the exact structured model-not-found error for the requested
  model. Other HTTP errors remain visible; message heuristics do not trigger a
  fallback.
- `"http"` uses HTTP only and never falls back to WebSocket.
- `"websocket"` uses WebSocket only.

The same three transport values and both timeout options can be overridden for
one call without changing the LM defaults:

```python
outputs = lm(
    "Return exactly the single lowercase token: pink",
    codex_transport="websocket",
    codex_websocket_connect_timeout=10.0,
    codex_websocket_idle_timeout=300.0,
)
```

The WebSocket connect timeout defaults to 10 seconds, and its per-event idle
timeout defaults to 300 seconds. Custom `api_base` values must use HTTP or
HTTPS; the WebSocket transport converts them to WS or WSS and appends the
`/responses` path. TLS uses Requests' CA bundle.

The shared client owns retries. `num_retries` counts additional attempts across
transports; retry delays begin at 0.5 seconds and double up to 8 seconds.
Auto selection may additionally make an HTTP routing probe. An explicit numeric
`timeout` also controls WebSocket idle time under the existing call/constructor
precedence; HTTP supports phase-specific `httpx.Timeout` values.

Authentication remains in the WebSocket handshake: the bearer credential is
never placed in the `response.create` data frame. HTTP requests use originator
`dspy_codex_auth`; WebSocket handshakes use originator `codex_cli_rs`. Both
preserve the honest `DSPy/<version>` User-Agent supplied by DSPy.

## Swapping Models

Call `dspy_codex_auth.install()` once near process startup. Codex model strings
use subscription auth; non-Codex model strings continue through DSPy's normal LM
behavior.

```python
import dspy
import dspy_codex_auth

dspy_codex_auth.install()


def configure_model(model: str, **kwargs):
    lm = dspy.LM(model, **kwargs)
    dspy.configure(lm=lm, adapter=dspy.JSONAdapter())
    return lm


codex_lm = configure_model("codex/gpt-5.5", cache=False)
api_lm = configure_model("openai/gpt-5.5", api_key="...", cache=False)
```

## Reasoning Summary

Pass `reasoning_effort` as usual. This package also supports
`reasoning_summary`, which maps to the Responses API `reasoning.summary` field.

```python
lm = dspy.LM(
    "codex/gpt-5.5",
    cache=False,
    reasoning_effort="medium",
    reasoning_summary="detailed",
)
```

DSPy predictions expose declared output fields. The lower-level LM history can
also include a returned reasoning summary:

```python
summary = lm.history[-1]["outputs"][0].get("reasoning_content")
```

## Fast Mode / Service Tier

Codex CLI config uses `service_tier = "fast"` for Fast mode. Internally, the
CLI normalizes that config value to the backend request tier `priority`; this
package does the same, so DSPy callers can use the documented Codex config
spelling directly:

```python
lm = dspy.LM(
    "codex/gpt-5.4",
    cache=False,
    service_tier="fast",
    reasoning_effort="low",
)
```

`service_tier="priority"` and `service_tier="flex"` are also passed through.
Omit `service_tier` to use the account and model default.

For raw Codex CLI calls, use `-c` config overrides:

```bash
codex exec \
  -m gpt-5.4 \
  -c 'service_tier="fast"' \
  -c 'model_reasoning_effort="low"' \
  --json \
  'Return exactly the single lowercase word: pink'
```

The DSPy-facing kwarg is `reasoning_effort`. For convenience, this package also
accepts Codex config-style aliases `model_reasoning_effort` and
`model_reasoning_summary`.

Relevant Codex docs:

- [Configuration reference](https://developers.openai.com/codex/config-reference)
- [Config feature flags](https://developers.openai.com/codex/config-basic#supported-features)

## OpenAI-Style Model String With Codex Auth

If you prefer to keep an `openai/...` model string and select Codex auth
explicitly:

```python
lm = dspy_codex_auth.LM(
    "openai/gpt-5.5",
    auth_provider="codex",
    cache=False,
    reasoning_effort="medium",
    reasoning_summary="detailed",
)
```

This is useful when the rest of your app treats model names as provider-neutral
strings and you want auth selection to be a separate setting.

## What It Fixes

The shared Codex client reconstructs missing output from streamed events,
preserving provider metadata, message text, function calls, and reasoning
summaries. This adapter converts that response into the objects DSPy expects.
Codex requests no longer pass through LiteLLM's Responses transport.

A provider refusal raises `openai_codex_auth.CodexError` in this adapter so it
cannot become an empty DSPy prediction. Direct `CodexClient` callers receive
the native refusal item in `CodexResponse.output`.

The adapter and shared core together handle:

- DSPy few-shot and conversation-history assistant messages by encoding them as
  Responses `output_text` blocks, which supports optimizers such as
  `LabeledFewShot`.
- GEPA reflection calls that invoke the LM with a plain prompt string.
- `response.output_item.done`
- `response.output_text.done`
- `response.output_text.delta`
- `response.reasoning_summary_text.done`
- `response.reasoning_summary_text.delta`
- streamed function-call output items

It also strips output-token cap fields that the Codex backend currently rejects:

- `max_tokens`
- `max_output_tokens`
- `max_completion_tokens`

It normalizes `service_tier="fast"` to the Codex backend request value
`service_tier="priority"`, matching Codex CLI behavior.

## French Example

```python
import dspy
import dspy_codex_auth

dspy_codex_auth.install()

lm = dspy.LM("codex/gpt-5.5", cache=False)
dspy.configure(lm=lm, adapter=dspy.JSONAdapter())


class TranslateFrenchToEnglish(dspy.Signature):
    """Translate the French input into short, natural English."""

    french: str = dspy.InputField(desc="French sentence")
    english: str = dspy.OutputField(desc="Natural English translation")


translator = dspy.Predict(TranslateFrenchToEnglish)
print(translator(french="merci beaucoup").english)
```

## Math Example With Reasoning Summary

```python
import dspy
import dspy_codex_auth

dspy_codex_auth.install()

lm = dspy.LM(
    "codex/gpt-5.5",
    cache=False,
    reasoning_effort="medium",
    reasoning_summary="detailed",
)
dspy.configure(lm=lm, adapter=dspy.JSONAdapter())


class SolveMath(dspy.Signature):
    """Solve the math problem. Return a concise numeric answer and a brief explanation."""

    problem: str = dspy.InputField(desc="Math problem")
    answer: str = dspy.OutputField(desc="Concise final answer")
    explanation: str = dspy.OutputField(desc="Brief explanation")


solver = dspy.Predict(SolveMath)
pred = solver(
    problem=(
        "Compute the integral of the standard normal probability density "
        "function from 0 to 1.5."
    )
)

print(pred.answer)
print(pred.explanation)
print(lm.history[-1]["outputs"][0].get("reasoning_content"))
```

## Attribution

`dspy-codex-auth` includes and adapts MIT-licensed DSPy integration code from
`dspy-lm-auth`:

https://github.com/MaximeRivest/dspy-lm-auth

The streamed-output reconstruction addresses a DSPy/Codex Responses streaming
compatibility issue that was also discussed in `dspy-lm-auth` PR #2:

https://github.com/MaximeRivest/dspy-lm-auth/pull/2

`dspy-lm-auth` is MIT-licensed. The original copyright notice is preserved in
`THIRD_PARTY_NOTICES.md`.

## Development

```bash
uv sync --dev
uv run pytest
uv run ruff check .
uv build --no-sources
```

The default test command skips the real-network live transport test. Run it
explicitly only with valid Codex subscription credentials:

```bash
DSPY_CODEX_AUTH_RUN_LIVE_TESTS=1 \
  uv run pytest -m live tests/test_live_websocket.py
```

## Release

Full PyPI update instructions are in [RELEASING.md](RELEASING.md).

Short local release flow:

```bash
uv version --bump patch
rm -rf dist
uv build --no-sources
uv run --with twine python -m twine upload dist/*
```

PyPI releases are immutable, so every update needs a new version number.
