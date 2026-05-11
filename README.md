# dspy-codex-auth

DSPy integration for using ChatGPT/Codex subscription credentials as a DSPy
language model.

This package is intentionally narrow:

- It includes ChatGPT/Codex OAuth login, token refresh, and Pi-compatible
  credential storage.
- It installs a DSPy `LM` wrapper for `codex/...` model strings.
- It fixes Codex Responses streaming shapes that DSPy 3.2 cannot parse from
  the current Codex backend response stream.

## Install

```bash
uv add dspy-codex-auth
```

## Login

If you already have Codex credentials in `~/.pi/agent/auth.json`, no extra
login is needed. The package reads and refreshes that Pi-compatible credential
file directly.

Otherwise:

```bash
uv run python -c "import dspy_codex_auth; dspy_codex_auth.login()"
```

## Basic Usage

```python
import dspy
import dspy_codex_auth

dspy_codex_auth.install()

lm = dspy.LM("codex/gpt-5.5", cache=False)
dspy.configure(lm=lm, adapter=dspy.JSONAdapter())
```

Use `codex/<model>` for the ChatGPT Codex subscription route. This is not the
OpenAI API key route and does not require `OPENAI_API_KEY`.

`cache=False` is recommended for Codex while iterating because stale DSPy cache
entries can preserve old empty-output responses across package upgrades.

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

The ChatGPT Codex backend streams useful output events, but the completed
LiteLLM Responses object can arrive with `response.output == []`. DSPy expects
Responses output items to contain final message text, function calls, and
reasoning summaries. This package reconstructs those output items from stream
events before DSPy parses the response.

It currently handles:

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

`dspy-codex-auth` includes and adapts MIT-licensed auth and DSPy integration
code from `dspy-lm-auth`:

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
