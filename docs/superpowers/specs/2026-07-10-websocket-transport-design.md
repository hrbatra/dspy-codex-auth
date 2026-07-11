# Codex WebSocket Transport Design

## Status

Approved by the maintainer in the Round 2 task after the read-only Round 1
design and live protocol probes recorded in
`.codex-runs/websocket-transport-run.jsonl`.

## Goal

Allow `dspy.LM("codex/gpt-5.6-luna", reasoning_effort="high")` to complete
through the ChatGPT Codex subscription shim while preserving HTTP behavior for
models such as `gpt-5.6-terra` and `gpt-5.6-sol`.

## Public contract

`LM` accepts this Codex-only constructor option:

```python
codex_transport: Literal["auto", "http", "websocket"] = "auto"
```

The same key is accepted as a per-call override.

- `auto` sends HTTP first and falls back to WebSocket only when LiteLLM raises
  the exact structured HTTP error observed for the requested model: status
  `404`, provider `openai`, matching wire model, and JSON error fields
  `message="Model not found <model>"`, `type="invalid_request_error"`,
  `param="model"`, and `code=null`.
- `http` never falls back.
- `websocket` bypasses HTTP.
- Invalid values fail with `ValueError`; they are never passed through to DSPy,
  LiteLLM, or the wire request.
- Transport selection participates in DSPy's completion cache key.

The WebSocket connect and idle timeouts are explicit constructor/per-call
options, defaulting to 10 and 300 seconds respectively. The connect default is
the Python `websockets` client default; the idle default matches Codex's
documented provider stream timeout.

## Protocol

The client opens one connection per completion. It converts the resolved Codex
HTTP base URL to `ws://` or `wss://` and appends `/responses`. The default URL is
`wss://chatgpt.com/backend-api/codex/responses`.

The HTTP upgrade carries bearer authorization, ChatGPT account id, an honest
`DSPy/<version>` user agent, request/session/thread ids, the beta header
`responses_websockets=2026-02-06`, and WebSocket-only
`originator=codex_cli_rs`. Live A/B probes showed that Luna fails when the
WebSocket originator remains `dspy_codex_auth`; the existing HTTP originator is
unchanged. Authorization never appears in a data frame or report.

After the upgrade, the client sends one JSON text frame with
`type="response.create"`. The normal Codex Responses request builder supplies
the request body, and only the leading `openai/` provider prefix is removed
from the wire model. There is no application-level hello frame or subprotocol.

Incoming JSON text frames are retained as Responses events. A valid
`response.completed` event supplies the terminal response object. An `error`
event, invalid JSON, binary frame, invalid terminal response, idle timeout, or
connection closure before completion raises a typed visible error. The
existing stream-output builder reconstructs message text, reasoning summaries,
and function calls from events when the terminal response has empty output.

TLS uses an `ssl.SSLContext` rooted at Requests' CA bundle, because the Round 1
probe reproduced a missing-issuer failure with this machine's default Python
trust store while Requests' bundle succeeded.

Upstream protocol references:

- <https://github.com/openai/codex/blob/08ba14b03d0b3ce3cfdf8c88c0469b9b1924953d/codex-rs/codex-api/src/common.rs>
- <https://github.com/openai/codex/blob/08ba14b03d0b3ce3cfdf8c88c0469b9b1924953d/codex-rs/codex-api/src/endpoint/responses_websocket.rs>
- <https://github.com/openai/codex/blob/08ba14b03d0b3ce3cfdf8c88c0469b9b1924953d/codex-rs/core/src/client.rs>

## Tests and release gates

Unit tests mock sync and async frames, handshake inputs, serialization,
terminal reconstruction, and protocol failures. Routing tests cover all three
transport modes, the exact fallback predicate, near-miss errors, constructor
selection, per-call overrides, and distinct cache inputs. One test is marked
`live` and is opt-in through `DSPY_CODEX_AUTH_RUN_LIVE_TESTS=1`; it proves Luna
uses WebSocket while Terra and Sol complete without entering WebSocket.

No release step begins unless the live Luna probe, Terra/Sol HTTP probes, full
pytest suite, Ruff checks, and source-tree smoke checks pass. Credential
presence is checked without printing secrets. Release `0.1.6` is created with
`uv version --bump patch`, a changelog entry, build/Twine checks, GitHub push,
PyPI upload, PyPI index verification, and a fresh-install smoke test. Any failed
gate stops the release.
