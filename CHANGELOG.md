# Changelog

## 0.3.0 - 2026-11-15

### Breaking changes

- Require DSPy 3.5 or newer. The adapter now uses sync and async engines with
  `complete(Request) -> Response`, replacing the deprecated `forward` and
  `aforward` integration hooks.
- Direct `messages=` calls and `forward`/`aforward` are removed. Use
  `dspy.lm15.Request`, `Message`, and `Config` for direct structured calls.
  `lm("text")`, `await lm.acall("text")`, and `dspy.Predict` remain supported.
- Typed requests return a `dspy.lm15.Response` and take generation settings only
  from `Request.config`. Native usage and reasoning are available on that
  response. Raw Codex metadata and the selected transport live under
  `response.provider_data["codex_response"]` and `["codex_transport"]`.
- DSPy now owns retries and usage accounting. `num_retries` still counts
  additional attempts, with DSPy's 1, 2, 4, ... second delays capped at 60
  seconds. The shared client runs one attempt per engine invocation.
- Provider failures and refusals surface through DSPy's public LM error family.
- Stop sequences raise `LMUnsupportedFeatureError` before provider I/O. The
  completion-only engine cannot stop generation at a matching sequence.
- Dropped or clamped Responses codec options and removed output-token caps
  are reported in `Response.adaptations`.

### Fixed

- Codex calls use the shared authenticated client on DSPy 3.5, instead of
  bypassing the adapter and reaching DSPy's built-in transport.
- Retain sync/async behavior, transport overrides, account-separated caches,
  and current file-backed credentials under the engine interface.
- Keep credentials and custom headers out of `LM.dump_state()` JSON state;
  preserve route and transport settings when copying or restoring an LM.
  DSPy's full-program pickle retains runtime objects and is not this sanitized
  JSON format.
- Preserve deepcopy and full-program save/load. Copies and saved LM state bind
  the resolved Codex model rather than a mutable custom alias registration.

Restart notebook kernels after upgrading to replace already imported classes.

## 0.2.0 - 2026-09-11

### Changed

- Codex HTTP/WebSocket transport, retries, and streaming reconstruction now
  come from `openai-codex-auth>=0.2.0`. This package contains the DSPy adapter;
  the old `dspy_codex_auth.responses_websocket` module has been removed.
- Credentials are read and refreshed for each uncached request, instead of
  snapshotting the token when an LM is constructed. Credential errors now
  surface on the first call; account identity is read during cache lookup too.
  Cache entries are isolated by account, explicit credential, and API endpoint.
- Shared retries use one bounded attempt budget, with delays starting at
  0.5 seconds and doubling up to 8 seconds. `num_retries` remains the number
  of additional attempts.
- Authentication now comes from the new `openai-codex-auth` dependency, which
  reads the Codex CLI's `codex login` credential (`~/.codex/auth.json`) and
  refreshes it when needed. The package's own OAuth login flow, the Pi-format
  `~/.pi/agent/auth.json` store, `login()`, `logout()`, and `AuthStorage` are
  gone; `CodexAuth` and `getauthtoken` are re-exported.
- `install(auth_storage=...)` and `LM(auth_storage=...)` take a `CodexAuth` or
  a path to a Codex CLI auth file.

## 0.1.7 - 2026-07-11

### Fixed

- Keep DSPy/LiteLLM client-only timeout and retry controls out of Codex
  WebSocket `response.create` frames.
- Honor a caller-supplied `timeout` as the per-call WebSocket receive idle
  timeout while preserving the existing HTTP transport behavior.
- Cover the Luna WebSocket route with an explicit timeout in mocked protocol
  tests and the opt-in live transport test.
