# Changelog

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
