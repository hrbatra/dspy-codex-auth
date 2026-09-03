# Changelog

## 0.1.8 - Unreleased

### Changed

- The auth layer (OAuth login, token refresh, Pi-compatible credential
  storage) moved to the new `openai-codex-auth` package, now a runtime
  dependency. `dspy_codex_auth.auth` is gone. `login`, `logout`,
  `getauthtoken`, and `AuthStorage` stay re-exported here; everything else
  is imported from `openai_codex_auth`.
- The login flow now identifies itself as `openai_codex_auth`; request headers
  still carry the `dspy_codex_auth` originator.

## 0.1.7 - 2026-07-11

### Fixed

- Keep DSPy/LiteLLM client-only timeout and retry controls out of Codex
  WebSocket `response.create` frames.
- Honor a caller-supplied `timeout` as the per-call WebSocket receive idle
  timeout while preserving the existing HTTP transport behavior.
- Cover the Luna WebSocket route with an explicit timeout in mocked protocol
  tests and the opt-in live transport test.
