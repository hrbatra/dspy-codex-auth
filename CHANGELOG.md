# Changelog

## 0.1.8 - Unreleased

### Changed

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
