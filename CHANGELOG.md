# Changelog

## 0.1.7 - 2026-07-11

### Fixed

- Keep DSPy/LiteLLM client-only timeout and retry controls out of Codex
  WebSocket `response.create` frames.
- Honor a caller-supplied `timeout` as the per-call WebSocket receive idle
  timeout while preserving the existing HTTP transport behavior.
- Cover the Luna WebSocket route with an explicit timeout in mocked protocol
  tests and the opt-in live transport test.
