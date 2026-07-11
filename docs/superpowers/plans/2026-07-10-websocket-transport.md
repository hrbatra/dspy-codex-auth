# Codex WebSocket Transport Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an HTTP-first Codex WebSocket fallback that unlocks Luna without changing successful Terra/Sol HTTP calls, then release version 0.1.6 only after every local and live gate passes.

**Architecture:** A focused `responses_websocket.py` module owns the WebSocket handshake, frame loop, TLS, and typed protocol errors. `lm.py` remains responsible for building Codex Responses requests, exact LiteLLM error classification, HTTP/WebSocket dispatch, retries, DSPy caching, and response reconstruction. The release remains a separate gated task after implementation, tests, documentation, and credential checks.

**Tech Stack:** Python 3.12, DSPy, LiteLLM, Requests CA bundle, `websockets` sync/async clients, pytest, Ruff, uv, Twine.

## Global Constraints

- `codex_transport: Literal["auto", "http", "websocket"] = "auto"` on the constructor, with a per-call override of the same key.
- `auto` attempts HTTP first and falls back only on the exact structured model-not-found error for the requested model.
- `http` never falls back; `websocket` bypasses HTTP.
- Terra and Sol remain on HTTP when HTTP succeeds; Luna completes through WebSocket.
- Preserve `py.typed`, existing public imports, non-Codex routing, and the existing Responses output reconstruction behavior.
- Add dependencies only with `uv add`; never edit dependency declarations in `pyproject.toml` by hand.
- Do not add legacy adapters, dual schemas, heuristic model allowlists, broad message matching, or silent fallbacks.
- Do not expose bearer tokens in frames, test output, logs, exceptions, or the report.
- Stop before versioning, pushing, building, or publishing if any required verification or credential gate fails.
- Keep `.codex-runs/websocket-transport-report.md` local and do not stage the existing `.codex-runs` artifacts.

## Dependency graph

```yaml
tasks:
  - id: T1
    title: Implement the typed WebSocket protocol client
    depends_on: []

  - id: T2
    title: Integrate constructor and per-call transport dispatch
    depends_on: [T1]

  - id: T3
    title: Add live coverage and user documentation
    depends_on: [T1, T2]

  - id: T4
    title: Run all verification and credential gates
    depends_on: [T1, T2, T3]

  - id: T5
    title: Version, changelog, push, build, publish, and verify
    depends_on: [T4]
```

---

### Task 1 (T1): Typed WebSocket protocol client

**Files:**

- Create: `src/dspy_codex_auth/responses_websocket.py`
- Create: `tests/test_responses_websocket.py`
- Modify through uv only: `pyproject.toml`, `uv.lock`

**Interfaces:**

- Consumes: a fully built Responses request, resolved Codex `api_base`, bearer token, and handshake headers.
- Produces: `WebSocketResult(events: list[dict[str, Any]], response: dict[str, Any])`, `websocket_response(...)`, `awebsocket_response(...)`, timeout constants, URL conversion, and typed protocol/response errors.

- [ ] **Step 1: Write the protocol tests before the module exists**

  Add tests that first assert the expected module is discoverable, then exercise
  fake sync/async connection context managers. The fake connections must capture
  `connect()` kwargs and the sent frame, return JSON text events ending in
  `response.completed`, and provide separate cases for nested server `error`,
  malformed JSON, a binary frame, early close, and timeout. Assertions must prove
  that the URL is `wss://chatgpt.com/backend-api/codex/responses`, protected
  handshake headers have the approved values, only `openai/` is stripped from
  the model, `type` is `response.create`, and the bearer token is absent from the
  serialized frame.

- [ ] **Step 2: Run the focused test and record RED**

  Run: `uv run pytest tests/test_responses_websocket.py -q`

  Expected: assertion failure because `dspy_codex_auth.responses_websocket` is
  not yet discoverable.

- [ ] **Step 3: Add the dependency through uv**

  Run: `uv add websockets`

  Expected: uv selects the current compatible version and updates both
  `pyproject.toml` and `uv.lock`.

- [ ] **Step 4: Implement the minimal protocol module**

  Define these exact public shapes:

  ```python
  DEFAULT_CODEX_WEBSOCKET_BETA = "responses_websockets=2026-02-06"
  DEFAULT_CODEX_WEBSOCKET_ORIGINATOR = "codex_cli_rs"
  DEFAULT_CODEX_WEBSOCKET_CONNECT_TIMEOUT = 10.0
  DEFAULT_CODEX_WEBSOCKET_IDLE_TIMEOUT = 300.0

  @dataclass(frozen=True, slots=True)
  class WebSocketResult:
      events: list[dict[str, Any]]
      response: dict[str, Any]

  def websocket_response(
      request: dict[str, Any],
      *,
      api_base: str,
      api_key: str,
      headers: Mapping[str, Any] | None,
      connect_timeout: float = DEFAULT_CODEX_WEBSOCKET_CONNECT_TIMEOUT,
      idle_timeout: float = DEFAULT_CODEX_WEBSOCKET_IDLE_TIMEOUT,
  ) -> WebSocketResult: ...

  async def awebsocket_response(
      request: dict[str, Any],
      *,
      api_base: str,
      api_key: str,
      headers: Mapping[str, Any] | None,
      connect_timeout: float = DEFAULT_CODEX_WEBSOCKET_CONNECT_TIMEOUT,
      idle_timeout: float = DEFAULT_CODEX_WEBSOCKET_IDLE_TIMEOUT,
  ) -> WebSocketResult: ...
  ```

  Use `websockets.sync.client.connect` and
  `websockets.asyncio.client.connect`, Requests' CA path in
  `ssl.create_default_context(cafile=requests.certs.where())`, case-insensitive
  replacement of protected headers, one JSON text request frame, and a receive
  loop ending only at a validated `response.completed`. Async idle timeout uses
  `asyncio.timeout`; sync uses `recv(timeout=...)`.

- [ ] **Step 5: Run focused tests and record GREEN**

  Run: `uv run pytest tests/test_responses_websocket.py -q`

  Expected: all protocol tests pass with no warnings from project code.

- [ ] **Step 6: Format and inspect the task diff**

  Run: `uv run ruff format src/dspy_codex_auth/responses_websocket.py tests/test_responses_websocket.py && uv run ruff check src/dspy_codex_auth/responses_websocket.py tests/test_responses_websocket.py && git diff --check`

  Expected: exit 0.

---

### Task 2 (T2): Constructor and per-call transport dispatch

**Files:**

- Modify: `src/dspy_codex_auth/lm.py`
- Modify: `src/dspy_codex_auth/__init__.py`
- Modify: `tests/test_lm.py`

**Interfaces:**

- Consumes: T1's sync/async WebSocket calls and result type.
- Produces: public `CodexTransport`, constructor storage, per-call override,
  exact error predicate, sync/async dispatcher, and cache-distinguishing
  transport argument.

- [ ] **Step 1: Write failing routing tests**

  Add tests for constructor default/explicit values, invalid constructor and
  per-call values, explicit HTTP, explicit WebSocket, HTTP-first auto success,
  exact auto fallback, near-miss 404/message/type/param/code/model errors that
  must not fallback, async parity, and a per-call override that supersedes the
  constructor. Use real `litellm.BadRequestError` instances whose status/model
  and JSON payload match the Round 1 probe. Capture the cached completion call
  and assert `codex_transport` is an explicit cache input.

- [ ] **Step 2: Run routing tests and record RED**

  Run: `uv run pytest tests/test_lm.py -q`

  Expected: failures because `LM` does not accept/store/dispatch the new
  transport contract.

- [ ] **Step 3: Implement the exact typed contract**

  Add:

  ```python
  type CodexTransport = Literal["auto", "http", "websocket"]

  def __init__(
      self,
      model: str,
      *args: Any,
      auth_storage: AuthStorage | str | os.PathLike[str] | None = None,
      auth_provider: str | None = None,
      codex_transport: CodexTransport = "auto",
      codex_websocket_connect_timeout: float = DEFAULT_CODEX_WEBSOCKET_CONNECT_TIMEOUT,
      codex_websocket_idle_timeout: float = DEFAULT_CODEX_WEBSOCKET_IDLE_TIMEOUT,
      **kwargs: Any,
  ) -> None: ...
  ```

  Pop `codex_transport` and timeout overrides from each call before merging wire
  kwargs. Call a sync/async dispatcher through `_get_cached_completion_fn` with
  the selected transport and timeouts as named arguments. Keep
  `_codex_responses_completion` and `_acodex_responses_completion` as the HTTP
  implementations. The dispatcher calls HTTP first for `auto`, catches only the
  exact structured predicate, and then calls WebSocket. It must never catch a
  WebSocket error and retry HTTP.

  Before calling T1, remove only known client configuration keys (`api_key`,
  `api_base`, `headers`, `model_type`, `use_developer_role`, and `rollout_id`),
  use `_build_codex_request`, omit `None` wire values, and reconstruct the final
  response with `_StreamOutputBuilder` plus `_reconstruct_stream_output`.

- [ ] **Step 4: Export public typing/constants**

  Export `CodexTransport` and the documented transport timeout constants from
  `dspy_codex_auth.__init__` and keep `__all__` synchronized.

- [ ] **Step 5: Run routing and protocol tests GREEN**

  Run: `uv run pytest tests/test_lm.py tests/test_responses_websocket.py -q`

  Expected: all selected tests pass.

- [ ] **Step 6: Verify the regression test really detects the fallback change**

  Temporarily invert the exact predicate result for its positive fixture, run
  the exact-fallback test and confirm it fails, restore the implementation, and
  rerun the test to pass. Do not commit the temporary inversion.

- [ ] **Step 7: Format and inspect the task diff**

  Run: `uv run ruff format src tests && uv run ruff check src tests && git diff --check`

  Expected: exit 0.

---

### Task 3 (T3): Live coverage and documentation

**Files:**

- Create: `tests/test_live_websocket.py`
- Create: `pytest.ini`
- Modify: `README.md`
- Modify: `tests/smoke_test.py` only if the exported public types/constants need
  an import smoke assertion.

**Interfaces:**

- Consumes: completed T1/T2 public API.
- Produces: opt-in `live` proof and user-facing transport/timeout/originator
  documentation.

- [ ] **Step 1: Add one opt-in live-marked test**

  Register `live` in `pytest.ini`. Add one test function marked
  `@pytest.mark.live` and skipped unless
  `DSPY_CODEX_AUTH_RUN_LIVE_TESTS == "1"`. Install the shim, wrap the real
  WebSocket completion call to record model ids, and call Luna, Terra, and Sol
  through `codex_transport="auto"`. Assert non-empty completions, Luna appears
  in the WebSocket record, and Terra/Sol do not.

- [ ] **Step 2: Run default suite behavior**

  Run: `uv run pytest tests/test_live_websocket.py -q`

  Expected: exactly one skipped test when the environment flag is absent.

- [ ] **Step 3: Document selection and security-relevant handshake behavior**

  Add a README transport section with the three modes, constructor and per-call
  examples, 10/300-second timeout kwargs, HTTP-first behavior, custom base URL
  conversion, Requests CA bundle, and the explicit WebSocket-only
  `codex_cli_rs` originator requirement. State that HTTP keeps
  `dspy_codex_auth` and bearer auth is handshake-only.

- [ ] **Step 4: Run docs/source checks**

  Run: `uv run ruff format . && uv run ruff check . && uv run python tests/smoke_test.py && git diff --check`

  Expected: exit 0.

---

### Task 4 (T4): Verification and credential gates

**Files:**

- Create/update locally only: `.codex-runs/websocket-transport-report.md`

**Interfaces:**

- Consumes: T1-T3 implementation and the machine's configured Codex, GitHub,
  and PyPI credentials.
- Produces: fresh test evidence and a binary release/no-release decision.

- [ ] **Step 1: Run the exact live-marked test**

  Run: `DSPY_CODEX_AUTH_RUN_LIVE_TESTS=1 uv run pytest tests/test_live_websocket.py -vv -s`

  Expected: Luna completes and records WebSocket; Terra/Sol complete without a
  WebSocket record.

- [ ] **Step 2: Run a standalone exact Luna shim probe**

  Run a Python script that calls `dspy_codex_auth.install()`, constructs exactly
  `dspy.LM("codex/gpt-5.6-luna", reasoning_effort="high")`, requests a short
  deterministic response, and prints only model, selected transport
  instrumentation, and completion text—never auth/header data.

- [ ] **Step 3: Run standalone Terra/Sol HTTP guards**

  Wrap the WebSocket completion entry point with an assertion-raising guard,
  then call default-auto Terra and Sol LMs. Both must return non-empty text
  without invoking the guard.

- [ ] **Step 4: Run the full quality gate from a fresh command**

  Run: `uv run pytest && uv run ruff check . && uv run ruff format --check . && uv run python tests/smoke_test.py && git diff --check`

  Expected: pytest has zero failures, Ruff exits 0, smoke exits 0, and diff check
  exits 0.

- [ ] **Step 5: Check credentials without disclosing values**

  Check readable Codex auth, successful read-only GitHub remote access, a
  non-mutating Git push dry run, and either a readable configured PyPI section
  in `~/.pypirc` or the required Twine environment variable names. Print only
  booleans/status. If any check fails, write the report and stop before T5.

- [ ] **Step 6: Write the report evidence**

  Include protocol notes, design choices, baseline and final test tails, live
  outputs, credential statuses without values, commit SHAs, and release status
  in `.codex-runs/websocket-transport-report.md`. Do not stage `.codex-runs`.

---

### Task 5 (T5): Version, changelog, push, build, publish, and verify

**Files:**

- Create: `CHANGELOG.md`
- Modify through uv only: `pyproject.toml`, `uv.lock`
- Update locally only: `.codex-runs/websocket-transport-report.md`

**Interfaces:**

- Consumes: a fully passing T4 and present credentials.
- Produces: GitHub `main` containing release 0.1.6 and PyPI version 0.1.6.

- [ ] **Step 1: Review and commit the feature as cohesive local commits**

  Inspect `git status -sb`, `git diff --stat`, `git diff --check`, and targeted
  diffs. Stage only source, tests, dependency lock, README, pytest config, and
  approved design/plan docs. Never stage `.codex-runs`. Use descriptive messages
  matching the repository's imperative style.

- [ ] **Step 2: Bump with uv and create the changelog**

  Run: `uv version --bump patch`

  Expected: version changes from 0.1.5 to 0.1.6 in `pyproject.toml` and
  `uv.lock`. Create `CHANGELOG.md` with a `0.1.6 - 2026-07-10` entry describing
  the WebSocket transport, exact fallback, routing controls, mock/live tests,
  and timeout/originator documentation.

- [ ] **Step 3: Re-run the full gate against the versioned tree**

  Run: `uv run pytest && uv run ruff check . && uv run ruff format --check . && uv run python tests/smoke_test.py && git diff --check`

  Expected: all commands exit 0. Any failure stops before commit/push.

- [ ] **Step 4: Commit and push main**

  Commit the version/changelog as `Release 0.1.6`, confirm the working tree has
  only ignored/local `.codex-runs` artifacts, then run
  `git push origin main`. Confirm `git status -sb` reports `main...origin/main`
  with no divergence.

- [ ] **Step 5: Build only after the successful push**

  Run: `rm -rf dist && uv build --no-sources && uv run --with twine python -m twine check dist/*`

  Expected: sdist and wheel for 0.1.6 are created and both pass Twine check.

- [ ] **Step 6: Inspect wheel contents and smoke-install it**

  Confirm the wheel contains `py.typed` and the new module, then install the
  wheel into a fresh uv temp project and import `CodexTransport`, `LM`, and the
  WebSocket constants. Stop before upload if this fails.

- [ ] **Step 7: Publish non-interactively**

  Run: `uv run --with twine python -m twine upload --non-interactive dist/*`

  Expected: both 0.1.6 artifacts upload successfully. If any artifact uploads
  and a later upload fails, do not reuse 0.1.6.

- [ ] **Step 8: Verify PyPI and a fresh index install**

  Query `https://pypi.org/pypi/dspy-codex-auth/0.1.6/json`, create a new temp uv
  project, run `uv add --refresh dspy-codex-auth==0.1.6`, and print the installed
  distribution version, absence of `dspy_lm_auth`, `LM.__module__`, and presence
  of the WebSocket module/public type. Remove the temp project afterward.

- [ ] **Step 9: Finalize the local report**

  Record published version `0.1.6`, artifact names, Git SHAs, PyPI verification,
  fresh-install output, and the final clean/divergence status. Leave the report
  untracked under `.codex-runs` as requested.
