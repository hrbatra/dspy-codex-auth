# Contract: DSPy 3.5 Codex engine migration
status: approved

## Interface
`dspy_codex_auth.LM`, `install`, `uninstall`, `resolve_lm_route`, and model alias
registration remain public. DSPy 3.5 is the dependency floor. A Codex LM accepts
`lm(str)` / `await lm.acall(str)` returning DSPy's list convenience output, and
`lm(dspy.lm15.Request)` / `await lm.acall(Request)` returning `dspy.lm15.Response`.
The request's model equals `lm.model` (the resolved `openai/<model>` string).
Direct `messages=`, `forward`, and `aforward` are removed; callers use typed
requests. Native Codex requests still go through `openai_codex_auth.CodexClient`.
The constructor and public calls retain `api_key`, `chatgpt_account_id`,
`api_base`, `headers`, `originator`, `timeout`, and `codex_*` transport settings.
`auth_storage` stays a `CodexAuth` or path. Standard non-Codex LMs retain DSPy's
own behavior, including caller-supplied engines. `copy` and `dump_state` preserve
reconstructible adapter configuration; serialized state excludes credentials and
custom headers. Errors produced during engine execution use DSPy's public LM
error family; invalid call shapes and options fail before provider I/O.

## Clauses
- C1 [request] Sync and async Codex calls execute through canonical engines, never the built-in OpenAI/LiteLLM transport, without deprecation warnings. Plain prompts retain DSPy's list output; explicit requests return a canonical response containing text, provider model, usage, reasoning and function calls. Raw provider response and selected transport are retained in `Response.provider_data` under `codex_response` and `codex_transport`.
- C2 [inferred] Canonical system text, user/assistant text, images, files, function declarations, function calls/results, JSON schema output, and reasoning settings map structurally to native Responses fields. Existing reasoning aliases and service-tier options remain supported for string calls. The shared client still removes output caps, normalizes `fast` to `priority`, and enforces stream/store. Any codec adaptations and the removed output cap remain visible in `Response.adaptations`. `Predict` with JSONAdapter, ChatAdapter few-shot examples, and plain-prompt reflection use the same engines.
- C3 [inferred] Transport and connection options retain constructor/per-call precedence without mutating defaults or leaking credentials into payloads, history, or saved state. Defaults remain auto/10 seconds connect/300 seconds idle. A per-call numeric timeout wins over per-call idle; per-call idle wins over constructor timeout; otherwise constructor numeric timeout wins over constructor idle. `timeout=None` clears a constructor timeout. Structured HTTP timeouts retain their phase deadlines. Concurrent async calls on one LM keep their own overrides.
- C4 [inferred] DSPy owns response caching and usage accounting. Equal calls reuse cached responses; changed account identity in the same auth file, explicit credential, account override, endpoint, transport, timeout, headers, or rollout produce distinct cache entries. Sync and async cache entries stay separate. File auth identity is read for every cache lookup; uncached requests use current file credentials. Cached responses add no billed usage; each uncached response is accounted for exactly once.
- C5 [inferred] DSPy owns the only retry loop: `num_retries=k` permits at most k+1 engine attempts, and shared client attempts use `max_retries=0`. Structured HTTP/WebSocket 429 and 5xx, network errors, and timeouts are retryable; authentication errors, invalid requests, refusals, and protocol errors remain visible without retry. Refusals never become empty successful predictions. Errors and cancellation never populate the successful response cache.
- C6 [inferred] Installation keeps Codex aliases (`codex`, `chatgpt`, `openai-codex`), explicit `auth_provider='codex'`, and the default Codex endpoint routing through Codex. Non-Codex routes retain DSPy behavior. Uninstall restores DSPy's class. Credentials are not read at construction. Copies preserve auth/transport configuration, isolate history, and accept connection overrides; saved/restored Codex LMs reconstruct the route and transport without saving secrets.
- C7 [request] The old LM call shapes are not retained. `messages=` is rejected with migration guidance, and `forward`/`aforward` are absent. Invalid transport values and nonpositive/nonfinite/bool timeouts fail before I/O. Codex options cannot be silently accepted on non-Codex routes. Wrong client-only options or attempts to replace the Codex engine fail loudly rather than leaking into provider payloads. Stop sequences are explicitly unsupported: this completion-only adapter cannot enforce a streaming stop. Invalid native `text`/`instructions` shapes are errors, not silently coerced values.

## Examples
- E1 (C1) Sync `lm('hello')` and async `lm.acall('hello')`, with provider text `ok`, return `[{'text': 'ok'}]` and no DeprecationWarning; the shared client receives the requested user text. Explicit Request yields Response with `text == 'ok'` and the provider's model.
- E2 (C1) A response containing reasoning `Need weather.`, function `lookup_weather` with call id `call_weather` and arguments `{"city":"Chicago"}`, and usage input=12/output=7/total=19 preserves those values in canonical parts/usage and retains raw metadata and selected websocket transport.
- E3 (C2) Request(system='Be concise.', messages=(user text+image URL, assistant 'example', user 'followup')) maps to instructions plus input_text/input_image/output_text in order. Tool call/result id `call_weather` remains correlated; file URL/data/id and schema fields remain intact.
- E4 (C2) `Predict('question -> answer')` with JSONAdapter receives `{"answer":"four"}` and returns answer `four`; ChatAdapter few-shot input contains assistant output_text; `lm('Reflect.', reasoning_effort='low', reasoning_summary='detailed', service_tier='fast')` sends the matching reasoning and tier.
- E5 (C3) Constructor websocket/timeout=600 then calls (http, connect=21, idle=22), (idle=23, timeout=700), defaults, (timeout=None, idle=24) yield respectively (http,21,22,600), (websocket,10,700,700), (websocket,10,600,600), (websocket,10,24,no timeout). Next default call still uses websocket/600. Concurrent async calls with HTTP and websocket overrides retain their own transport and key.
- E6 (C4) A repeated cached prompt makes one provider request and bills usage once; changing token/account/endpoint/transport/deadline/headers/rollout causes a new request, while restoring an earlier context reuses its cache. Updating account_id in one auth file causes one new provider request; another repeat hits that new cache.
- E7 (C5) `num_retries=1` with 429 then success makes two provider attempts, records one successful history entry, and caches only success. `num_retries=0` makes one attempt. A 401 or provider refusal raises a visible DSPy LM error after one attempt even when num_retries=2.
- E8 (C6) `install(); dspy.LM('chatgpt/gpt-test')('hi')` calls Codex; a non-Codex LM with a supplied echo engine returns that engine's response. Missing auth file permits construction. Copy with changed key/transport uses the new values and empty independent history. A state round trip restores the auth path/route/transport, but omits synthetic keys and Authorization headers.
- E9 (C7) `lm(messages=[...])` raises TypeError mentioning Request/3.5; accessing forward/aforward fails. Transport='invalid', connect_timeout=0, idle_timeout=NaN, or a Codex transport override on a non-Codex LM raises before I/O. `max_retries=`/`request_timeout=` passed as generation options are rejected.
- E10 (C2, C7) `Config(max_tokens=1234, seed=7)` records the removed cap and unsupported Responses seed in adaptations. `Config(stop=('END',))` raises LMUnsupportedFeatureError before provider I/O. Native `text=3` or `instructions=[]` fails visibly before provider I/O.

## Amendments
- A1: Explicitly record upstream codec adaptations and reject unsupported stop sequences. This makes the completion-only transport limitation visible instead of silently ignoring a requested stop. The red source revision remains the original legacy implementation; supplemental tests freeze with this amendment before further repairs.

## Not contractual
- Internal helper names, class decomposition, engine object identity, callback implementation, private DSPy method calls, ordering of JSON object keys, exact error prose beyond actionable migration guidance.
- Token streaming is not introduced by this migration; completion-only engines retain the existing fully collected response behavior.
- DSPy 3.2/3.4 compatibility, the removed OpenAI messages dictionary API, and legacy forward hooks are not supported in this release.

## Dependency graph
```yaml
tasks:
  - id: T1
    title: Freeze public behavioral tests against DSPy 3.5
    depends_on: []
  - id: T2
    title: Implement canonical Codex engines and LM configuration
    depends_on: [T1]
  - id: T3
    title: Update documentation and distribution verification
    depends_on: [T2]
  - id: T4
    title: Validate, release to GitHub and PyPI, verify fresh installation
    depends_on: [T2, T3]
```
