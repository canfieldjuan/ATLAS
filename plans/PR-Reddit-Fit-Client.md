# PR-Reddit-Fit-Client

## Why this slice exists

Fifth slice of the approved Reddit Listening v2 arc (#1931 comment
4872154794), after S1-S4. This adds the LLM judge CLIENT -- one narrow
OpenAI-compatible chat-completions call behind an injectable transport --
so a real model can produce fit output that the arc already knows how to
parse (S2), guard (S3), and persist (S4). No live network in CI (the HTTP
boundary is faked); the runner that USES the client is S6.

## Scope (this PR)

Ownership lane: content-ops/reddit-listening/fit-eval
Slice phase: Vertical slice

1. `atlas_reddit/config.py`: `ATLAS_REDDIT_FIT_*` settings -- `fit_backend`
   (off|openrouter|local), `fit_base_url`, `fit_model`, `fit_api_key`
   (SecretStr), `fit_max_calls_per_run`, `fit_min_score`,
   `fit_timeout_seconds` -- with `MAX_FIT_*` shared constants + `le=` bounds.
2. `atlas_reddit/fit_client.py`: `OpenAICompatibleJudgeClient` (stdlib
   urllib behind an injectable `Transport`; POST `{base}/chat/completions`;
   `judge()` -> `(FitDecision | None, FitCallMeta)`); `build_judge_client`
   (returns None when off, fails closed on a misconfigured backend);
   `FitClientError` / `RedditFitConfigError`; `FitCallMeta`.
3. `tests/test_atlas_reddit_fit_client.py`: fail-closed config, both
   backends' structured-output strategy, happy path + usage, the
   no-Reddit-creds-in-request probe, the malformed-content->parse_error
   path, HTTP/envelope errors, and the no-network purity probe.

### Review Contract

- Acceptance criteria:
  - [ ] `build_judge_client` returns None for backend=off; fails closed
        (`RedditFitConfigError`) on invalid backend, missing base_url/model,
        or missing key for openrouter; a local backend may run keyless.
  - [ ] `judge()` returns a parsed `FitDecision` + token usage on valid
        content; malformed model content returns `(None, meta)` with a
        closed `PARSE_ERROR_CODES` value (data, not a crash).
  - [ ] HTTP non-2xx and non-OpenAI-shaped responses raise
        `FitClientError`; a transport error surfaces as `FitClientError`.
  - [ ] The request body + headers carry NO Reddit credentials, and the
        bearer is the FIT key -- never the B2B/global OpenRouter key.
  - [ ] OpenRouter uses strict `json_schema`; local uses `json_object`.
  - [ ] No live network in CI: every test uses the injectable transport;
        the module imports nothing but stdlib + the standalone package.
- Reachability proof (#1952): the real `build_judge_client` + `judge()`
  path is exercised end-to-end through a fake transport that returns an
  OpenAI-shaped body, yielding a real `FitDecision` -- the observable
  result. Live-backend wiring (the S6 `judge-fit` runner) is the named
  next slice.
- Affected surfaces: config settings, one new standalone module, one test
  file. No store, digest, poller, or Reddit-auth surface.
- Risk areas: credential isolation (own namespace; probed); structured-
  output variance across backends (json_object default, json_schema for
  OpenRouter); failure taxonomy (config vs transport vs parse -- each
  probed).
- Reviewer rules triggered: R1, R2 (both sides: valid + every failure
  class), R11 (zero new dependencies; stdlib urllib), R12 (test
  auto-enrolls via the glob), R14 (reachability + credential-isolation
  probe named above).
- Test-adapter posture (#1934 real-adapters rule): the HTTP/model boundary
  is the one true external surface, faked via an injectable transport;
  real settings, real parser, real config validation.

### Files touched

- `atlas_reddit/config.py`
- `atlas_reddit/fit_client.py`
- `plans/INDEX.md`
- `plans/PR-Reddit-Fit-Client.md`
- `plans/archive/PR-Reddit-Fit-Store.md`
- `tests/test_atlas_reddit_fit_client.py`

## Mechanism

`OpenAICompatibleJudgeClient.judge()` builds `{model, messages,
temperature, max_tokens, response_format}` and POSTs to
`{base_url}/chat/completions` via the transport. `response_format` is
strict `json_schema` (wrapping `FIT_OUTPUT_JSON_SCHEMA`) for OpenRouter and
`json_object` for local. A non-2xx status or a response that is not
`choices[0].message.content`-shaped raises `FitClientError`; valid content
goes through `parse_fit_decision`, so malformed model output returns
`(None, meta)` with `meta.parse_error` a closed `PARSE_ERROR_CODES` value
(the same taxonomy the S1 harness envelope uses) and a valid decision
returns `(FitDecision, meta)` with token usage. `build_judge_client` gates
construction on the backend and its required fields, returning None for
off and raising `RedditFitConfigError` otherwise. The bearer header is set
only from `fit_api_key`; the request never sees Reddit credentials.

## Intentional

- **stdlib urllib, not httpx**: the atlas_reddit CI job installs no httpx
  and the package is deliberately stdlib-first; one POST needs nothing
  more. The transport is injectable so tests fake it and CI stays offline.
- **Own credential namespace**: `ATLAS_REDDIT_FIT_API_KEY`, never the
  B2B/global OpenRouter key; the client is standalone and imports no
  atlas_brain.
- **Malformed content is data, not an exception**: `judge()` returns a
  parse_error code so S6 drops it straight into a prediction envelope; only
  transport/HTTP/envelope failures raise.
- **FitParseError.code IS the envelope code**: no translation layer, so
  the client and the S1 harness speak the same closed taxonomy.
- **json_object default**: strict `json_schema` is attempted only for
  OpenRouter; the S2 parser remains the authoritative gate regardless.

## Deferred

- S6 fit runner + digest integration: the `judge-fit` command that selects
  candidates (reusing `list_candidates` + the `fit_min_score` gate + the
  `fit_max_calls_per_run` cap), calls this client, guards + persists via
  the S4 store, renders guard_ok reviews in the digest, and offers the
  `--eval-cases` harness mode. Per the approved arc.

Parked hardening: none.

## Verification

- `.venv/bin/python -m pytest tests/test_atlas_reddit_fit_client.py -q`:
  15 passed (fail-closed config, structured-output per backend, happy path
  + usage, no-Reddit-creds probe, malformed-content parse_error, HTTP and
  envelope errors, purity).
- Full package suite `.venv/bin/python -m pytest
  tests/test_atlas_reddit_*.py -q`: 533 passed.
- Reachability: `build_judge_client(settings, transport=fake).judge(msgs)`
  returns a real `FitDecision` with token usage; no network touched.
- ASCII byte-scan on changed Python files: clean.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_reddit/config.py` | 46 |
| `atlas_reddit/fit_client.py` | 195 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Reddit-Fit-Client.md` | 128 |
| `plans/archive/PR-Reddit-Fit-Store.md` | 0 |
| `tests/test_atlas_reddit_fit_client.py` | 251 |
| **Total** | **623** |
