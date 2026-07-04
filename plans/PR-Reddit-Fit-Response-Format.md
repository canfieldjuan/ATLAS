# PR-Reddit-Fit-Response-Format

## Why this slice exists

Live testing of the merged v2 fit pass against a local LM Studio server
found a real integration bug the 562 unit tests could not: the S5 client
sends `response_format: {"type": "json_object"}` for local backends, but
LM Studio rejects it with HTTP 400 (`'response_format.type' must be
'json_schema' or 'text'`). Every local fit call fails. The fake transport
in the unit suite never spoke LM Studio's dialect, so it passed. This makes
the local backend -- the whole point of a keyless, offline judge -- unusable
as shipped.

## Scope (this PR)

Ownership lane: content-ops/reddit-listening/fit-eval
Slice phase: Production hardening

Root cause: response_format was tied to the BACKEND NAME with a wrong
assumption ("local servers are model-dependent, default json_object"). Which
structured-output mode a server accepts is a property of the SERVER, not the
backend name -- and json_object is in fact *less* supported by modern local
servers than json_schema.

1. `atlas_reddit/config.py`: `FIT_RESPONSE_FORMATS` constant +
   `fit_response_format` setting (`json_schema` default | `json_object` |
   `text`).
2. `atlas_reddit/fit_client.py`: `_response_format()` is mode-driven (not
   backend-driven); `text` returns None so the request omits the key
   entirely; `build_judge_client` validates the mode fail-closed and threads
   it into the client.
3. `tests/test_atlas_reddit_fit_client.py`: default is json_schema for both
   backends; json_object and text modes; invalid mode fails closed.

### Review Contract

- Acceptance criteria:
  - [ ] Default `fit_response_format` is `json_schema` for BOTH local and
        openrouter (the previously-broken local path now works).
  - [ ] `json_object` mode sends `{"type": "json_object"}`; `text` mode omits
        `response_format` from the payload entirely.
  - [ ] An invalid mode raises `RedditFitConfigError` (fail closed), like the
        other fit-config validation.
  - [ ] The parser remains the authoritative gate in every mode (text sends
        no server constraint yet still parses/validates).
- Reachability proof (#1952): with `ATLAS_REDDIT_FIT_BACKEND=local` +
  default json_schema, `judge-fit --eval-cases` against a real LM Studio
  server returns 16/16 parsed predictions (0 unparsed) -- where the old
  json_object code returned 16/16 `model_http_error`. Verified live, in
  Verification below.
- Affected surfaces: one config field, the client's response_format
  selection, one test. No store, poller, digest, runner, or Reddit-auth
  change.
- Risk areas: back-compat (a server that only supports json_object now needs
  the explicit setting -- documented; default serves the common case);
  text-mode payload shape (key omitted, not null).
- Reviewer rules triggered: R1, R2 (each mode + fail-closed both sides),
  R11 (zero new dependencies), R12 (test auto-enrolls via the glob), R14
  (live reachability named above).
- Test-adapter posture (#1934): unit tests fake the HTTP boundary
  (injectable transport); the live LM Studio proof is the real integration.

### Files touched

- `atlas_reddit/config.py`
- `atlas_reddit/fit_client.py`
- `plans/PR-Reddit-Fit-Response-Format.md`
- `tests/test_atlas_reddit_fit_client.py`

## Mechanism

`OpenAICompatibleJudgeClient` gains a `response_format` mode. The
mode-selection helper returns the strict json_schema dict for `json_schema`,
`{"type": "json_object"}` for `json_object`, and None for `text`; the judge
call adds the `response_format` key only when non-None, so text mode sends an
unconstrained request. `build_judge_client` validates
`settings.fit_response_format` against `FIT_RESPONSE_FORMATS` (fail closed)
and passes it through. The default `json_schema` is what LM Studio, vLLM and
OpenRouter all accept.

## Intentional

- **json_schema default, not backend-conditional**: the old backend-name
  branch was the bug. One default that modern servers accept, overridable
  for the rare server that needs json_object or neither.
- **text omits the key**: a server that supports no structured-output mode
  still works -- the S2 parser validates the content regardless, so text is
  a safe universal fallback.
- **Parser stays the only trusted gate**: response_format is a compliance
  hint, never the validation boundary.

## Deferred

- None. This is a focused fix.

## Verification

- `.venv/bin/python -m pytest tests/test_atlas_reddit_fit_client.py -q`:
  22 passed (default json_schema both backends; json_object; text-omits-key;
  invalid fails closed).
- Full package suite `tests/test_atlas_reddit_*.py -q`: 565 passed.
- LIVE reachability (real LM Studio, qwen3.6-35b-a3b): `judge-fit
  --eval-cases` with `ATLAS_REDDIT_FIT_BACKEND=local` (default json_schema)
  returns 16/16 parsed predictions, 0 unparsed. The pre-fix json_object path
  returned 16/16 model_http_error.
- ASCII byte-scan on changed Python files: clean.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_reddit/config.py` | 16 |
| `atlas_reddit/fit_client.py` | 29 |
| `plans/PR-Reddit-Fit-Response-Format.md` | 116 |
| `tests/test_atlas_reddit_fit_client.py` | 31 |
| **Total** | **192** |
