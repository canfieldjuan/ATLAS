# PR-Security-Structured-JSON-Logging

## Why this slice exists

#1656 lists "Structured JSON logging + error tracking" as a remaining security
hardening gap. `atlas_brain.main` currently configures plain text stdout logs
with `logging.basicConfig`, so production operators cannot reliably parse log
records, correlate error events, or extract exception details from the main
service without ad hoc text matching.

Root cause: Atlas has only a text formatter at the application startup layer;
there is no structured formatter contract for the main service. This slice
fixes the first vertical slice of that root by adding opt-in JSON log formatting
with exception fields and startup wiring. External error tracking remains a
follow-up so this PR does not also introduce exporter infrastructure.

Review-fix root cause: the first JSON formatter implementation treated "can be
passed to `json.dumps` once" as the parseability contract, then merged extras
directly into the schema and only formatted root logger handlers. That fixed the
happy path but left the production stream able to emit invalid JSON, drop records
on mixed-key nested extras, clobber reserved fields, and mix uvicorn text lines
with app JSON lines. This revision fixes the root within this slice by making
the formatter normalize every extra recursively, emit strict JSON, namespace
extras, and format uvicorn handlers too.

Diff budget: this revision is over the 400 LOC target because the review found a
formatter defect class rather than one isolated bug. The added LOC is mostly
focused adversarial tests for the five same-class failures: reserved-field
collisions, non-finite numbers, cached exception text, uvicorn handlers, and
mixed-key nested extras.

## Scope (this PR)

Ownership lane: security/structured-json-logging
Slice phase: Production hardening
Max files: 7

1. Add an `ATLAS_LOG_FORMAT` setting with `text` default and `json` opt-in.
2. Add a reusable Atlas JSON log formatter that emits parseable timestamp,
   level, logger, message, source location, extra fields, and exception details.
3. Wire `atlas_brain.main` startup logging through the new configuration helper.
4. Add exception tracebacks to caught startup/shutdown error logs so JSON mode
   carries real exception objects instead of only stringified messages.
5. Archive the merged #1823 plan doc as required teardown housekeeping.

### Review Contract

- Acceptance criteria:
  - [ ] Default `text` logging keeps the existing timestamp/level/logger/message
        format.
  - [ ] `json` logging emits valid JSON with stable top-level fields.
  - [ ] Exception logs include type, message, and traceback text.
  - [ ] Cached `exc_text` records still produce a structured exception object.
  - [ ] Extra log-record fields are preserved under `extra`, recursively
        normalized, and safely stringified otherwise.
  - [ ] Reserved schema fields cannot be overwritten by caller extras.
  - [ ] Non-finite floats are stringified so the emitted line is strict JSON.
  - [ ] Nested mixed-key dictionaries cannot make the formatter raise.
  - [ ] Uvicorn access/error handlers receive the JSON formatter when JSON mode
        is enabled.
  - [ ] Unknown log formats fail closed instead of silently falling back.
- Affected surfaces: Atlas Brain startup logging configuration, logging helper
  tests, startup/shutdown exception call sites, uvicorn process logging, and plan
  archive housekeeping.
- Risk areas: changing pytest/root logger state, breaking existing plain-text
  local logs, adding traceback noise to caught error paths, and losing exception
  information.
- Reviewer rules triggered: R1, R2, R3, R5, R8, R11, R12, R14.

### Files touched

- `atlas_brain/config.py`
- `atlas_brain/logging_config.py`
- `atlas_brain/main.py`
- `plans/INDEX.md`
- `plans/PR-Security-Structured-JSON-Logging.md`
- `plans/archive/PR-Security-Paid-Funnel-Alert-Channel.md`
- `tests/test_atlas_main_voice_startup.py`

## Mechanism

`atlas_brain.logging_config` provides `AtlasJsonFormatter`,
`build_log_formatter`, and `configure_logging`. `configure_logging` keeps the
existing `logging.basicConfig` path for `text`; when `json` is selected it
ensures a root stream handler exists and installs the JSON formatter on root and
uvicorn handlers. The formatter emits UTC ISO timestamps, level, logger,
message, module/function/line, optional exception details, cached exception text,
and non-standard record attributes under `extra`.

The formatter normalizes extras recursively before the final strict
`json.dumps(..., allow_nan=False)`: mapping keys become strings, nested values
are sanitized, non-finite floats become strings, unsupported objects become
strings, and caller extras cannot overwrite the canonical schema fields.

`main.py` calls `configure_logging` using `settings.log_level` and the new
`settings.log_format` field. Caught startup/shutdown error logs pass
`exc_info=True` so JSON mode includes real exception type/message/traceback
fields for those failure paths.

## Intentional

- JSON logging is opt-in for this slice so existing local/dev log expectations
  remain stable.
- No Sentry/OTel dependency is added here; this slice creates structured error
  events that a later exporter can consume.
- The formatter avoids adding process/thread metadata until a concrete consumer
  needs it, keeping the JSON contract small.
- Caller extras are intentionally namespaced under `extra` instead of being
  promoted to top-level fields; the top-level JSON object is the stable schema.

## Deferred

- #1656 follow-up: add Sentry or OpenTelemetry export once the destination and
  environment variables are chosen.
- #1656 follow-up: add request/correlation IDs to API logs after the middleware
  boundary is named.

Parked hardening: none.

## Verification

- Focused logging/startup tests: `28 passed, 1 warning in 2.48s`.
- Python compile check for touched runtime/test modules: passed.
- Whitespace diff check: passed.
- Local review bundle: passed via `scripts/push_pr.sh` pre-push hook.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/config.py` | 4 |
| `atlas_brain/logging_config.py` | 162 |
| `atlas_brain/main.py` | 66 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Security-Structured-JSON-Logging.md` | 138 |
| `plans/archive/PR-Security-Paid-Funnel-Alert-Channel.md` | 0 |
| `tests/test_atlas_main_voice_startup.py` | 198 |
| **Total** | **571** |
