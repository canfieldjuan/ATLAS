# PR-Brand-Voice-Json-Report

## Why this slice exists

PR #1354 explicitly deferred JSON CLI/report mode. The brand-voice gate is now
readable in CI logs, but scripts still have to scrape text to count findings,
inspect severities, or surface suggestions. This slice adds a thin
machine-readable result path while preserving default text output and exit
codes.

Review on #1356 found that malformed config errors still escaped as tracebacks
under `--format json`. This update keeps the slice in the same lane by making
reachable setup failures parseable too. The estimated diff is now over the 400
LOC soft cap because the review fix adds bad-config regression tests for both
shape-invalid YAML and syntax-invalid YAML.

## Scope (this PR)

Ownership lane: content-marketing/brand-voice-checks
Slice phase: Vertical slice

1. Add a `--format {text,json}` CLI option to
   `atlas_brain/brand/voice_validator.py`, defaulting to today's text output.
2. Emit a JSON validation-result envelope with `ok`, `status`, `file`,
   `content_type`, `strict`, `summary`, and a `findings` array carrying each
   finding's structured fields.
3. Preserve the existing default/strict exit-code matrix for clean,
   advisory-only, blocking-only, and mixed findings.
4. Make `--format json` fail closed with a parseable `error` envelope when the
   requested input file is missing or the brand-voice config is invalid.
5. Add focused CLI tests that parse real stdout as JSON for clean,
   advisory-default, advisory-strict, mixed blocking/advisory, missing-file,
   bad-config-shape, and bad-config-syntax cases.
6. Document the local JSON command in the marketing README.

### Review Contract

- Acceptance criteria:
  - [ ] Text mode remains the default and existing text CLI assertions pass.
  - [ ] Clean JSON exits 0 with `ok: true`, `status: pass`, zero counts, and no findings.
  - [ ] Advisory JSON exits 0 by default with `status: warn` and suggestion metadata.
  - [ ] Advisory JSON exits 1 under `--strict` with `status: fail`.
  - [ ] Mixed JSON exits 1 and reports blocking/advisory counts plus all fields.
  - [ ] Missing-file JSON exits 1 with a parseable `error` object.
  - [ ] Bad-config JSON exits 1 with a parseable `invalid_config` error object.
- Affected surfaces: brand-voice CLI, validator tests, marketing guide.
- Risk areas: changed default CLI behavior, contradictory JSON/exit status,
  missing suggestion metadata, unparseable error output.
- Reviewer rules triggered: R1, R2, R10, R11.

### Files touched

- `atlas_brain/brand/voice_validator.py`
- `marketing/README.md`
- `plans/PR-Brand-Voice-Json-Report.md`
- `tests/test_brand_voice_validator.py`

## Mechanism

The CLI already splits `BrandVoiceFinding` objects into blocking and advisory
groups. This PR reuses that split, adds a finding serializer, derives
`exit_code`/`status` once, and renders either the existing text helpers or one
JSON object with `ok`, `status`, `content_type`, `summary`, `findings`, and
optional `error`. The CLI wraps validator construction/validation errors so
bad configs return `invalid_config` envelopes in JSON mode and clean `Error:`
lines in text mode. JSON mode exits with the same code the text path would
return for the same findings and strict flag.

## Intentional

- The option is named `--format` instead of a standalone `--json` flag because
  it leaves room for future formats without adding mutually exclusive switches.
- Missing input files get `file_not_found`; invalid config/setup failures get
  `invalid_config` so downstream consumers can distinguish read failures from
  config failures without parsing tracebacks.
- No workflow change yet; CI should stay optimized for readable failure logs
  until a downstream consumer needs JSON artifacts.

## Deferred

- Stem-aware vocabulary patterns.
- Larger marketing corpus expansion.
- Workflow-level strict policy if needed later.
- A JSON `schema_version` field if/when a downstream consumer needs schema
  negotiation.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_brand_voice_validator.py -q` -- 72 passed.
- Seed content CLI checks for landing page, blog post, release notes, and tweet
  -- PASS.
- JSON CLI smokes for clean, advisory, mixed, and missing-file outputs with
  `--format json` -- passed; exit codes were clean:0, advisory:0, mixed:1,
  missing:1.
- Review-fix JSON CLI smokes for bad config shape and bad YAML syntax with
  `--format json` -- passed; exit codes were bad_shape:1, bad_syntax:1.
- `bash scripts/local_pr_review.sh --current-pr-body-file <body-file>`
  -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/brand/voice_validator.py` | 153 |
| `marketing/README.md` | 7 |
| `plans/PR-Brand-Voice-Json-Report.md` | 110 |
| `tests/test_brand_voice_validator.py` | 230 |
| **Total** | **500** |
