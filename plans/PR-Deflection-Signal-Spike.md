# PR-Deflection-Signal-Spike

## Why this slice exists

#1612's plan lock says S1 must not freeze the new action-section report
contract until a Signal Spike answers what the real Zendesk-shaped source data
can actually support: CSAT availability/trustworthiness, defensible cost basis,
deterministic owner-lane/fix-type feasibility, support-resolution evidence, and
snippet/phrasing safety. #1741 closed the immediate scrub regression, so the
next root-cause step is not adding sections; it is proving which inputs are
ready, partial, or insufficient before S1 bakes them into `deflection.v1`.

Root cause: the planned report shape depends on data assumptions that are not
captured by the current full-report QA harness. Existing tests prove rendering,
artifact consistency, and some synthetic outcome diagnostics, but they do not
produce a sanitized, repeatable source-availability summary an operator can run
against a local real export. This PR fixes that discovery gap; it does not
change customer-facing report output.

Diff budget note: this slice exceeds the 400 LOC soft cap because the
operator-run probe, synthetic signal-present/absent fixtures, CLI failure
branch, no-raw-source output tests, CI enrollment, and summary-only spike
artifact are one safety contract. Splitting the tests/artifact from the probe
would leave the privacy boundary and `insufficient_data` behavior unproven.

## Scope (this PR)

Ownership lane: content-ops/deflection-report-actionability
Slice phase: Functional validation

1. Add an operator-run Signal Spike CLI that reads a local support-ticket CSV,
   JSON, JSONL, or Zendesk full-thread JSON export and emits only sanitized
   aggregate availability/readiness signals.
2. Summarize S1 dependencies as `ready`, `partial`, or `insufficient_data`:
   support-resolution evidence, CSAT prioritization, cost basis,
   owner-lane/fix-type feasibility, and snippet/phrasing projection safety.
3. Add fixture tests proving both useful-signal and insufficient-data cases,
   plus output redaction/no-raw-source guarantees.
4. Enroll the probe test in extracted checks and commit a summary-only run
   against the existing sanitized Zendesk-shaped product-proof corpus.
5. Fix review-discovered overclaim cases before S1 consumes the spike: cost
   fields count only included rows, single-ticket Zendesk exports auto-detect,
   singleton CSAT averages are suppressed, and owner-lane/fix-type readiness
   requires product plus problem context.

### Review Contract

- Acceptance criteria:
  - [ ] The CLI accepts generic support-ticket row files and Zendesk full-thread
        exports without requiring raw data to be committed.
  - [ ] Output contains only aggregate counts/statuses and safe enum strings;
        it does not echo ticket IDs, source text, emails, URLs, or snippets.
  - [ ] CSAT and cost degrade honestly: sparse/textual CSAT is `partial`, absent
        CSAT is `insufficient_data`, and the current report cost basis is
        labeled benchmark-only unless source cost fields are present.
  - [ ] Owner-lane/fix-type readiness reports `Unknown` fallback pressure when
        structured product/issue/category fields are sparse.
  - [ ] Snippet/phrasing safety is explicitly marked as scrubbed-only /
        allowlist-required, not declared fully safe from an unlabeled export.
  - [ ] `--require-s1-ready` exits non-zero when any S1 dependency remains
        `insufficient_data`.
- Affected surfaces: one operator script, its tests, extracted-checks
  enrollment, a summary-only validation artifact, and this plan doc. No report
  renderer, snapshot, payment, delivery, or hosted page changes.
- Risk areas: accidentally committing or printing raw source material, or
  overclaiming that a sparse export supports CSAT/cost-driven sections.
- Reviewer rules triggered: R1, R2, R6, R9, R10, R14.

### Files touched

- `.github/workflows/extracted_pipeline_checks.yml`
- `docs/extraction/validation/fixtures/deflection_signal_spike_20260620/summary.json`
- `plans/PR-Deflection-Signal-Spike.md`
- `scripts/probe_deflection_signal_spike.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/maturity_sweep/baseline_deflection_lane.json`
- `tests/maturity_sweep/baseline_scripts.json`
- `tests/test_probe_deflection_signal_spike.py`

## Mechanism

`scripts/probe_deflection_signal_spike.py` loads the source file into
support-ticket rows. For generic JSON/JSONL/CSV it preserves row aliases and
feeds them into `build_support_ticket_input_package`; for Zendesk full-thread
JSON it first uses `rows_from_zendesk_full_thread`, matching the existing
product-proof importer. Auto-detection requires actual thread shape, so a
generic JSON row with `satisfaction_rating` stays generic and keeps
`resolution_text`. It then builds a sanitized JSON summary from package metadata
and normalized rows:

- row counts and package/import warning counts;
- support-resolution evidence count and readiness;
- status/CSAT coverage and CSAT basis (`numeric`, `textual`, `mixed`, or
  `absent`);
- source cost-field availability on the included/truncated row set, versus the
  current benchmark-only report cost basis;
- structured context coverage, complete product-plus-problem context coverage,
  and explicit `Unknown` fallback pressure;
- scrub/projection safety counts for customer text/resolution text, with a
  hard statement that S1 still needs fail-closed allowlist projection for
  snippet-bearing fields.

The CLI writes this aggregate JSON to stdout or `--output`. It never writes raw
rows, source IDs, snippets, emails, URLs, or scrubbed text samples. The
`--require-s1-ready` mode is a fail-closed reviewer/operator guard for cases
where the caller expects every S1 dependency to be at least partial/ready.
Numeric CSAT averages are omitted until at least three numeric scores are
included, so a one-ticket export cannot reveal a per-ticket satisfaction value
inside a summary-only artifact.

The committed
`docs/extraction/validation/fixtures/deflection_signal_spike_20260620/summary.json`
run against the 50-ticket Zendesk-shaped corpus records the current pre-S1
decision: support-resolution evidence is ready; status is ready; CSAT is
partial/textual; cost is partial/benchmark-only; owner-lane/fix-type must use
`Unknown` fallback; and snippet projection remains partial until S1's
fail-closed allowlist plus shared detector contract lands.

## Intentional

- No customer-facing report sections in this slice. The Signal Spike informs S1;
  it does not start S1.
- No raw proof bundle is committed. Real exports stay local; committed fixtures
  are synthetic or summary-only outputs from already-sanitized fixtures.
- No attempt to infer owner lane or fix type with an LLM. This slice measures
  deterministic field availability and reports `Unknown` fallback pressure.
- Maturity-sweep baseline updates are narrow and intentional: they accept the
  reviewed-benign heuristic score for the new probe script only. The
  `_intable()` swallowed-except remains a boolean predicate, not an error
  boundary, and parse failures still fail closed via `SystemExit`.
- Snippet safety remains `partial` by design: without labeled PII recall data,
  the spike can prove the hardened scrub is invoked and can count redactions,
  but it cannot prove open-set recall. That root privacy boundary remains S1's
  fail-closed projection + shared detector contract.

## Deferred

- S1 action-section report model contract: fail-closed snapshot allowlist,
  unified detector contract, action-section section IDs/data contract, and
  cost/CSAT graceful-degradation fields.
- #1742 recall/precision harness: measure PII recall/precision on a
  surrogated, labeled eval artifact. This spike is source-availability
  discovery, not recall scoring.
- Actual operator run against the chosen real customer export. This PR includes a
  summary-only run against the existing sanitized Zendesk-shaped product-proof
  corpus, not a new raw/customer export.

Parked hardening: none. The maturity-sweep baseline entries are accepted
heuristic inventory for this new operator probe, not parked product debt.

## Verification

- Command: `python -m py_compile` on `scripts/probe_deflection_signal_spike.py`
  and `tests/test_probe_deflection_signal_spike.py` -- passed.
- Command: `python -m pytest` on `tests/test_probe_deflection_signal_spike.py`
  with `-q` -- 9 passed.
- Command: `python` on `scripts/audit_extracted_pipeline_ci_enrollment.py` --
  passed; 187 matching tests enrolled.
- Command: `python` on `scripts/probe_deflection_signal_spike.py` with
  `docs/extraction/validation/fixtures/zendesk_product_proof_corpus.json`,
  `--source-format json`, `--zendesk-thread auto`, `--output`
  `docs/extraction/validation/fixtures/deflection_signal_spike_20260620/summary.json`,
  and `--require-s1-ready` -- passed; regenerated the committed summary-only
  spike artifact.
- Command: `python scripts/maturity_sweep.py scripts --tests-root tests
  --baseline tests/maturity_sweep/baseline_scripts.json --update-baseline` --
  passed; then baseline diff was narrowed to the new probe entry only.
- Command: `python scripts/maturity_sweep.py scripts --tests-root tests
  --baseline tests/maturity_sweep/baseline_scripts.json --min-score 8
  --sensitive-glob 'scripts/**'` -- passed; ratchet gate reported no new
  brittleness above baseline.
- Command: Python invocation of `scripts/maturity_sweep_file_lane.py` over the CI
  deflection-file lane with `--tests-root tests --baseline
  tests/maturity_sweep/baseline_deflection_lane.json --update-baseline` --
  passed; then baseline diff was narrowed to the new probe entry only.
- Command: Python invocation of `scripts/maturity_sweep_file_lane.py` over the CI
  deflection-file lane with `--tests-root tests --baseline
  tests/maturity_sweep/baseline_deflection_lane.json --min-score 8` plus the
  workflow sensitive globs -- passed; ratchet gate reported no new brittleness
  above baseline.
- Command: `scripts/run_extracted_pipeline_checks.sh` through bash -- passed;
  reasoning core 295 passed, extracted content 4727 passed / 15 skipped.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/extracted_pipeline_checks.yml` | 4 |
| `docs/extraction/validation/fixtures/deflection_signal_spike_20260620/summary.json` | 102 |
| `plans/PR-Deflection-Signal-Spike.md` | 196 |
| `scripts/probe_deflection_signal_spike.py` | 646 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/maturity_sweep/baseline_deflection_lane.json` | 8 |
| `tests/maturity_sweep/baseline_scripts.json` | 8 |
| `tests/test_probe_deflection_signal_spike.py` | 305 |
| **Total** | **1270** |
