# PR-Reddit-Fit-Eval-Harness

## Why this slice exists

First slice of the approved Reddit Listening v2 arc (#1931 comment
4872154794; Fable trial continuation after the merged MVP arc #1934). The
operator's revision (comment 4872062786) fixed the order: the evaluation
harness ships FIRST -- "the harness becomes the ruler" -- so later model
slices must satisfy a judgment contract instead of merely proving JSON
parsing. This slice delivers that ruler: the claim-safety rule catalogue,
a 16-case fixture corpus across 8 categories, deterministic grading of
prediction envelopes, and a CLI that writes machine-readable summaries.
Zero model calls, zero network, zero store/digest changes.

Diff-budget note: over the soft cap because the ruler must prove BOTH
sides at birth -- the corpus ships pass AND fail prediction files, and
every fail envelope declares the exact checks and codes it must fire,
enforced by tests. Splitting fixtures out would ship an unproven ruler.

## Scope (this PR)

Ownership lane: content-ops/reddit-listening/fit-eval
Slice phase: Robust testing

1. `atlas_reddit/fit_rules.py`: the single-source claim-safety catalogue
   (20 rules in three families: 12 claim, 2 posture, 6 PII) + the fit
   output contract constants (verdicts, risk flags, 280-char caps) +
   `scan_fit_text` with per-fixture PII allowlist. Later slices import
   this: the S2 prompt builder renders boundaries from rule messages, the
   S3 runtime guard enforces the same catalogue under a parity test.
2. `atlas_reddit/fit_eval.py`: fixture/envelope loading (fail-closed on
   duplicate/unknown case ids and malformed JSONL), the strict prediction
   shape checker (S1-local twin of the future S2 parser, same constants),
   per-case named checks, result/summary shapes, CLI `main`.
3. `scripts/evaluate_atlas_reddit_fit.py`: thin wrapper
   (`raise SystemExit(main())`), mirroring the support-ticket eval wrapper.
4. `tests/fixtures/atlas_reddit_fit_eval/`: `cases.jsonl` (16 cases, 2 per
   category), `predictions_pass.jsonl` (16 correct, grades fully green),
   `predictions_fail.jsonl` (16 characteristic failures incl. one
   prediction-null envelope and one strictness violation, each declaring
   `expects_failing_checks` + `expects_codes`).
5. `tests/test_atlas_reddit_fit_eval.py`: every rule family fired by
   parametrized probes; every shape-rejection class; the shipped-corpus
   contract (fail file fires EXACTLY its declared checks/codes); CLI exit
   contract through the real `main()`; summary privacy; purity probe.
6. `.github/workflows/atlas_reddit_checks.yml`: one `paths:` line so
   script-only changes also trigger the suite (the test glob already
   auto-enrolls the new test file).

### Review Contract

- Acceptance criteria:
  - [ ] Harness is deterministic: no model calls, no network, no clock,
        no randomness; purity probe pins no network/praw imports.
  - [ ] Both sides proven at the CLI: pass corpus exits 0; fail corpus
        exits 1 under `--fail-on-eval-fail` (0 without the flag; artifacts
        always written).
  - [ ] Every major failure branch fires at least once by NAME: the 9
        per-case checks, the 11 shape-rejection codes, all 20 rule codes.
  - [ ] Fixture labels live in JSONL (not hidden in code) so future
        models are evaluated against the same corpus.
  - [ ] Summary is machine-readable, `schema_version`-stamped, and
        privacy-stripped: codes + case ids only, never candidate or
        prediction text (probed against the PII trap).
  - [ ] Structural input defects (duplicate/unknown case ids, malformed
        JSONL) fail closed with exit 2 -- model misbehavior is DATA
        (graded FAIL), tooling breakage is an ERROR.
  - [ ] The existing no-write probe and fixture-fidelity tests stay green
        over the new modules; new tests auto-enroll via the workflow glob.
- Reachability proof (#1952): the real entrypoint
  `scripts/evaluate_atlas_reddit_fit.py` is executed against the shipped
  fixture pair; observable results = process exit codes (0 pass / 1 fail
  / 2 structural) and the written summary JSON artifact. Exercised both
  in the Verification commands below and in-process by the CLI tests.
- Affected surfaces: two new pure modules, one thin script, fixtures,
  one workflow paths line. No store, digest, poller, tracker, purge,
  config, or credential surface is touched.
- Risk areas: regex under/over-match on short advisory text (the trap
  corpus is the regression net); fixture rot (prevented by the
  expects-codes contract test).
- Reviewer rules triggered: R1, R2 (guard-shaped rules probed both
  sides: every family fires AND clean advisory text passes the whole
  catalogue), R11 (zero new dependencies; stdlib-only additions), R12
  (CI enrollment in the same PR), R14 (reachability proof named above).
- Test-adapter posture (#1934 real-adapters rule): nothing is faked --
  the harness has no external boundary; real rule catalogue, real fixture
  files from disk, real CLI main() in-process.

### Files touched

- `.github/workflows/atlas_reddit_checks.yml`
- `atlas_reddit/fit_eval.py`
- `atlas_reddit/fit_rules.py`
- `plans/PR-Reddit-Fit-Eval-Harness.md`
- `scripts/evaluate_atlas_reddit_fit.py`
- `tests/fixtures/atlas_reddit_fit_eval/cases.jsonl`
- `tests/fixtures/atlas_reddit_fit_eval/predictions_fail.jsonl`
- `tests/fixtures/atlas_reddit_fit_eval/predictions_pass.jsonl`
- `tests/test_atlas_reddit_fit_eval.py`

## Mechanism

`atlas_reddit/fit_rules.py` declares `FitRule(code, pattern, message)` tuples in three
families (claim, posture, PII) compiled once via `lru_cache`;
`scan_fit_text` returns findings carrying code + character span only --
never matched text -- so summaries stay privacy-strippable; a PII finding
is suppressed only when its exact matched text is in the caller's
allowlist. `atlas_reddit/fit_eval.py` loads cases and prediction envelopes fail-closed,
then grades each case through named checks: envelope present, strict
shape (exact 4 keys, verdict enum, 280-char caps after whitespace
collapse, verdict-conditional angle where a `no` must carry null/empty --
advisory text on a rejected thread is where pitch language leaks),
verdict against the case's allowed SET, reason/angle term grounding,
per-case forbidden terms, then the catalogue scan split into
`no_forbidden_claims` / `no_reply_draft` / `no_pii_echo`. A case passes
only when every check passes. The summary carries failing checks only,
stamped `atlas_reddit_fit_eval_summary.v1`. The CLI wires it together
with the repo's eval exit contract (0/1-under-flag/2) borrowed from the
local MCP eval harness.

## Intentional

- **Rules born here, not in the guard**: the S3 runtime guard and S2
  prompt builder will import `fit_rules`, so the ruler is mechanical --
  the model is told, graded on, and blocked by the same catalogue. The
  guard slice adds only blocking policy plus a parity test.
- **Model garbage is data**: emitters (S5/S6) will always write a valid
  envelope with `prediction: null` + `parse_error` on model failure; the
  harness grades that as a case FAIL. Only malformed FILES exit 2.
- **`expects_failing_checks`/`expects_codes` on fail envelopes**: legal
  extra keys on the tolerant envelope, asserted by tests -- the corpus
  cannot rot into passing for the wrong reason.
- **Findings carry spans, not text**: privacy discipline starts at the
  lowest layer, so no later surface has to remember to redact.
- **`no` verdicts require empty angle** (deterministic parse-level FAIL),
  resolving the operator contract's "absent or clearly declined".
- **The shape checker is an S1-local twin** of the S2 parser by design;
  S2 swaps it for the real parser and the harness must stay green -- that
  swap is the proof the ruler measures the runtime contract.

## Deferred

- S2 fit contract + prompt builder; S3 runtime guard (+ parity test);
  S4 store schema v5 (`candidate_fit_reviews` + approved `body_excerpt`)
  + manual import CLI; S5 OpenAI-compatible judge client
  (`ATLAS_REDDIT_FIT_*`); S6 runner + digest integration -- per the
  approved arc order (#1931 comment 4872154794).
- `required_risk_flags` per-case grading: field reserved in the design,
  not read yet (no fixture requires a specific flag in v1).

Parked hardening: none.

## Verification

- `.venv/bin/python -m pytest tests/test_atlas_reddit_fit_eval.py -q`:
  52 passed (every rule family, every shape-rejection code, corpus
  both-sides contract, CLI exits, privacy, purity).
- Full package suite `.venv/bin/python -m pytest
  tests/test_atlas_reddit_*.py -q`: 385 passed (no-write probe and
  fixture-fidelity stay green over the new modules).
- Reachability (real entrypoint, observable result):
  `python scripts/evaluate_atlas_reddit_fit.py --cases
  tests/fixtures/atlas_reddit_fit_eval/cases.jsonl --predictions
  tests/fixtures/atlas_reddit_fit_eval/predictions_pass.jsonl
  --fail-on-eval-fail` exits 0; same command with
  `predictions_fail.jsonl` and a `--summary-output` path exits 1 and
  writes the summary artifact.
- ASCII byte-scan on all new files: clean.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_reddit_checks.yml` | 1 |
| `atlas_reddit/fit_eval.py` | 389 |
| `atlas_reddit/fit_rules.py` | 287 |
| `plans/PR-Reddit-Fit-Eval-Harness.md` | 166 |
| `scripts/evaluate_atlas_reddit_fit.py` | 15 |
| `tests/fixtures/atlas_reddit_fit_eval/cases.jsonl` | 16 |
| `tests/fixtures/atlas_reddit_fit_eval/predictions_fail.jsonl` | 16 |
| `tests/fixtures/atlas_reddit_fit_eval/predictions_pass.jsonl` | 16 |
| `tests/test_atlas_reddit_fit_eval.py` | 322 |
| **Total** | **1228** |
