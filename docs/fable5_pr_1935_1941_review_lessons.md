# Fable 5 Arc (PRs 1935-1941) Lessons Index

Date: 2026-07-02. Sources: the #1934 trial arc (PRs 1935-1941), the #1936
reconciliation case, and the post-merge audits recorded on #1940/#1941/#1934.

This is an INDEX, not a checklist for a model to follow. The arc produced
~38 reviewer-caught defects that the builder's own suites missed while the
rules lived in prose, so the standing rule of this document is:

**A lesson may only be listed here with an enforcement status. Prose that
asks a future builder to "remember" something is a failure of this index,
not a norm.**

## Lesson-to-enforcement map

| # | Lesson (arc evidence) | Status | Enforced by |
|---|---|---|---|
| 1 | 400 LOC diff budget (missed 6/6 slices with retroactive prose) | **ENFORCED** | `diff-budget` gate: `.github/workflows/diff_budget.yml` + `scripts/check_diff_budget.py` (#1944); required-check drift audit via `scripts/check_required_status_checks.py` (#1945) |
| 2 | Every automated-review finding reconciled before merge | **ENFORCED for contradictions; residual gap TRACKED** (pre-dated arc) | `scripts/check_ai_reconciliation_live.py` fails open-bot-threads-vs-all-clear contradictions; `scripts/audit_ai_reconciliation.py` classifies the body record when present. Gap: a finding resolved in the GitHub UI with no body record passes both (the live gate returns OK on zero open threads before reading the body, and `scripts/local_pr_review.sh` runs the body audit without `--require`) -- logged on #1942 |
| 3 | Plan doc shape / files-touched / size drift / reviewer rules | **ENFORCED** (pre-dated arc) | pre-push-audit suite (`scripts/local_pr_review.sh` bundle) |
| 4 | Producer/consumer shape fidelity: consumer tests must use fixtures the real producer emits (S6 mass-deletion P1: bare ids vs Reddit API fullnames stayed green) | **ENFORCED for the converted suites; opt-in elsewhere -- residual TRACKED** | `tests/atlas_reddit_fixtures.py` factory + `tests/test_atlas_reddit_fixture_fidelity.py` (#1947): the purge/tracker suites seed through the REAL producer. Gap: nothing forbids a new test from seeding the store directly and bypassing the factory -- logged on #1942 (candidate: static probe over atlas_reddit test seeding) |
| 5 | Adversarial-negative fixtures for every boundary (exact bool/int/finite-float, hostile strings) -- reviewers caught 8/8 injection holes in S3 alone | **ENFORCED for negatives PRESENCE; adversarial QUALITY review-owned** | `NO_RAISES_TESTS` + `NO_TEST_FILE` in `SENSITIVE_ZERO_TOLERANCE` (`scripts/maturity_sweep.py`): a sensitive-glob module whose tests never assert a raise (AST-verified against the real assertion APIs in asserting positions) -- or that has no tests, or only a stub test file -- fails the ratchet outright. The gate proves negatives EXIST; whether they probe the exact boundaries the lesson names (bool/int/finite-float, hostile strings) is the review contract's job (R2) |
| 6 | Recurrence of retired failure modes (plan-weakening, test-weakening, scope drift, symptom patching) | **INVESTIGATION** | `docs/retired_failure_mode_detection_layer.md` (merged via #1943); no architecture chosen yet |
| 7 | Gate scripts execute from the PR merge ref (self-passing edit risk) -- applies to EVERY gate above | **ENFORCED for the PR meta-gates; maturity sweeps TRACKED** | Meta-gates (diff-budget, body contract, reconciliation, pre-push audit) restore gate scripts from the base ref (#1949, #1950; closed #1944 waiver 18). Gap: the maturity-sweep workflows still run `scripts/maturity_sweep.py` from the merge ref, so a PR can weaken that gate and self-pass -- logged on #1942 |

## Practices (real, but not mechanically enforceable)

These recurred across the arc as judgment patterns. They stay here as
review vocabulary, not as promises:

- **Root-cause over cited-symptom**: the arc repeatedly fixed upstream of
  the reviewer's suggestion (S2 stale-replay invariant; S5 schema-v2
  sticky flag) -- and rejected symptom-site patches on its own prior waves.
- **Shared validation across entry paths**: a knob that enters via
  settings, env, CLI, SQL, or adapter needs one centralized bound (the S4
  `MAX_*` two-entry-path lesson; recurred within-slice twice before it
  generalized).
- **Boundary-wide error contracts**: wrap the whole external boundary
  (open/connect/auth/fetch/write), not the first line that failed locally.
- **Persist lifecycle state**: never infer durable state from absence in a
  limited API window (S5 eviction-vs-inactivity; S6 tombstones).
- **Narrow conflict handling**: suppress only the exact replay conflict;
  let integrity violations raise (S2 P1).
- **Threat-model the gate before hardening it**: #1944 took 5 review waves
  to articulate that its adversary was honest-but-hasty authors -- after
  which 3 findings were correctly waived instead of fixed. Write the
  threat model in the PR body first.

## Deferrals: acceptable vs not

- The Reddit arc's deferrals (scheduling, LLM fit, live smokes, unread
  badges) were sound: each slice still proved what it claimed.
- The #1936 class is NOT repeatable: never waive strict typing, per-run
  output isolation, or raw error redaction for any live or tenant-adjacent
  runbook. Those waivers are tracked in #1942 and gate any live run.

## Positive results worth keeping

- Context and product objective survived compaction across all six slices
  (cross-slice references stayed accurate; S5 proactively closed a gap S2
  left open). The compact-instructions baton in CLAUDE.md works.
- Severity honesty: reviewer labels were reported without inflation.
- The override mechanism works as designed: #1944 shipped its initial
  slice at exactly 400/400, then carried its review-fix growth on a real,
  logged override -- friction plus record, not a wall.

## Adding to this index

New lesson -> new row with one of: ENFORCED (name the gate), OPEN (name
the planned slice), TRACKED (name the waiver/issue), or INVESTIGATION
(name the brief). A lesson without an enforcement path goes to the issue
tracker, not this file.
