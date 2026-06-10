# PR-Deflection-Synthetic-Ticket-Generator

## Why this slice exists

The deflection hardening backlog (#1454-#1463) and the upcoming clustering
rework both need labeled ground truth: ticket datasets where the right
clusters, questions, and resolution-evidence counts are known in advance. Real
exports cannot provide that (unlabeled), and LLM-generated data cannot either
(non-reproducible, unlabeled, and off-thesis for a pipeline marketed as
deterministic). The existing fixture tool
(`scripts/build_deflection_messy_csv_fixtures.py`) only re-shapes rows that
already exist; it cannot create tickets from scratch.

This slice adds a deterministic, zero-LLM synthetic ticket generator: seeded
template expansion over a curated intent bank, emitting (a) a support-ticket
CSV shaped exactly like what ingestion consumes and (b) a ground-truth sidecar
naming the expected cluster for every ticket. Same seed, byte-identical
output. Messiness injectors map one-to-one onto filed brittleness issues so
the generator doubles as the regression harness for the fixes in flight.

Diff-budget note: the total runs over the 400 soft cap because the intent /
slot template banks are data, not logic; the executable surface is small and
the data is what makes the fixtures realistic. Splitting the injectors out
would orphan the regression-harness value this exists for.

## Scope (this PR)

Ownership lane: deflection/synthetic-ticket-fixtures
Slice phase: Robust testing

1. `scripts/build_synthetic_support_tickets.py`: seeded generator with a
   six-intent template bank, slot fills, per-intent resolution coverage,
   ground-truth sidecar, and messiness injector flags.
2. Tests proving seed determinism, ground-truth consistency, injector
   behavior, and a clean round-trip through
   `build_support_ticket_input_package` with zero warnings.

### Review Contract
- Acceptance criteria: same seed -> byte-identical CSV and sidecar; different
  seed -> different bytes; sidecar cluster assignments match emitted rows
  one-to-one; each injector flag produces its documented breakage; clean
  output round-trips through the ingestion package with no warnings and the
  full row count; tests run in CI.
- Affected surfaces: scripts / tests only. No app code, no DB, no network.
- Risk areas: none production-facing (test tooling); main risk is silent
  nondeterminism (e.g. set iteration, datetime.now) breaking golden tests.
- Reviewer rules triggered: R2 (failure-branch fixtures), R10
  (maintainability), R12 (CI actually runs the new test).

### Files touched
- `scripts/build_synthetic_support_tickets.py`
- `tests/test_build_synthetic_support_tickets.py`
- `plans/PR-Deflection-Synthetic-Ticket-Generator.md`

## Mechanism

A fixed-order intent bank (password reset, billing dispute, broken
integration, slow dashboard, refund request, data export), each intent
carrying subject templates, body templates, a `pain_category`, a canonical
question, and resolution templates for a subset of intents only, so the
resolution-evidence lane is exercised in both the has-answer and no-answer
directions. Slot banks (product, error code, weekday, company) fill the
templates through `random.Random(seed)`; iteration order is fixed tuples
everywhere, `created_at` derives from a fixed `--base-date` (default
2026-06-01) spread across `--window-days`, and nothing reads the clock, so a
seed fully determines every byte.

Output: a CSV with exactly the ingestion fields (`ticket_id`, `subject`,
`message`, `resolution_text`, `pain_category`, `created_at`, `company_name`)
plus a ground-truth JSON sidecar (ground_truth.json, a generated output, not
a repo file) recording seed, per-intent counts, the
ticket_id->intent map, and per-cluster expectations (size, pain_category,
canonical question, has_resolution). Injector flags rewrite the clean output:
`--encoding` (utf-16 / utf-8-sig, #1455), `--delimiter` (#1459),
`--html-bodies` (#1463), `--unmapped-body-column` (#1457), `--junk-rows`
(banner/short/blank rows), and `--no-labels` (empties `pain_category`, the
raw-untagged-export case that fragments exact-match clustering).

## Intentional
- Code templates, not an LLM: labels are knowable only because generation is
  mechanical. This is the point of the slice, and it matches the product's
  deterministic positioning.
- Injectors mirror filed issues one-to-one rather than inventing new
  messiness, so each fixture maps to an acceptance test for a known fix.
- `pain_category` is emitted by default and stripped via `--no-labels`,
  because the labeled and unlabeled cases test different pipeline paths.
- The sidecar lives next to the CSV, not inside it, so the CSV stays an
  honest simulation of a customer export.
- Resolution coverage is per-intent (always or never), keeping ground-truth
  assertions exact instead of probabilistic.

## Deferred
- Full Zendesk anatomy mode (status, priority, assignee, requester, tags,
  conversation threads) for demo-realistic exports and column-mapping torture
  tests.
- Additional intents and template variants; multi-language bodies.
- Lane B: creating real tickets in a Zendesk sandbox via the Tickets API.
- Wiring the generated fixtures into specific #1454-#1463 regression tests
  (each fix owns its own test slice).

Parked hardening: none.

## Verification
- `tests/test_build_synthetic_support_tickets.py`: same-seed byte equality
  across two runs; different-seed inequality; sidecar/CSV consistency;
  `--no-labels`, `--unmapped-body-column`, `--encoding utf-16`, `--junk-rows`
  each produce their documented effect; clean round-trip through
  `build_support_ticket_input_package` (zero warnings, full row count).
- `scripts/check_ascii_python.sh` clean; `scripts/local_pr_review.sh` green.

## Estimated diff size
| Area | Est LOC |
|---|---:|
| Plan | ~120 |
| Generator script | ~420 |
| Tests | ~160 |
| Total | ~700 |

Over the 400 soft cap; justified in "Why this slice exists" (template banks
are data-heavy; executable logic is small; splitting injectors orphans the
regression-harness purpose).
