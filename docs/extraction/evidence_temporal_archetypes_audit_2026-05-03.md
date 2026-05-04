# Reasoning Core Consolidation Audit: Evidence, Temporal, Archetypes

Date: 2026-05-03

## Executive Decision

Consolidate four files from `atlas_brain/reasoning/` into
`extracted_reasoning_core/` and replace the three drifted forks in
`extracted_content_pipeline/reasoning/` with re-export wrappers.
Implements three of the `NotImplementedError` stubs in
`extracted_reasoning_core/api.py`: `score_archetypes`,
`evaluate_evidence`, `build_temporal_evidence`.

This is PR 3 in the follow-up sequence defined by PR #79's
`reasoning_boundary_audit_2026-05-03.md`. It unblocks PR 4
(semantic-cache split), which depends on stable `EvidenceItem` and
evidence-shape semantics.

## Verified Current State

| File | atlas_brain LOC | content_pipeline LOC | LOC delta | Nature of divergence | atlas_brain deps |
| --- | ---: | ---: | ---: | --- | --- |
| `archetypes.py` | 592 | 590 | -2 | minor: docstrings, `frozen=True`, formatting; description-string rewrites for ~6 archetypes (user-facing copy) | 0 (stdlib + dataclasses only) |
| `evidence_engine.py` | 548 | 338 | -210 | **different implementation, not drift**; see Drift Reality Check | yaml + stdlib |
| `evidence_map.yaml` | 284 | not present | n/a | content_pipeline never carried the YAML; loads rules via custom `_load_rules` helper | yaml data file |
| `temporal.py` | 490 | 466 | -24 | `frozen=True` + 2 new defensive helpers (`_numeric_value`, `_row_get`); `MIN_DAYS_FOR_PERCENTILES` 7 -> 3 | stdlib only |
| **total** | **1,914** | n/a | n/a | mixed: minor + structural + parameter | none |

Zero `atlas_brain.*` imports in any of the four files. Pure
consolidation candidates by dependency graph; not all by behavior.

## Drift Reality Check

The first measurement in this audit's initial draft used
`diff -q | grep -cE '^[+-]'` and reported drift as "20 / 60 / 48
lines" -- a hunk-marker count, not a real LOC delta. A direct LOC
comparison surfaced a much larger divergence on `evidence_engine.py`
(548 vs 338) that the original draft did not anticipate. This audit
records the corrected measurement and adjusts the consolidation plan
accordingly.

What `extracted_content_pipeline/reasoning/evidence_engine.py`
actually contains:

- Same dataclasses (`ConclusionResult`, `SuppressionResult`).
- A re-implementation of `EvidenceEngine` that loads rules via a
  custom `_load_rules(map_path)` helper, not from `evidence_map.yaml`.
- A renamed plural method
  `evaluate_conclusions(...) -> list[ConclusionResult]` (atlas has
  the singular `evaluate_conclusion(...) -> ConclusionResult` plus an
  internal collection helper).
- **Drops the entire per-review enrichment surface** that atlas owns:
  `compute_urgency`, `override_pain`, the recommend / price regex
  pre-compilation, and `_check_condition_simple`. ~210 LOC of
  per-review business logic does not exist in content_pipeline.

Implication: this is the case the original Risks section warned
about. The "fork" is a fundamentally smaller engine focused on the
conclusions + suppression surface; the per-review enrichment lives
only in atlas and is therefore atlas-flavored business logic, not
shared-core machinery.

Existing tests already in `tests/`:

- `test_extracted_reasoning_archetypes.py`
- `test_extracted_reasoning_evidence_engine.py`
- `test_extracted_reasoning_temporal.py`
- `test_b2b_phase2_subject_gate.py`, `test_b2b_phase3_polarity_gate.py` (consumers)
- `test_archetype_propagation.py`, `test_evidence_engine.py`,
  `test_reasoning_market_pulse.py`, `test_reasoning_temporal.py`,
  `test_reasoning_live.py` (atlas-side; not migrated by this PR)

## Type-Contract Collisions (require resolution)

PR #79's audit pinned `ArchetypeMatch` and `TemporalEvidence` as
supporting types. The atlas_brain canonical files have classes by the
same names but different shapes. Reconciliation is required.

### Collision 1: `ArchetypeMatch`

| Field | atlas_brain (`archetypes.py:59`) | PR #79 contract (`types.py`) | Resolution |
| --- | --- | --- | --- |
| identifier | `archetype: str` | `archetype_id: str` | rename to `archetype_id` |
| display | (derived) | `label: str` | new field; populate from `ArchetypeProfile.name` |
| score | `score: float` | `score: float` | match |
| hits | `matched_signals: list[str]` | `evidence_hits: Sequence[str]` | rename + tuple |
| misses | `missing_signals: list[str]` | `missing_evidence: Sequence[str]` | rename + tuple |
| risk | `risk_level: str` | `risk_label: str` | rename |

**Decision:** atlas_brain's class is renamed `_ArchetypeMatchInternal`;
the existing PR #79 public dataclass remains the contract.
`score_archetypes` returns the public version after field translation.

### Collision 2: `TemporalEvidence`

PR #79 used `Mapping[str, Any]` defaults -- coarse and lossy.
atlas_brain has a rich typed shape with four sub-dataclasses
(`VendorVelocity`, `LongTermTrend`, `CategoryPercentile`,
`AnomalyScore`).

**Decision:** Amend PR #79's contract to use the rich shape. Coarse
`Mapping[str, Any]` is too lossy for products that render
velocity/trend charts. Promote the four sub-dataclasses to public types
(frozen, immutable). Update `extracted_reasoning_core/types.py` and the
audit doc in PR #79 in the same code-PR commit.

This is the second amendment to PR #79 (the first was the
`reasoning_input` rename plus `tier: ReasoningDepth`). Worth flagging
that PR #79's contract is being calibrated against ground truth as we
extract. Frequency is currently acceptable; if it happens again in
PR 4 / cache split, that is a signal the audit was under-specified.

### Collision 3: `evaluate_evidence` shape

PR #79 stubbed
`evaluate_evidence(evidence, *, policy) -> EvidenceDecision`.
atlas_brain's `EvidenceEngine` has two finer-grained methods:
`evaluate_conclusion(...) -> ConclusionResult` and
`evaluate_suppression(...) -> SuppressionResult`.

**Decision:** Keep both surfaces.

- Public helper `evaluate_evidence(...) -> EvidenceDecision` for the
  common "is this evidence sufficient?" case.
- `EvidenceEngine` exposed as a public class with `evaluate_conclusion`
  and `evaluate_suppression` methods for callers that need the richer
  outcomes.
- `ConclusionResult` and `SuppressionResult` become public supporting
  types alongside `EvidenceDecision`.

Add `ConclusionResult`, `SuppressionResult` to PR #79's
supporting-types table in the same amendment.

### Collision 4: `evaluate_conclusion` vs `evaluate_conclusions`

content_pipeline's fork uses a plural method
`evaluate_conclusions(...) -> list[ConclusionResult]` that loops the
`_conclusions` rule-set internally. atlas exposes the singular
`evaluate_conclusion(rule_id, evidence) -> ConclusionResult` and lets
callers iterate.

**Decision:** Keep both. Singular is the lower-level public method;
plural is a convenience wrapper that iterates. The plural shape is
genuinely useful (content_pipeline already wrote it that way) and
the cost of carrying both is one short delegation method.

## Module Disposition

| File | Disposition | Notes |
| --- | --- | --- |
| `archetypes.py` | Move atlas as canonical to `extracted_reasoning_core/archetypes.py`; carry `frozen=True` from content_pipeline; archetype description-string rewrites flagged for a future `content_review_pack` | Add public-shape `ArchetypeMatch` adapter; keep `ArchetypeProfile`, `SignalRule`, `ARCHETYPES` catalog as core internals |
| `evidence_engine.py` | **Slim core + enrichment pack split.** Move conclusions + suppression surface to `extracted_reasoning_core/evidence_engine.py` (the surface content_pipeline currently exposes). Move per-review enrichment (`compute_urgency`, `override_pain`, recommend / price regex pre-compilation, `_check_condition_simple`) to `atlas_brain/reasoning/review_enrichment.py` (renamed) AND scaffold a `content_review_pack` that consumes it. Keep YAML loader in core; rule shape unchanged. | Class stays public; both `evaluate_conclusion` and `evaluate_conclusions` exposed; per-review methods removed from core public API |
| `evidence_map.yaml` | Move verbatim to `extracted_reasoning_core/evidence_map.yaml`; ships as default policy data | Products override via `get_evidence_engine(map_path=...)` or by injecting their own `EvidencePolicy` |
| `temporal.py` | Move atlas as canonical to `extracted_reasoning_core/temporal.py`; **carry forward content_pipeline's `_numeric_value` / `_row_get` defensive helpers** (real engineering value, not drift); **parameterize `MIN_DAYS_FOR_PERCENTILES`** via `TemporalEngine` constructor (atlas=7, content_pipeline=3 are both valid for their use cases) | `TemporalEngine` class stays public; promote `VendorVelocity`, `LongTermTrend`, `CategoryPercentile`, `AnomalyScore` to public types |

## evidence_map.yaml Placement Decision

The YAML is loaded relative to `evidence_engine.py`
(`Path(__file__).parent / "evidence_map.yaml"`). When
`evidence_engine.py` moves to `extracted_reasoning_core/`, the YAML
must move alongside or the default load breaks.

**Decision:** Bundle `evidence_map.yaml` as default policy data shipped
with core. Document that products may override via
`get_evidence_engine(map_path=...)`. The YAML is the v2 evidence-rule
schema (enrichment, conclusions, suppression, confidence_tiers);
product-specific rule sets become product-pack assets in a future PR.

## Drift-Forward Plan (per file)

Triage based on the corrected measurements above:

**`archetypes.py` (-2 LOC; ~30 differing characters)**

- (b) forward port: docstrings, `frozen=True`, formatting -- land in
  core unchanged from atlas
- (a) content-pipeline-specific: archetype description-string
  rewrites and falsification template wording for ~6 archetypes; this
  is user-facing copy, not engine behavior. Flag for the future
  `content_review_pack` (catalog override). Core ships atlas's
  technical descriptions.

**`evidence_engine.py` (-210 LOC; structural divergence)**

This file does NOT fit the simple (a)/(b)/(c) drift model. The
content_pipeline fork is a different implementation, not a drifted
copy. Decision is in Module Disposition: split into a slim conclusions
+ suppression core (matches content_pipeline's public shape) plus a
`review_enrichment` module that becomes part of an atlas-flavored
`content_review_pack`. The code PR delivers the slim core and the
extracted review_enrichment module; the pack scaffolding is a separate
follow-up PR.

**`temporal.py` (-24 LOC; mostly real)**

- (b) forward port: `frozen=True`, formatting, multi-line
  comprehensions -- land in core
- new functionality from content_pipeline to ADOPT (forward port from
  the fork, not from atlas): `_numeric_value` and `_row_get` defensive
  helpers handle messy input data; this is engineering value worth
  carrying into core
- (parameterize) `MIN_DAYS_FOR_PERCENTILES` -- atlas=7,
  content_pipeline=3; constructor parameter with default=7 (matches
  atlas) and content_pipeline overrides at instantiation

The code PR's commit message lists each non-trivial hunk's
classification. Future review cannot ask "where did the
content-pipeline behavior go" without an answer in the commit.

## Test Migration

Existing tests get renamed and redirected, not duplicated:

| Current test file | Action |
| --- | --- |
| `tests/test_extracted_reasoning_archetypes.py` | Rename to `tests/test_extracted_reasoning_core_archetypes.py`; switch imports to `extracted_reasoning_core.archetypes` |
| `tests/test_extracted_reasoning_evidence_engine.py` | Rename to `tests/test_extracted_reasoning_core_evidence_engine.py`; redirect imports |
| `tests/test_extracted_reasoning_temporal.py` | Rename to `tests/test_extracted_reasoning_core_temporal.py`; redirect imports |

**PR-C1k amendment (2026-05-04):** the "rename to `*_core_*` + redirect imports" line above turned out to be wrong for two of the three files, and was partially superseded during PR-C1h / PR-C1i / PR-C1j:

  - PR-C1h / PR-C1i / PR-C1j each shipped a **new** `tests/test_extracted_reasoning_core_*.py` file containing **unit-level** tests of the canonical core (constants, frozen invariants, single-rule `_check_requirement` operators, helper coercion). Those are now the authoritative core tests.
  - The original `tests/test_extracted_reasoning_*.py` files turned out to be **integration-style scenario tests** (pricing_crisis met-and-amplified, feature_gap winning when capability signals dominate, integration_break recognition from nested weakness_evidence text). They are *not* redundant with the unit tests.
  - For evidence_engine specifically, the wrapper test exercises the content-pipeline-specific `_DEFAULT_RULES` catalog (`pricing_crisis`, `losing_market_share`, `active_churn_wave`, `support_quality_risk`). Redirecting its imports to `extracted_reasoning_core.evidence_engine` would run the same assertions against core's `evidence_map.yaml`, which carries a different conclusion vocabulary -- the assertions would all fail.

PR-C1k therefore renames the wrapper-tests to a clearer prefix (`tests/test_extracted_content_pipeline_reasoning_*.py`) and **keeps imports unchanged** (still pointing at `extracted_content_pipeline.reasoning.*`). The unit-level core tests stay at `tests/test_extracted_reasoning_core_*.py`. Both layers are kept; coverage is complementary, not duplicative.

`tests/test_extracted_reasoning_core_api.py` updated: drop the three
now-implemented stubs from
`test_stubbed_public_entry_points_fail_closed_until_consolidated` (the
forcing-function pattern from PR #80).

Atlas-side tests (`test_archetype_propagation.py`,
`test_evidence_engine.py`, `test_reasoning_temporal.py`,
`test_reasoning_market_pulse.py`, `test_reasoning_live.py`,
`test_b2b_phase{2,3}_*.py`) stay where they are; they consume
`atlas_brain.reasoning.*` paths which PR 7 (Product Migration) will
redirect.

## Code-PR Scope (single PR, follows this audit)

Files touched by the code PR:

```
NEW:    extracted_reasoning_core/archetypes.py
NEW:    extracted_reasoning_core/evidence_engine.py
        (slim conclusions + suppression surface;
         per-review enrichment NOT included)
NEW:    extracted_reasoning_core/evidence_map.yaml
NEW:    extracted_reasoning_core/temporal.py
        (with _numeric_value / _row_get helpers carried
         from content_pipeline fork; MIN_DAYS_FOR_PERCENTILES
         parameterized)
NEW:    atlas_brain/reasoning/review_enrichment.py
        (extracted from atlas evidence_engine: compute_urgency,
         override_pain, recommend/price regex pre-compilation,
         _check_condition_simple. Atlas-internal until the
         content_review_pack PR follows up.)
EDIT:   atlas_brain/reasoning/evidence_engine.py
        (delete the per-review enrichment surface; remaining
         conclusions + suppression behavior delegates to or
         imports from extracted_reasoning_core)
EDIT:   extracted_reasoning_core/api.py
        (impl 3 stubs; export new public types)
EDIT:   extracted_reasoning_core/types.py
        (replace TemporalEvidence; add 4 sub-types,
         ConclusionResult, SuppressionResult)
EDIT:   extracted_content_pipeline/reasoning/archetypes.py        (-> wrapper)
EDIT:   extracted_content_pipeline/reasoning/evidence_engine.py   (-> wrapper)
EDIT:   extracted_content_pipeline/reasoning/temporal.py          (-> wrapper)
EDIT:   tests/test_extracted_reasoning_core_api.py
        (drop now-implemented stubs)
RENAME: tests/test_extracted_reasoning_*.py
        -> tests/test_extracted_reasoning_core_*.py
EDIT:   docs/extraction/reasoning_boundary_audit_2026-05-03.md
        (PR #79 contract amendment)
```

NOT touched by the code PR: `extracted_competitive_intelligence/*`
(no archetypes, evidence, or temporal forks there), any other
product. The code PR DOES touch `atlas_brain/reasoning/` for the
slim-core split (this is the first PR in the sequence to do so;
subsequent PRs in PR #79's sequence continue the atlas-side
migration). PR 7 / Product Migration still owns the broader
atlas-to-core wedge_registry alignment.

**Follow-up PR (out of scope for this code PR):** scaffold a
`content_review_pack` that consumes
`atlas_brain.reasoning.review_enrichment` (or a future
`extracted_reasoning_packs.review_enrichment`) and exposes the
per-review enrichment surface to content_pipeline through a Protocol.

## Risks

- **Soft dependency on PR #80 merging.** The four implementation files
  add to an `api.py` that PR #80 just established. If #80 changes
  during review, the code PR rebases. Acceptable.
- **PR #79 contract amendment in same PR.** This is the second
  contract amendment after the `reasoning_input` rename plus `tier`
  typing. Frequency is currently acceptable; if PR 4 produces a third,
  that is a signal the original audit was under-specified.
- **First atlas-side edits in this sequence.** Earlier PRs in the
  reasoning extraction (PR #79, #80) only added or modified extracted
  files. This code PR is the first to touch `atlas_brain/reasoning/`
  (creating `review_enrichment.py`, slimming `evidence_engine.py`).
  Atlas tests (`test_evidence_engine.py`,
  `test_b2b_phase{2,3}_*.py`) must pass after the slim. If they
  fail, the slim plan needs revision before the code PR opens.
- **Slim-core decision is load-bearing.** If review-enrichment ever
  needs to live in core (because a non-atlas product needs per-review
  enrichment), the split has to be revisited. Today only atlas needs
  it, so the split is safe. Watch for the second consumer.
- **Audit measurement was wrong on first pass.** Using
  `diff -q | grep -cE` instead of LOC comparison undercounted
  evidence_engine drift by an order of magnitude. Future audit docs
  should report `wc -l` deltas alongside structural-divergence notes,
  not hunk-marker counts.

## Acceptance Criteria (for the code PR)

- 3 stubs in `extracted_reasoning_core/api.py` no longer raise
  `NotImplementedError`
- 3 forks in `extracted_content_pipeline/reasoning/` are now re-export
  wrappers (`<= 30` lines each, byte-identical pattern to current
  `wedge_registry.py` wrappers)
- `extracted_reasoning_core/evidence_engine.py` exposes only the
  conclusions + suppression surface; no per-review enrichment methods
  in core public API
- `atlas_brain/reasoning/review_enrichment.py` exists with the
  per-review enrichment surface; `atlas_brain/reasoning/evidence_engine.py`
  no longer contains those methods
- All existing `test_extracted_reasoning_*` tests pass after rename and
  redirect
- All atlas tests touching `evidence_engine` (`test_evidence_engine.py`,
  `test_b2b_phase2_subject_gate.py`, `test_b2b_phase3_polarity_gate.py`)
  still pass after the slim
- `tests/test_extracted_reasoning_core_api.py::test_stubbed_public_entry_points_fail_closed_until_consolidated`
  updated to remove the 3 now-implemented entries
- `extracted_reasoning_core/types.py` exports the rich
  `TemporalEvidence` plus 4 sub-types plus `ConclusionResult` plus
  `SuppressionResult`
- `TemporalEngine` constructor accepts `min_days_for_percentiles`
  parameter (default=7)
- `TemporalEngine` carries `_numeric_value` and `_row_get` defensive
  helpers from the content_pipeline fork
- PR #79 audit doc amended in the same commit to reflect the contract
  amendment
- Drift-forward triage classifications appear in the commit message
  body for `archetypes.py` and `temporal.py`; `evidence_engine.py`
  notes reference the slim-core split documented in this audit
- `bash scripts/run_extracted_*_checks.sh` all pass
- `bash scripts/run_extracted_competitive_intelligence_checks.sh` passes
  (no behavioral changes there, but verifies the wedge_registry
  re-export wrapper still resolves)
