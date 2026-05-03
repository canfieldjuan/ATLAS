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

| File | atlas_brain LOC | content_pipeline fork | Drift (diff lines) | atlas_brain deps |
| --- | ---: | --- | ---: | --- |
| `archetypes.py` | 592 | yes (drifted) | ~60 | 0 (stdlib + dataclasses only) |
| `evidence_engine.py` | 548 | yes (drifted) | ~20 | yaml + stdlib |
| `evidence_map.yaml` | 284 | not present | n/a | yaml data file |
| `temporal.py` | 490 | yes (drifted) | ~48 | stdlib only |
| **total** | **1,914** | 3 of 4 forked | ~128 lines | none |

Zero `atlas_brain.*` imports in any of the four files. Pure
consolidation candidates.

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

## Module Disposition

| File | Disposition | Notes |
| --- | --- | --- |
| `archetypes.py` | Move verbatim to `extracted_reasoning_core/archetypes.py`; reconcile content_pipeline drift forward | Add public-shape `ArchetypeMatch` adapter; keep `ArchetypeProfile`, `SignalRule`, `ARCHETYPES` catalog as core internals |
| `evidence_engine.py` | Move verbatim to `extracted_reasoning_core/evidence_engine.py`; reconcile content_pipeline drift forward | Class stays public; `get_evidence_engine` factory stays public; `_DEFAULT_MAP_PATH` recomputed for new location |
| `evidence_map.yaml` | Move verbatim to `extracted_reasoning_core/evidence_map.yaml`; ships as default policy data | Products override via `get_evidence_engine(map_path=...)` or by injecting their own `EvidencePolicy` |
| `temporal.py` | Move verbatim to `extracted_reasoning_core/temporal.py`; reconcile content_pipeline drift forward | `TemporalEngine` class stays public; promote `VendorVelocity`, `LongTermTrend`, `CategoryPercentile`, `AnomalyScore` to public types |

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

Three forks differ from atlas_brain by a combined ~128 lines. Each
fork's drift must be classified before consolidation:

1. **Per-file drift triage** during the code PR: run
   `diff atlas_brain/reasoning/<file> extracted_content_pipeline/reasoning/<file>`,
   categorize each hunk as:

   - **(a)** content-pipeline-specific behavior; flag for extraction
     into a content reasoning pack (deferred PR)
   - **(b)** forward port to canonical; land in core
   - **(c)** equivalent rewrite; discard, use canonical
2. The code PR's commit message lists each hunk's category. Future
   review cannot ask "where did the content-pipeline behavior go"
   without an answer in the commit.

If most hunks land in category (a), the consolidation may need to wait
for a content reasoning pack to exist; triage during the code PR will
tell.

## Test Migration

Existing tests get renamed and redirected, not duplicated:

| Current test file | Action |
| --- | --- |
| `tests/test_extracted_reasoning_archetypes.py` | Rename to `tests/test_extracted_reasoning_core_archetypes.py`; switch imports to `extracted_reasoning_core.archetypes` |
| `tests/test_extracted_reasoning_evidence_engine.py` | Rename to `tests/test_extracted_reasoning_core_evidence_engine.py`; redirect imports |
| `tests/test_extracted_reasoning_temporal.py` | Rename to `tests/test_extracted_reasoning_core_temporal.py`; redirect imports |

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
NEW:    extracted_reasoning_core/evidence_map.yaml
NEW:    extracted_reasoning_core/temporal.py
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

NOT touched by the code PR: `atlas_brain/reasoning/*` (PR 7 / Product
Migration), `extracted_competitive_intelligence/*` (no archetypes,
evidence, or temporal forks there), any other product.

## Risks

- **Soft dependency on PR #80 merging.** The four implementation files
  add to an `api.py` that PR #80 just established. If #80 changes
  during review, the code PR rebases. Acceptable.
- **PR #79 contract amendment in same PR.** This is the second
  contract amendment after the `reasoning_input` rename plus `tier`
  typing. Frequency is currently fine; watch the pattern.
- **Drift-forward triage risk.** ~128 drifted lines across 3 files.
  If most hunks are category (a), the consolidation may need to wait
  for a content reasoning pack. Triage during the code PR will tell.

## Acceptance Criteria (for the code PR)

- 3 stubs in `extracted_reasoning_core/api.py` no longer raise
  `NotImplementedError`
- 3 forks in `extracted_content_pipeline/reasoning/` are now re-export
  wrappers (`<= 30` lines each, byte-identical pattern to current
  `wedge_registry.py` wrappers)
- All existing `test_extracted_reasoning_*` tests pass after rename and
  redirect
- `tests/test_extracted_reasoning_core_api.py::test_stubbed_public_entry_points_fail_closed_until_consolidated`
  updated to remove the 3 now-implemented entries
- `extracted_reasoning_core/types.py` exports the rich
  `TemporalEvidence` plus 4 sub-types plus `ConclusionResult` plus
  `SuppressionResult`
- PR #79 audit doc amended in the same commit to reflect the contract
  amendment
- Drift-forward triage table appears in the commit message body
- `bash scripts/run_extracted_*_checks.sh` all pass
