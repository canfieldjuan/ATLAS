# PR — Content-Ops Claims Map (operating-model slice 3)

## Why this slice exists

`docs/content_ops_operating_model.md` makes the **claims map** the highest-value piece of
the stage-3B (claims/compliance) gate: every claim a draft makes is mapped to the
messaging registry and status-checked, so marketing can't quietly invent a lawsuit
("Save 30% on all plans" vs the approved "Save up to 30% on eligible annual plans"). This
slice lands the deterministic core of that map; the part that *requires* an LLM (extracting
the claims from prose) is deferred so this stays additive and testable, matching slices 1-2.

Diff total sits just over the 400-LOC soft cap (see *Estimated diff size*); the excess is
test surface, not product — the module itself is ~150 LOC. The shippable surface stays
small.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Vertical slice

New owned module `extracted_content_pipeline/claims_map.py` + tests. Pure value types +
pure functions; no I/O, no Atlas imports, no DB, no LLM:

- `ClaimStatus` (StrEnum) — MATCH / MISMATCH / UNREGISTERED / EXPIRED.
- `RegistryClaim` (frozen dataclass) — a messaging-registry entry: `id`, `approved_wording`,
  `risk_tier` (slice 1 `RiskTier`), `expiration` (date or None).
- `ExtractedClaim` (frozen dataclass) — one claim as an extractor would emit it: `text`,
  `location`, candidate `registry_id` (the extractor that produces these from prose is the
  deferred LLM step).
- `MappedClaim` (frozen dataclass) — a claims-map row: text, location, registry_id,
  approved_wording, status, risk_tier.
- `map_claim(...)` / `build_claims_map(...)` — assign status against a registry as of a
  date; `blocking_claims(...)` / `is_clear(...)` — the gate signal (MISMATCH or EXPIRED
  blocks).


### Files touched

- `extracted_content_pipeline/claims_map.py`
- `extracted_content_pipeline/manifest.json`
- `plans/PR-Content-Ops-Claims-Map.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_extracted_content_claims_map.py`

## Mechanism

Status is decided by normalized comparison (casefold + collapsed whitespace): an unknown /
missing `registry_id` is `UNREGISTERED`; a matched entry past its `expiration` (relative to
a caller-supplied `as_of`) is `EXPIRED`; otherwise wording equal to the approved wording is
`MATCH`, else `MISMATCH`. `blocking_claims` returns the rows whose status is MISMATCH or
EXPIRED. Same conventions as slices 1-2 (`StrEnum` with the 3.10 fallback, frozen
dataclasses, `None`/non-`str` tolerated, not raised).

## Intentional

- Additive only — no existing code path changes; the stage-3B gate that *calls* this is a
  later slice.
- Matching is normalized-exact, not fuzzy/semantic. A real "close paraphrase" still reads
  as MISMATCH, which is the safe default for a compliance gate; fuzzy/semantic similarity
  is deferred.
- `UNREGISTERED` is reported but not itself blocking — surfacing an unknown claim is a
  human-review signal, not an automatic block (the doc routes that to the editor).

## Deferred

- The LLM extractor (prose -> `ExtractedClaim` list with candidate registry ids) — needs
  the host LLM factory; later slice.
- Wiring the claims map into a stage-3B gate / the Content-PR coverage matrix (slice 4).
- Fuzzy/semantic wording match; registry persistence schema. Calibration library +
  adversarial pass (slice 5). Multi-model disagreement orchestration stays parked.

## Verification

- pytest `tests/test_extracted_content_claims_map.py`
- `scripts/check_ascii_python.sh` (run via bash) -- ASCII gate
- `scripts/check_extracted_imports.py` (run via python3) -- import structure
- `scripts/audit_extracted_pipeline_ci_enrollment.py` -- new test enrolled
- `scripts/audit_extracted_standalone.py` (--fail-on-debt) -- no Atlas runtime imports
- `scripts/audit_pr_session_drift.py` + `scripts/sync_pr_plan.py` -- plan shape/drift

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/claims_map.py` | 155 |
| `extracted_content_pipeline/manifest.json` | 3 |
| `plans/PR-Content-Ops-Claims-Map.md` | 90 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_extracted_content_claims_map.py` | 169 |
| **Total** | **418** |
