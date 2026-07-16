# PR-Content-Factory-Contract-Hardening

## Why this slice exists

The Content Factory artifact contracts (`atlas_brain/schemas/content_factory.py`,
merged in #2116) deferred three edge-hardening items to #2120 under the 3-round
Codex cap, gated on "a consumer exists so the `extra='forbid'` flip can be validated
against real round-trips." That consumer now exists: the artifact store (#2121) and
stage runner (#2125) both validate and round-trip artifacts. This slice does the
deferred pass as one deliberate change: flip the iteration-phase `extra='allow'` to
`extra='forbid'` (terminally closing the schema-key leak class), make `EvidenceRow.id`
non-blank (a blank id cannot anchor a `Claim.source_id`, so it silently breaks the
no-orphan-claim invariant), and require an `EvidencePacket` to carry at least one
evidence row or one gap (so a truncated worker output cannot masquerade as the honest
"no evidence, logged gaps" result).

### Problem-derived contract

From the #2120 findings, a correct fix must:
- Reject any unmodeled key on the artifact contracts (`extra='forbid'`), which also
  makes a reserved `artifact_schema` duplicate of the version tag fail closed instead
  of riding through and being emitted alongside `schema` on dump.
- Reject a blank `EvidenceRow.id` (make it `NonEmptyStr`), matching the existing
  non-blank treatment of `quote`/`source_id`.
- Reject an `EvidencePacket` with neither evidence nor gaps, so an empty/truncated
  packet cannot pass as a legitimate empty result.

## Scope (this PR)

Ownership lane: content-factory
Slice phase: production hardening

The three contract closures plus their tests, and a one-line comment update in the
store where the flip changes why its explicit `artifact_schema` pre-check exists. No
runtime behavior beyond stricter validation; the contracts are still consumed only by
the store/runner already merged.

### Review Contract

- Acceptance criteria:
  - [ ] An artifact with any unmodeled key is rejected; a reserved `artifact_schema`
        key fails closed.
  - [ ] `EvidenceRow` with a blank/whitespace `id` is rejected; a non-blank id validates.
  - [ ] `EvidencePacket` with neither evidence nor gaps is rejected; one with gaps (or
        evidence) validates.
  - [ ] The store/runner round-trip tests still pass under the stricter contracts.
- Reachability proof: consumed by `content_factory_store.write_artifact` /
  `content_factory_runner.run_stage` (both merged); proof is the three test files.
- Affected surfaces: the contracts module, one store comment, and the schema test file.
- Risk areas: the `extra='forbid'` flip (validated against the store/runner round-trip
  fixtures); the no-orphan-claim invariant (blank id); honest-empty-packet invariant.
- Reviewer rules triggered: R14. The change tightens guard/validator admission rules on
  the artifact contracts.

### Files touched

- `atlas_brain/schemas/content_factory.py`
- `atlas_brain/services/content_factory_store.py`
- `plans/PR-Content-Factory-Contract-Hardening.md`
- `tests/test_content_factory_schemas.py`

## Mechanism

`_BASE_CONFIG` flips to `extra='forbid'` (shared by every artifact model, so extra-key
leakage is closed at every nesting level, not only the five top-level artifacts).
`EvidenceRow.id` becomes `NonEmptyStr` (strip + min_length=1). `EvidencePacket` gains an
after-validator requiring `evidence` or `gaps` non-empty. The store's `artifact_schema`
pre-check comment is updated to note the contracts' `extra='forbid'` now also rejects
that key (the explicit pre-check is kept for its specific error message).

## Intentional

- The flip is applied to the shared `_BASE_CONFIG` (all models, including nested
  `EvidenceRow`/`Claim`/`StageEntry`/`Approval`), not only the five top-level artifacts
  the issue names, so no unmodeled key leaks at any level -- a consistent fail-closed
  posture matching the followup_workflow contract.
- The store's explicit `artifact_schema` pre-check is retained (not removed as now-
  redundant): it fires before validation with a specific `ArtifactStoreError` message.

## Deferred

- None. All three #2120 items are closed here.

## Verification

```
python -m pytest tests/test_content_factory_schemas.py tests/test_content_factory_store.py tests/test_content_factory_runner.py -q
```
76 tests pass: an unmodeled/`artifact_schema` key is rejected; a blank `EvidenceRow.id`
is rejected while a real id validates; an `EvidencePacket` with neither evidence nor gaps
is rejected while one with gaps validates; the store/runner round-trip fixtures still pass.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/schemas/content_factory.py` | 33 |
| `atlas_brain/services/content_factory_store.py` | 4 |
| `plans/PR-Content-Factory-Contract-Hardening.md` | 104 |
| `tests/test_content_factory_schemas.py` | 28 |
| **Total** | **169** |
