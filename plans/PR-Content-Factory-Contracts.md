# PR-Content-Factory-Contracts

## Why this slice exists

The Local Content Factory (epic #2109) writes one JSON artifact per pipeline
stage (brief, evidence, draft, audit, manifest). The Phase 1.4 end-to-end run
proved the plumbing but the artifacts are unvalidated free-form JSON: a malformed
stage output (the Editor's raw tool-call blob in that run) is only caught by
eyeball. This slice adds versioned Pydantic v2 contracts for the five artifact
shapes so any stage output can be validated deterministically, independent of
which local model produced it. It is the foundation for the Action function
(Phase 2.2) that writes artifacts and the Filter (Phase 4.2) that enforces them.

### Problem-derived contract

A correct fix must:
- Define the five artifact shapes (ContentBrief, EvidencePacket, DraftArtifact,
  EditorialAudit, ArtifactManifest) as versioned models whose fields match the
  shapes the Phase 1.4 workers actually emit.
- Encode the one load-bearing invariant the pipeline depends on: an evidence row
  is inadmissible without a `source_id` (and a `quote`) -- this is what stops
  "writer invents research".
- Be model-agnostic: no dependency on any specific LLM or on a chosen default.
- Follow the `atlas_brain/schemas/` house style (`extra='allow'`,
  `schema_version`, JSON key `schema` for the type tag).
- Add nothing to a runtime path yet -- contracts and tests only.

## Scope (this PR)

Ownership lane: content-factory
Slice phase: vertical slice

Lane: Content Factory (parallel feature, epic #2109), arc Phase 2.1 (structured
contracts). Intent: add the artifact contracts + tests, no runtime wiring.

### Files touched
- `atlas_brain/schemas/content_factory.py`
- `tests/test_content_factory_schemas.py`
- `plans/PR-Content-Factory-Contracts.md`

Max files: 3

## Mechanism

Five Pydantic v2 `BaseModel`s (plus nested `EvidenceRow`, `Claim`,
`CopyVerification`, `StageEntry`, `Approval`), each with
`ConfigDict(extra='allow', populate_by_name=True)` and `schema_version=1`. The
artifact type tag is stored under the JSON key `schema` via an alias
(`artifact_schema`) to avoid shadowing `BaseModel.schema`. A `model_for(data)`
helper dispatches a raw artifact dict to its contract class by the `schema` tag.
Required fields encode the invariants: `EvidenceRow` requires `quote` and
`source_id`; `DraftArtifact` requires `body_markdown`; `ContentBrief` requires
`project_id` and `request_raw`. Tests use fixtures shaped like the real Phase 1.4
run, including the actual empty-evidence packet.

## Intentional

- `extra='allow'` (not `forbid`): lets artifacts written before a field was added
  round-trip while the pipeline iterates; a later slice flips to `forbid` once the
  shapes are stable. Matches the campaigns.py precedent.
- Contracts only, no runtime wiring: consumers (Action fn, Filter) are separate
  slices, so this can land and be reviewed in isolation.
- Model-agnostic by construction: no default model is referenced, so the pending
  model-selection decision does not block or bias this slice.

## Deferred

- JSON schema export to `docs/schemas/content_factory_*.json` -> follow-up slice
  (no external consumer needs it yet).
- The OWUI Action function that writes artifacts -> Phase 2.2.
- The Filter that enforces these contracts and fails closed -> Phase 4.2.
- Flipping `extra='allow'` to `extra='forbid'` after a soak window.

## Verification

```
python -m pytest tests/test_content_factory_schemas.py -q
```
13 tests pass: round-trip preserves the `schema` tag for brief/evidence/draft;
`model_for` dispatch (incl. unknown-tag ValueError); the real empty-evidence
packet validates; boundary probe -- an evidence row missing `source_id` or
`quote` is rejected; draft without body rejected; brief without project/request
rejected; extra keys round-trip; audit defaults to `revise`; manifest approval
defaults to `pending`.

## Estimated diff size

| File | Lines |
|---|---|
| atlas_brain/schemas/content_factory.py | ~235 |
| tests/test_content_factory_schemas.py | ~135 |
| plans/PR-Content-Factory-Contracts.md | ~70 |
| **Total** | **~440** |

Over the 400-LOC soft cap by ~40: the overage is entirely declarative schema +
tests (no logic, no runtime path), which is low-risk to review as one unit.
