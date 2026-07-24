# PR-CF-Advisory-Warning-Layer

## Why this slice exists

#2136 item 2: the pipeline records only the promote-blocking verdict; the
source verifier tool's softer "needs human review" checks (owner-routing
coverage, unqualified answer/ownership claims, honest-CTA reminder) exist
only in the opt-in OWUI tool, so a pipeline audit carries no advisory
signal for the approving human.

### Problem-derived contract

- Root cause: the deterministic gate covers blocking categories only;
  the advisory layer was never ported into the repo pipeline.
- Correct fix must: add a deterministic, PII-safe advisory producer next
  to the gate; persist its output on the audit artifact; inject it in the
  runner with the same self-report discipline as the verdict (worker
  claims discarded); change NO gating behavior.
- Must not change: `verify_copy` verdict semantics, the promote validator,
  the store, the OWUI tool (it is the source and already carries this
  layer, re-synced v0.2.0).

## Scope (this PR)

Ownership lane: content-factory
Slice phase: vertical slice

1. `atlas_brain/services/content_factory_copy_verification.py`:
   `advisory_warnings(text)` ports the tool's soft checks (unqualified
   answer claims, unqualified ownership claims, report-shape without
   owner routing, unconditional honest-CTA reminder). The whole text is
   PII-redacted BEFORE sentence extraction: the sentence splitter breaks
   on the dot inside an email address, and per-sentence redaction leaks
   the truncated fragment ("bob@example") -- found by this slice's own
   probe; the source tool has the same flaw.
2. `atlas_brain/schemas/content_factory.py`: `EditorialAudit` gains
   `advisory_warnings: list[str]` (default empty). Deliberately NOT
   referenced by any validator -- warnings never gate the recommendation.
3. `atlas_brain/services/content_factory_runner.py`:
   `_enforce_copy_verification` overwrites `advisory_warnings` from the
   edited body alongside the verdict; empty body clears the list
   (a fabricated checklist must not blind the reviewer).
4. Proof: category tests both directions, PII probe, promote-with-warnings
   validity, runner overwrite + empty-body clearing, old artifacts without
   the field still validate.

### Review Contract

- Acceptance criteria:
  1. Each advisory category warns on its trigger and stays silent on the
     qualified/covered form (test-asserted both directions).
  2. A passing verdict may promote regardless of warnings; warnings are
     absent from every gating validator (schema review + test).
  3. Runner-persisted audits carry the deterministic checklist, never the
     worker's; empty edited body yields an empty checklist plus the
     existing fail verdict.
  4. No recorded warning can carry raw email/phone text (pre-redaction
     probe includes the sentence-split truncation case).
- Reachability proof: `run_stage` on any editorial audit (schema-gated,
  same path as the #2137 verdict enforcement); artifact lands in the
  git-backed job folder for the approving human.
- Affected surfaces: copy-verification module, editorial-audit contract,
  runner enforcement hook. Gating behavior unchanged.
- Risk areas: warning noise (CTA reminder is unconditional by design,
  mirroring the source tool -- consumers treat it as a checklist line,
  not a signal); regex drift vs the OWUI tool (the repo module is
  canonical; the tool is the synced copy).
- Reviewer rules triggered: R1 (#2136 item 2), R2 (both-direction tests),
  R5 (no gating change, old artifacts validate), R10 (advisory logic
  lives beside the gate it complements), R14.

### Files touched

- `atlas_brain/schemas/content_factory.py`
- `atlas_brain/services/content_factory_copy_verification.py`
- `atlas_brain/services/content_factory_runner.py`
- `plans/PR-CF-Advisory-Warning-Layer.md`
- `tests/test_content_factory_copy_verification.py`
- `tests/test_content_factory_runner.py`

## Mechanism

One producer, one optional contract field, one runner injection point.
The runner treats the checklist exactly like the verdict (deterministic,
overwritten, never worker-supplied), so the reviewing human reads trusted
advisory output in audit.json without any new gate.

## Intentional

- **CTA reminder is unconditional** -- faithful to the source tool; every
  audit carries at least one checklist line, so an empty list is not a
  "clean" signal and is not treated as one anywhere.
- **Warnings never block** -- no validator references them; blocking
  advisory content would collapse the gate/checklist distinction #2136
  drew on purpose.
- **Pre-redaction over per-sentence redaction** -- closes the
  email-dot/sentence-split truncation leak this slice's probe found.

## Deferred

- #2136 item 4 (catalogue growth) remains standing operator policy.
- Rendering warnings in any UI/manifest surface (Phase 7 observability).

Parked hardening: none new.

## Verification

- Content-factory suites: 150 passed (12 new advisory tests, 2 new runner
  tests). Adjacent `tests/test_leads_intake.py` green (187 combined).
- `python -m py_compile` on the three touched modules.
- NOT run: live OWUI worker pass (advisory output shape is fully covered
  by unit tests; next live pipeline run will carry the checklist).

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/schemas/content_factory.py` | 6 |
| `atlas_brain/services/content_factory_copy_verification.py` | 105 |
| `atlas_brain/services/content_factory_runner.py` | 12 |
| `plans/PR-CF-Advisory-Warning-Layer.md` | 120 |
| `tests/test_content_factory_copy_verification.py` | 90 |
| `tests/test_content_factory_runner.py` | 35 |
| **Total** | **~368** |
