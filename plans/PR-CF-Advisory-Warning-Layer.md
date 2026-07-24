# PR-CF-Advisory-Warning-Layer

## Why this slice exists

#2136 item 2: the pipeline records only the promote-blocking verdict; the
source verifier tool's softer "needs human review" checks (owner-routing
coverage, unqualified answer/ownership claims, honest-CTA reminder) exist
only in the opt-in OWUI tool, so a pipeline audit carries no advisory
signal for the approving human.

This slice exceeds the 400-LOC soft cap and is indivisible: the advisory
producer, the versioned contract that carries it (v2 + frozen v1), the
runner normalization, and the both-direction precision tests are one
reviewable behavior -- landing the producer without the versioned carrier
would re-open the rollback hazard round 2 flagged, and landing the carrier
without tests would ship unproven heuristics.

### Problem-derived contract

- Root cause: the deterministic gate covers blocking categories only;
  the advisory layer was never ported into the repo pipeline.
- Correct fix must: add a deterministic, PII-safe advisory producer next
  to the gate; persist its output on the audit artifact; inject it in the
  runner with the same self-report discipline as the verdict (worker
  claims discarded); change NO gating behavior.
- Correct fix must also (contract REVISED in round 3, per the review):
  version the audit contract (v2 carries the checklist; v1 frozen) and
  admit both versions at the store's audit stage -- persisting the new
  field is impossible without the version-admission touch, so the store's
  STAGE_SCHEMAS/admission check is IN scope for exactly that change.
- Must not change: `verify_copy` verdict semantics, the promote
  validator, any other store behavior, the OWUI tool (it is the source
  and already carries this layer, re-synced v0.2.0), the pre-#2181
  `EditorialAudit.model_validate(v1_payload)` API.

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

### Review round 4 (Codex)

Four findings, all fixed — the big one structurally: advisory warnings no
longer persist ANY free draft text. Each warning records only the claim
code, the 1-based sentence number, and the matched keyword (word
characters by construction), so the no-raw-PII criterion holds by
construction instead of by redaction completeness ("020 - 7946 - 0958"
and every future separator style included). The reviewer locates the
sentence in the draft artifact beside the audit. Consequences: sentence/
clause boundaries are precomputed once per draft with O(log n) lookups
(kills the quadratic rescan finding); qualifiers bind per coordinated
clause (and/or are boundaries now, so "one answer when evidence exists
and another answer regardless" warns); owner-routing coverage requires a
negation-free COMPLETE clause ("assigned to nobody", "routing remains
unresolved" now warn). The gate's claim-hit evidence (which does record
matched phrases) keeps the digit-run redaction backstop.

### Review round 3 (Codex)

Four findings, all fixed: `EditorialAudit` keeps its pre-change v1 API
(the v2 contract is `EditorialAuditV2`; registry dispatches both);
evidence redaction gains a CLASS backstop -- any 5+ digit run joined by
single non-word separators is masked (`020/7946/0958` included), ending
the format-enumeration game on the evidence path; owner-routing coverage
requires a NON-NEGATED affirmative relation ("no one is assigned",
"not routed to Billing" now warn; bare "responsible for" no longer
suppresses); and this contract's "must not change the store" line was
revised to name the version-admission touch the v2 artifact requires
(the contradiction the review flagged).

### Review round 2 (Codex)

Nine findings, all fixed: sentence terminators added to clause boundaries;
owner-routing suppression requires AFFIRMATIVE assignment/review language
(bare "owner" no longer suppresses "the owner is unknown"); report-shape
matching is relational (report-noun + output-verb, or product terms) so
"The compliance audit passed" is silent; responsibility claims need an
owner-like subject; international AND local phone-shaped digit runs are
redacted in advisory evidence but the GATE expansion was REVERTED (the
slice contract freezes verdict semantics -- widening the PII block is a
separate operator decision); `editorial_audit.v2` carries
advisory_warnings while v1 is FROZEN and stays admissible for the audit
stage (rollback-safe: no v1-tagged artifact ever carries the new field;
the runner normalizes worker output to v2); a run_stage boundary test
proves warning persistence at the real entrypoint; the plan size table
was re-synced to the actual diff with this override rationale.

### Review round 1 (Codex)

Five precision/PII findings on the ported heuristics, all fixed: owner-routing
suppression requires relational routing language (bare topic nouns like
"billing" no longer suppress); product names (Resolution Audit/Snapshot) are
excluded from the answer-claim detector; qualifiers are evaluated per CLAUSE
so one qualified assertion cannot hide a separate unqualified one;
report-shape matching drops context-free nouns (draft/ranked/faqs);
international phone formats are redacted AND block the gate (+country-code
patterns; over-matching feeds redaction/blocking, both fail-closed).
The httpx workflow dependency this PR's CI surfaced was split into
PR #2183 (trusted-base execution means an in-PR workflow edit can never
fix its own CI) and is merged on main.

### Files touched

- `atlas_brain/schemas/content_factory.py`
- `atlas_brain/services/content_factory_copy_verification.py`
- `atlas_brain/services/content_factory_runner.py`
- `atlas_brain/services/content_factory_store.py` (round 2: audit stage
  admits both audit versions)
- `plans/PR-CF-Advisory-Warning-Layer.md`
- `tests/test_content_factory_copy_verification.py`
- `tests/test_content_factory_runner.py`
- `tests/test_content_factory_schemas.py` (round 2: v2 coverage + frozen-v1
  proofs)

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

- Content-factory suites: 178 passed (12 new advisory tests, 2 new runner
  tests). Adjacent `tests/test_leads_intake.py` green (187 combined).
- `python -m py_compile` on the three touched modules.
- NOT run: live OWUI worker pass (advisory output shape is fully covered
  by unit tests; next live pipeline run will carry the checklist).

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/schemas/content_factory.py` | 75 |
| `atlas_brain/services/content_factory_copy_verification.py` | 175 |
| `atlas_brain/services/content_factory_runner.py` | 25 |
| `atlas_brain/services/content_factory_store.py` | 12 |
| `plans/PR-CF-Advisory-Warning-Layer.md` | 200 |
| `tests/*` (three files) | 210 |
| **Total** | **~700 gross (over the 400 soft cap; override rationale in
"Why this slice exists")** |
