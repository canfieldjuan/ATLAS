# PR-CF-Phase6-Round10-Remediation

## Why this slice exists

PR #2192 merged at 2026-07-25T22:13:10Z, seconds after five new review
findings landed on its published head. The merged implementation therefore
contains five confirmed defects in the Phase 6 contract/gate surface:
international vanity contact data passes, ordinary three-digit renderer prose
can be rejected as a phone number, default-ignorable Unicode marks split one
channel into multiple routing identities, worker responses can be stamped
against a same-revision draft they did not see, and IDNA-equivalent domain
separators bypass email gating.

This production-hardening follow-up carries the already verified correction
onto the merged mainline. It preserves the Phase 6 product shape and closes the
reported classes at their shared classifier and execution seams.

### Problem-derived contract

- Root cause: the contact decision treated keypad-mappable spelling as enough
  phone evidence while restricting phoneword validity to NANP, so it was
  simultaneously over-broad on detached prose and under-broad on explicit
  international phonewords. The routing key removed only `Cf`/`Cc`, not the
  Unicode default-ignorable marks that also have no routing identity. The
  runner fingerprinted after the worker returned, so its fingerprint attested
  to persistence-time state rather than dispatch-time source. Email matching
  recognized only the ASCII spelling of an IDNA label separator.
- Correct fix must touch/change: the Phase 6 runner's single contact-admission
  decision, its committed-source dispatch/persistence boundary, the channel
  routing key, and both-direction tests at the real `run_stage` entrypoint.
  Canonical store reads must remain committed-object reads so a source snapshot
  cannot be replaced by worktree residue.
- Must not change: Phase 6 artifact schema tags/field shapes, editorial
  decision semantics, negative-prompt exclusion semantics, body-copy verifier
  policy, image generation, worker wrappers, or any EOM/CRM lane.

## Scope (this PR)

Ownership lane: content-factory
Slice phase: production hardening
Max files: 7

1. Make mixed digit/letter contact admission evidence-gated in both directions:
   attached NANP vanity stays structural; detached spelling requires nearby
   dial intent; international phonewords require explicit `+`/`00` structure
   and an E.164-bounded keypad mapping.
2. Normalize IDNA-equivalent domain stops before the redacted email decision
   and normalize Unicode default-ignorables out of channel routing identities.
3. Snapshot committed draft identity immediately before dispatch and compare
   it under the existing per-job lock before any audit, repurposing, or
   image-prompt response persists.
4. Record the originating plan's round-10 reconciliation and add
   grammar-derived both-direction, default-ignorable class, and real
   mid-worker replacement proofs.

### Review Contract

- Acceptance criteria:
  1. `Call +44 800 FLOWERS today` and equivalent `00`/separator-generated
     phonewords fail the prompt gate with a redacted phone hit.
  2. Ordinary three-digit values crossed with art-direction phrases, including
     `room 212 art deco sign`, pass without dial intent.
  3. `email` and `email` plus any tested default-ignorable variation selector,
     grapheme joiner, or format character are duplicate channels.
  4. A same-revision committed draft replacement during the worker call makes
     audit, repurposing, and image-prompt responses unpersistable, including
     unready intermediate Phase 6 artifacts.
  5. U+3002, U+FF0E, and U+FF61 domain separators receive the same failing
     email verdict as ASCII dot without leaking the raw address into hits.
  6. Prior Phase 6 gate, store, schema, and intake tests remain green; maturity,
     guard-closure, plan, static, and diff audits pass.
- Reachability proof: `run_stage(job_id, stage, model, user_content, ...)` calls
  the worker, applies prompt enforcement, compares committed source identity
  under `job_lock`, validates, and writes. Tests replace the committed draft
  from inside the worker boundary and exercise image prompts through the real
  entrypoint/store.
- Affected surfaces: content-factory schemas, runner, committed Git artifact
  store, the originating Phase 6 plan record, and their focused tests.
- Risk areas: false-positive/false-negative contact classification, Unicode
  normalization overreach, worker-time source races, and failed-commit residue.
- Reviewer rules triggered: R1, R2, R3, R6, R8, R13, R14.

### Decision-Seam Analysis

- **One decision:** `_phone_evidence` decides whether a mixed digit/letter
  candidate has enough bounded evidence to be contact data rather than
  renderer prose.
- **Why it was wrong:** keypad mapping alone admitted ordinary language, while
  NANP membership excluded explicit international phonewords. These are the
  two error directions of one classifier seam.
- **Structural direction:** ambiguous detached prose defaults to admissible.
  Detached spelling requires nearby dial intent. International spelling also
  requires an explicit `+`/`00` prefix and a 7-15 digit E.164 mapping. Tests
  cross generators on both sides rather than enumerate reported strings.

### Execution model

- **Closed components:** committed Git objects are canonical source state;
  existing POSIX `flock` is the per-job mutual-exclusion primitive.
- **Invariant:** the committed draft fingerprint observed immediately before
  worker dispatch must equal the fingerprint re-read under the job lock before
  a source-derived response may persist.
- **Failure boundary:** worker transport remains outside the lock. A
  cooperative writer may replace the draft during transport, but the stale
  response is rejected. Ordinary store failures restore worktree/index hygiene;
  abrupt residue remains non-canonical because readers use `HEAD`.
- **Actors/assumptions:** synchronous same-host callers using this store;
  trusted local root; Git/POSIX flock available; bypass writers are unsupported.
- **Rejected alternative:** holding the filesystem lock across a network call
  would block all job mutations for up to the worker timeout. Snapshot/compare
  provides the needed optimistic concurrency check with the existing closed
  components and no lease/retry protocol.

### Files touched

- `atlas_brain/schemas/content_factory.py`
- `atlas_brain/services/content_factory_runner.py`
- `atlas_brain/services/content_factory_store.py`
- `plans/PR-CF-Phase6-Repurposing-Contracts.md`
- `plans/PR-CF-Phase6-Round10-Remediation.md`
- `tests/test_content_factory_runner.py`
- `tests/test_content_factory_schemas.py`

## Mechanism

The runner preserves a leading `+`, keypad-maps mixed tokens, and classifies
explicit international phonewords as ambiguous structural candidates governed
by the existing bounded intent window. Space-joined candidate extensions use
that same intent requirement; attached domestic vanity syntax remains
unambiguous. Email matching translates the three IDNA domain-stop equivalents
only for admission.

The schema defines the Unicode default-ignorable ranges once, excludes them
from visible-only content, and removes them from the NFKC/casefold routing key.

`run_stage` reads the committed draft fingerprint before `call_worker`. After
the response is enforced, it takes `job_lock`, rejects any source-derived
schema whose current committed fingerprint differs, stamps audits with the
dispatch fingerprint, runs readiness/lineage, and commits. Store reads use
`git show HEAD:<stage>.json`; cleanup is hygiene rather than correctness.

## Intentional

- Detached phonewords without dial intent pass, even if their letters happen to
  keypad-map to a valid number; rejecting that open prose category would repeat
  the reported false-positive class.
- International phonewords require an explicit international prefix and intent.
  Arbitrary letter/digit prose is not promoted to contact data.
- The default-ignorable routing normalization is broader than the two reported
  code points by design; one invisible Unicode property is one defect class.
- The originating Phase 6 plan remains modified in this follow-up so its
  review-round record and execution contract match the code that will replace
  the merged defective state.

## Deferred

- None for the five round-10 findings.
- Existing Phase 6 deferrals remain: ComfyUI generation, OWUI wrappers, and
  Phase 7 manifest entries.

Parked hardening: none.

## Verification

- Focused content-factory/intake suite: 714 passed.
- Ruff, `python -m py_compile`, and `git diff --check`: passed before branch
  transfer; rerun on the exact follow-up head before push.
- Schema maturity ratchet: no new brittleness.
- Store explicit-file maturity lane: 0 flagged.
- Guard class-closure, plan shape/files/diff-size/rules, plan/code consistency,
  and plan sync: rerun on the exact follow-up head before push.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/schemas/content_factory.py` | 43 |
| `atlas_brain/services/content_factory_runner.py` | 233 |
| `atlas_brain/services/content_factory_store.py` | 102 |
| `plans/PR-CF-Phase6-Repurposing-Contracts.md` | 202 |
| `plans/PR-CF-Phase6-Round10-Remediation.md` | 184 |
| `tests/test_content_factory_runner.py` | 286 |
| `tests/test_content_factory_schemas.py` | 40 |
| **Total** | **1090** |
