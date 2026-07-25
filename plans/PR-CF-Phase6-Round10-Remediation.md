# PR-CF-Phase6-Round10-Remediation

## Why this slice exists

PR #2192 merged at 2026-07-25T22:13:10Z, seconds after five new review
findings landed on its published head. The merged implementation therefore
contains five confirmed defects in the Phase 6 contract/gate surface:
international vanity contact data passes, ordinary three-digit renderer prose
can be rejected as a phone number, default-ignorable Unicode marks split one
channel into multiple routing identities, worker responses can be stamped
against a same-revision draft they did not see, and IDNA-equivalent domain
separators bypass email gating. Current-head review then found that two class
boundaries were still incomplete: the routing key removed ignorables only
after they could affect normalization, and the dial-token grammar treated a
whitespace separator as exactly one code point rather than a run.
The next current-head review exposed three deeper incomplete boundaries: phone
classification ran before compatibility normalization and encoded separators
in multiple narrow regexes; detached spelling still treated proximity to an
open intent vocabulary as mechanical dial evidence; and `run_stage` accepted
caller-built prompt text independently from the committed draft bytes it
fingerprinted.

This production-hardening follow-up carries the already verified correction
onto the merged mainline. It preserves the Phase 6 product shape and closes the
reported classes at their shared classifier and execution seams.

### Problem-derived contract

- Root cause: the contact decision treated keypad-mappable spelling as enough
  phone evidence while restricting phoneword validity to NANP, so it was
  simultaneously over-broad on detached prose and under-broad on explicit
  international phonewords. The routing key removed only `Cf`/`Cc`, not the
  Unicode default-ignorable marks that also have no routing identity, then
  removed the broader class after NFKC even though an ignorable can block
  canonical composition. The dial-token grammar modeled a whitespace
  separator as one character rather than the entire separator run. The runner
  fingerprinted after the worker returned, so its fingerprint attested to
  persistence-time state rather than dispatch-time source; moving that read
  before dispatch still did not prove the already-built prompt came from those
  bytes. The detached-phoneword decision used an open semantic category
  (nearby intent words) without a closed structural bridge, and phone parsing
  happened before NFKC with separator rules repeated across recognizers. Email
  matching recognized only the ASCII spelling of an IDNA label separator.
- Correct fix must touch/change: the Phase 6 runner's single contact-admission
  decision, its committed-source dispatch/persistence boundary, the channel
  routing key, and both-direction tests at the real `run_stage` entrypoint.
  Default-ignorables must be removed before normalization, and numeric
  whitespace must be consumed as one separator run while compact dial symbols
  remain E.164-bounded. Phone admission must classify an NFKC-normalized view
  with one bounded separator grammar and must replace open keyword proximity
  with a finite structural bridge whose ambiguous default is admissible.
  `run_stage` must construct source-bound prompt text from the exact committed
  bytes whose fingerprint it records, then retain the post-worker under-lock
  comparison. Canonical store reads must remain committed-object reads so a
  source snapshot cannot be replaced by worktree residue.
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
  7. A default-ignorable inserted between a base and combining mark cannot
     prevent canonically equivalent channel labels from colliding.
  8. One or more whitespace code points between international numeric groups
     preserve the same phoneword verdict without widening the compact E.164
     symbol bound or promoting detached prose without dial intent.
  9. NFKC-equivalent phoneword spellings and every admitted dial separator,
     including slash, receive the same verdict as their ASCII/space form.
  10. Detached spelling fails only when a dial marker is connected to it by
      the finite structural bridge; descriptive intervening words default to
      admissible in an evidence-keyed generated oracle.
  11. For every source-bound stage, `run_stage` builds the dispatched prompt
      from the same committed draft bytes it fingerprints. A draft replacement
      before entry changes the prompt source, while a replacement after prompt
      construction remains rejected under the job lock.
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
  Detached spelling requires a dial marker connected through a finite bridge
  containing only direct-address/preposition structure, not arbitrary nearby
  words. International spelling also requires an explicit `+`/`00` prefix and
  a 7-15 digit E.164 mapping. Tests cross evidence signals on both sides rather
  than enumerate reported strings.

### Execution model

- **Closed components:** committed Git objects are canonical source state;
  existing POSIX `flock` is the per-job mutual-exclusion primitive.
- **Invariant:** the committed draft fingerprint observed immediately before
  worker dispatch must be computed from the same bytes used to build the worker
  prompt and must equal the fingerprint re-read under the job lock before a
  source-derived response may persist.
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
by a finite structural dial bridge. Phone admission first creates an
NFKC-normalized view, then one separator grammar governs tokenization,
partitioning, and compact-length checks, including whitespace, dot, hyphen, and
slash. `_dial_shape` still removes separators before enforcing the 7-15 digit
E.164 bound. Space-joined candidate extensions use the structural bridge;
attached domestic vanity syntax remains unambiguous. Email matching translates
the three IDNA domain-stop equivalents only for admission.

The schema defines the Unicode default-ignorable ranges once, excludes them
from visible-only content, and removes them before the NFKC/casefold routing
key so an ignored code point cannot alter composition.

For source-bound work, `run_stage` receives a prompt builder rather than
already-built text. It reads committed draft bytes once, parses those bytes for
the builder, hashes the same bytes, and dispatches the resulting prompt. After
the response is enforced, it takes `job_lock`, rejects any source-derived
schema whose current committed fingerprint differs, stamps audits with the
dispatch fingerprint, runs readiness/lineage, and commits. Store reads use
Git's committed-object lookup for the stage JSON; cleanup is hygiene rather
than correctness.

## Intentional

- Detached phonewords without dial intent pass, even if their letters happen to
  keypad-map to a valid number; rejecting that open prose category would repeat
  the reported false-positive class.
- International phonewords require an explicit international prefix and intent.
  Arbitrary letter/digit prose is not promoted to contact data.
- Whitespace run length is not dial evidence. The existing numeric-group cap,
  compact symbol bound, explicit international prefix, and intent gate remain
  the bounded evidence.
- Compatibility normalization and separator parity apply only to the
  admission-only phone view; raw prompt/artifact text is not rewritten.
- Arbitrary proximity to an intent-like word is not dial evidence. Ambiguous
  detached spelling without the finite bridge remains admissible.
- Source-bound stage callers now provide a prompt builder. This intentionally
  changes the unshipped runner API before the deferred OWUI wrappers are wired.
- The default-ignorable routing normalization is broader than the two reported
  code points by design; one invisible Unicode property is one defect class.
- The originating Phase 6 plan remains modified in this follow-up so its
  review-round record and execution contract match the code that will replace
  the merged defective state.

## Deferred

- None for the five round-10 findings or five current-head class-boundary
  findings.
- Existing Phase 6 deferrals remain: ComfyUI generation, OWUI wrappers, and
  Phase 7 manifest entries.

Parked hardening: none.

## Verification

- Focused content-factory/intake suite: 928 passed.
- Latest current-head regression subset: 126 passed, covering
  compatibility/separator parity, evidence-keyed dial bridges, prebuilt-prompt
  rejection, pre-entry source replacement, and during-worker replacement.
- Prior current-head review regressions remain covered by generated
  canonical-composition and whitespace-run classes.
- Ruff, `python -m py_compile`, and `git diff --check`: passed on the exact
  follow-up tree.
- Schema maturity ratchet: no new brittleness.
- Store explicit-file maturity lane: 0 flagged.
- Guard class-closure, plan shape/files/diff-size/rules, plan/code consistency,
  and plan sync: passed against the merged `origin/main`.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/schemas/content_factory.py` | 60 |
| `atlas_brain/services/content_factory_runner.py` | 340 |
| `atlas_brain/services/content_factory_store.py` | 102 |
| `plans/PR-CF-Phase6-Repurposing-Contracts.md` | 243 |
| `plans/PR-CF-Phase6-Round10-Remediation.md` | 246 |
| `tests/test_content_factory_runner.py` | 508 |
| `tests/test_content_factory_schemas.py` | 68 |
| **Total** | **1567** |
