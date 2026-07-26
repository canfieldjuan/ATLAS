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
The following current-head review found three remaining trust-boundary gaps:
slash-delimited numeric renderer specifications entered through the
unconditional E.164 shortcut; the otherwise finite bridge excluded a single
formatting line break; and a caller callback could ignore the supplied draft
while inheriting its fingerprint.
The latest current-head review found three further edge gaps at those same
boundaries: trailing renderer nouns after an ambiguous candidate were treated as
reverse dial intent, parenthesized phoneword groups were outside the shared
separator grammar, and a missing committed draft serialized as JSON `null`
instead of failing before worker dispatch.

This production-hardening follow-up carries the already verified correction
onto the merged mainline. It preserves the Phase 6 product shape and closes the
reported classes at their shared classifier and execution seams.

**Diff-budget overage (~2,200 LOC vs the 400 soft cap) -- why this slice is
indivisible.** Every finding lands on ONE of two shared choke points: the
contact classifier and the committed-source execution boundary. They cannot be
split into separate slices without publishing a broken intermediate state:

* the classifier findings are alternative bypasses of the SAME decision. Fixing
  the default-ignorable class while leaving the bridge open (or the reverse)
  ships a gate that is still bypassable by a one-character or one-word edit --
  a gate that is 90% closed is not 90% of a gate.
* the execution-boundary findings share one invariant: a stage acts on the
  committed bytes it fingerprinted. Snapshot-vs-compare, prompt construction,
  callback removal and stage admission are the same rule enforced at four
  points; landing a subset leaves a path that still dispatches on unverified
  source.
* roughly 60% of the LOC is tests. The class-closure discipline these findings
  are filed under requires generated cross-products, not examples, so the
  proofs are necessarily larger than the mechanisms. Splitting a mechanism from
  its proof would land the safety decision without the evidence that closes it.

Diff-budget override: shared choke-point fixes are indivisible from their
generated both-direction and reachability proofs.

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
  happened before NFKC with separator rules repeated across recognizers. The
  first finite bridge still ran symmetrically, so renderer nouns such as
  `contact sheet` and `text treatment` after the candidate became reverse dial
  markers. The bounded separator grammar also omitted parenthesized numeric
  groups, and source prompt construction treated a missing committed draft as a
  serializable `None` snapshot. Email matching recognized only the ASCII
  spelling of an IDNA label separator.
- Correct fix must touch/change: the Phase 6 runner's single contact-admission
  decision, its committed-source dispatch/persistence boundary, the channel
  routing key, and both-direction tests at the real `run_stage` entrypoint.
  Default-ignorables must be removed before normalization, and numeric
  whitespace must be consumed as one separator run while compact dial symbols
  remain E.164-bounded. Phone admission must classify an NFKC-normalized view
  with one bounded separator grammar and must replace open keyword proximity
  with a finite structural bridge whose ambiguous default is admissible.
  Slash-delimited numeric shapes must require structural dial syntax instead
  of entering through an unconditional prefix shortcut, while one logical line
  break remains formatting inside the finite bridge. `run_stage` must own a
  fixed stage prompt template and construct it from the exact committed bytes
  whose fingerprint it records; caller-controlled builders are not proof of
  derivation. Missing committed draft bytes must fail before any source-bound
  worker dispatch. Parenthesized phoneword groups must share the same bounded
  grammar as their numeric keypad equivalents, and trailing renderer nouns must
  not govern ambiguous candidates as reverse dial markers. The post-worker
  under-lock comparison remains required.
  Canonical store reads must remain committed-object reads so a source snapshot
  cannot be replaced by worktree residue.
- Must not change: Phase 6 artifact schema tags/field shapes, editorial
  decision semantics, negative-prompt exclusion semantics, body-copy verifier
  policy, image generation, worker wrappers, or any EOM/CRM lane.

## Scope (this PR)

Ownership lane: content-factory
Slice phase: production hardening
Max files: 8

1. Make mixed digit/letter contact admission evidence-gated in both directions:
   attached NANP vanity stays structural; detached spelling requires structural
   dial intent before the ambiguous candidate; international phonewords require
   explicit `+`/`00` structure and an E.164-bounded keypad mapping.
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
  12. Slash-delimited numeric renderer values such as `+1920/1080` and
      `+2026/07/25` remain admissible without structural dial syntax, while
      slash-separated phonewords and numeric candidates under that syntax
      retain the failing phone verdict.
  13. A single LF, CR, or CRLF between a dial marker and candidate is formatting
      and preserves dial evidence; paragraph breaks and descriptive intervening
      words remain inadmissible as bridges.
  14. Source-bound stages accept no caller-controlled prompt or callback. The
      runner selects the stage instruction and serializes the committed draft
      snapshot itself; a callback that ignores its argument is rejected before
      worker dispatch.
  15. Parenthesized numeric groups in a phoneword, e.g. `Call +44 (800) FLOWERS`,
      receive the same failing verdict as the unparenthesized spelling and the
      keypad-equivalent numeric spelling.
  16. Ambiguous phonewords are not governed by trailing renderer nouns: `room 212
      art deco contact sheet`, `room 212 art deco text treatment`, and `poster
      for room 212 art deco, contact sheet layout` remain admissible.
  17. Every source-bound stage requires a valid committed draft before worker
      dispatch; no stage may send `Committed draft JSON:\nnull`.
  21. An overloaded dial marker governs an ambiguous candidate only when it
      HEADS its clause: `Text to +44 800 FLOWERS` fails, while `center text on
      1920 1080 canvas` and `place text at 1920 1080 coordinates` render.
  22. The `audit-v2` stage admits only editorial-audit schemas; a mislabeled
      worker artifact cannot be committed under it.
  20. EVERY Unicode Default_Ignorable_Code_Point -- sampled from the shared
      range table itself, not a hand-written list -- fails at every contact
      seam in the prompt gate and `verify_copy`, and none leaves an address
      readable in redacted evidence. Routing keys and the contact scan use ONE
      predicate.
  19. A dial marker connected to its number by the dative `to` -- `Text to
      +44 800 FLOWERS`, `SMS to ...`, `WhatsApp to ...` -- fails, matching the
      same marker and number without the connector. Possessive determiners
      stay OUT of the bridge: `text your 1920 1080 export` and `call our 1920
      1080 render sheet` remain renderer prose.
  18. A default-ignorable code point inserted at ANY seam of a contact string
      -- before or inside the `+`/`00` prefix, between or inside numeric
      groups, inside the vanity suffix, or inside an address -- preserves the
      failing verdict, in the prompt gate AND in `verify_copy`. Redacted claim
      evidence never carries an address that evaded the pattern that way.
      Whitespace controls are preserved, so stripping cannot glue a word to a
      following number and manufacture a candidate.
- Reachability proof: `run_stage(job_id, stage, model, user_content, ...)` calls
  the worker, applies prompt enforcement, compares committed source identity
  under `job_lock`, validates, and writes. Tests replace the committed draft
  from inside the worker boundary and exercise image prompts through the real
  entrypoint/store.
- Affected surfaces: content-factory schemas, runner, the shared copy
  verifier's PII SCAN INPUT (its verdict policy is unchanged -- see the
  round-eleven note under Mechanism), committed Git artifact store, the
  originating Phase 6 plan record, and their focused tests.
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
  Detached spelling requires a preceding dial marker connected through a finite
  bridge containing only direct-address/preposition structure, not arbitrary
  nearby words or trailing renderer nouns. International spelling also requires
  an explicit `+`/`00` prefix and a 7-15 digit E.164 mapping. Tests cross
  evidence signals on both sides rather than enumerate reported strings.

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
- `atlas_brain/services/content_factory_copy_verification.py`
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
partitioning, and compact-length checks, including whitespace, dot, hyphen,
slash, and parenthesized numeric groups. `_dial_shape` still removes separators
before enforcing the 7-15 digit E.164 bound. Space-joined candidate extensions
use the forward structural bridge; attached domestic vanity syntax remains
unambiguous. Email matching translates the three IDNA domain-stop equivalents
only for admission.

The schema defines the Unicode default-ignorable ranges once, excludes them
from visible-only content, and removes them before the NFKC/casefold routing
key so an ignored code point cannot alter composition.

For source-bound work, `run_stage` accepts no caller prompt payload. It selects
a fixed instruction from the known stage contract, requires committed draft
bytes to exist, parses and deterministically serializes that snapshot into the
prompt, hashes the same bytes, and dispatches. After the response is enforced,
it takes
`job_lock`, rejects any source-derived schema whose current committed
fingerprint differs, stamps audits with the dispatch fingerprint, runs
readiness/lineage, and commits. Store reads use Git's committed-object lookup
for the stage JSON; cleanup is hygiene rather than correctness.

### Review round 11 (Codex) -- default-ignorable admission

One finding, confirmed against the code before correction and WIDER than
reported. A default-ignorable code point renders as nothing, so inserting one
must not change a verdict -- but it defeated:

- the `+`/`00` international prefix (`Call +44<ZWSP>800 FLOWERS`), as reported;
- the structural NANP pattern (`reach 555-123<ZWSP>-4567`) and the vanity
  suffix (`FLOW<ZWSP>ERS`), which the report did not name;
- the SAME class in `verify_copy` -- the merged body-copy gate behind the
  editorial promote decision -- for both phone and address;
- `_redact_pii`, where the digit theorem masks phone digits but an address's
  LETTERS are not digits, so `a@b<ZWSP>.com` persisted intact in claim
  evidence.

Root fix at the admission boundary rather than per-pattern: a shared
`scan_view()` removes default-ignorables (category Cf, category Cc except the
whitespace controls, plus the variation-selector / Mongolian / tag / Hangul
filler ranges Unicode marks Default_Ignorable but `unicodedata` does not
expose) before ANY contact parsing. Claim detection keeps the ORIGINAL text,
because its locators are sentence indices and rewriting the input would shift
them.

This does not change body-copy verdict POLICY -- which real forms count as PII
is untouched, so #2181's frozen semantics hold. It denies an evasion of the
existing policy. Mutation-checked: neutralizing `scan_view` fails 64 tests.

### Review round 12 (Codex) -- recipient-marking bridge

One finding, confirmed: the bridge vocabulary omitted `to`, so `Text to +44
800 FLOWERS` produced no hit while `Text +44 800 FLOWERS` was rejected. A
one-word grammatical connector defeated the gate.

`to` is the dative marker every message verb takes for its recipient, so it
belongs to the same closed function-word class the bridge already models.

Possessive determiners are deliberately NOT added, and the boundary is
recorded so the exclusion is not mistaken for an oversight: `our`/`your`/
`their` are function words too, but they occur in ordinary renderer prose
("text your 1920 1080 export"), so admitting them would read a typography
instruction as a dial instruction. The class is function words that mark a
RECIPIENT, not every function word. Covered by a 7 intents x 5 bridge-forms
cross-product plus the prose-side probes.

### Review round 13 (Codex) -- one shared default-ignorable predicate

One finding, confirmed, and the residual leak was worse than the phone case.
Round 11's `scan_view` tested category Cf/Cc plus a hand-built range set, which
misses U+034F COMBINING GRAPHEME JOINER -- category **Mn**. So after the
zero-width class was closed, `Call +44<CGJ>800 FLOWERS` and
`555-123<CGJ>-4567` still passed, and `_redact_pii` left
`alice@example<CGJ>.com` FULLY intact in persisted claim evidence: the digit
theorem cannot help, because an address has no digits to mask.

The root cause is not the missing codepoint, it is the SECOND DEFINITION. The
schema module already had the complete Default_Ignorable range table for
routing keys; any rule elsewhere phrased as "Cf plus some ranges" leaves the
bypass open by construction. `is_default_ignorable` is now exported as the one
definition, used by routing keys, the contact scan, and evidence redaction
alike.

The corpus is derived FROM that table rather than listed, so a codepoint added
to it is covered automatically and a future local copy fails the shared-
predicate assertion first. Mutation-checked: restoring the Cf/Cc-only rule
fails 134 tests.

### Review round 14 (Codex) -- dial-verb evidence, stage admission, plan record

Three findings, all confirmed.

1. **P1 renderer coordinates inside the dial bridge.** `center text on 1920
   1080 canvas` and `place text at 1920 1080 coordinates` produced a phone hit,
   because `text` is an admitted marker and `on`/`at` are admitted bridges.
   Pre-existing rather than introduced by the round-12 `to` addition -- `on`
   was already in the set at `f5485fcad`.

   Fixed by POSITION, not by a list of renderer verbs (an open class): an
   imperative CTA heads its clause, while a typography instruction makes the
   same word the object of an earlier verb. Only overloaded markers
   (`text`, `contact`) are subject to the test, so unambiguous markers keep
   full bridge behaviour and `please call me at 12 34 56 78` still fails.
   Residual recorded in the code: `you can text me at <ambiguous number>` is
   not read as a CTA -- unambiguous dial syntax never reaches this bridge, so
   what is given up is a phrasing renderer instructions do not use.

2. **P2 unregistered `audit-v2` stage.** The runner names it as a source-bound
   stage but the store had no entry, so it was treated as a CUSTOM stage that
   may carry ANY artifact: a worker returning `content_brief.v1` committed
   successfully under that stage, and since its schema is outside
   `_SOURCE_BOUND_SCHEMAS` the dispatch-source comparison was skipped as well.
   Mapped to the same admissible set as `audit`.

3. **P1 plan record.** The over-budget rationale lived in the commit message,
   but the repository contract requires it in the plan's `Why this slice
   exists`. Added there with the actual indivisibility argument.

## Intentional

- Detached phonewords without preceding structural dial intent pass, even if
  their letters happen to keypad-map to a valid number; rejecting that open
  prose category would repeat the reported false-positive class.
- International phonewords require an explicit international prefix and intent.
  Arbitrary letter/digit prose is not promoted to contact data.
- Whitespace run length is not dial evidence. The existing numeric-group cap,
  compact symbol bound, explicit international prefix, and intent gate remain
  the bounded evidence.
- Compatibility normalization and separator parity apply only to the
  admission-only phone view; raw prompt/artifact text is not rewritten.
- Arbitrary proximity to an intent-like word is not dial evidence. Ambiguous
  detached spelling without the forward finite bridge remains admissible.
- Source-bound stage callers now pass no prompt payload; the runner owns the
  fixed instruction plus committed-draft serialization. This intentionally
  changes the unshipped API before the deferred OWUI wrappers are wired.
- The default-ignorable routing normalization is broader than the two reported
  code points by design; one invisible Unicode property is one defect class.
- The originating Phase 6 plan remains modified in this follow-up so its
  review-round record and execution contract match the code that will replace
  the merged defective state.

## Deferred

- None for the five round-10 findings or eleven current-head class-boundary
  findings.
- Existing Phase 6 deferrals remain: ComfyUI generation, OWUI wrappers, and
  Phase 7 manifest entries.

Parked hardening: none.

## Verification

- Focused content-factory/intake suite: 956 passed.
- Latest round-14 regression subset: 88 passed, covering
  parenthesized phoneword groups, trailing renderer nouns after ambiguous
  candidates, missing committed-draft rejection before dispatch, slash-separated
  renderer/phone both-direction cases, bounded line-break bridges,
  runner-owned source prompts, and callback rejection. The prior current-head
  subsets remain covered for compatibility/separator parity, evidence-keyed dial
  bridges, prebuilt-prompt rejection, pre-entry source replacement, and
  during-worker replacement.
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
| `atlas_brain/schemas/content_factory.py` | 72 |
| `atlas_brain/services/content_factory_copy_verification.py` | 56 |
| `atlas_brain/services/content_factory_runner.py` | 421 |
| `atlas_brain/services/content_factory_store.py` | 108 |
| `plans/PR-CF-Phase6-Repurposing-Contracts.md` | 284 |
| `plans/PR-CF-Phase6-Round10-Remediation.md` | 440 |
| `tests/test_content_factory_runner.py` | 951 |
| `tests/test_content_factory_schemas.py` | 68 |
| **Total** | **2400** |
