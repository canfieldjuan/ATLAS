# PR-CF-Phase6-Repurposing-Contracts

## Why this slice exists

Epic #2109 Phase 6 (repurposing + image workflow) is the first phase whose
output is the operator's actual workload: channel variants and media assets
from an approved draft. Phase 3 is now closed (3a shipped; 3b/RAG is
won't-do), so Phase 6 is the next unstarted phase.

Phase 6 has two halves. This slice is the ARTIFACT half: the contracts for
`repurposing.v1` and `image_prompt.v1`, their stage wiring, and their
deterministic gates. Image GENERATION (ComfyUI) is deliberately not here --
the epic keeps prompt designer and generator split, and generation is
human-triggered and VRAM-guarded.

**Diff-budget overage (~3,150 LOC vs the 400 soft cap) — why this slice is
indivisible.** The two contracts, their stage admission, and their
deterministic gates are one enforceable behaviour, and splitting them
produces a strictly worse intermediate state:

* contracts without their gates would let a worker self-declare shippable
  or renderable copy -- the exact failure the editorial gate exists to
  prevent, shipped as a landed artifact surface;
* gates without the readiness contracts would have nothing to refuse,
  since the flags they guard would not exist;
* stage admission without either would let unvalidated artifacts land in
  the git-backed job folder under a Phase 6 stage name.

Roughly 46% of the LOC is tests (ten review rounds of adversarial
regressions, kept because each pins a defect that actually shipped in an
earlier revision of this branch). The behaviour itself is two contracts,
one enforcement hook, and one lineage check.

### Problem-derived contract

- Root cause: the pipeline stops at an approved audit. Nothing turns an
  approved draft into per-channel copy or image prompts, and nothing gates
  those outputs -- yet variants are the copy that actually ships and prompt
  text is rendered into artwork where no text check can see it.
- Correct fix must: define both artifacts with invariants that make the
  failure modes unrepresentable (orphan variants, duplicate channels,
  self-declared shippability, empty packages); admit both at the store's
  stage map; and run the SAME deterministic overwrite the editorial audit
  uses so a worker cannot self-approve either one.
- Must not change: the editorial gate's DECISION behavior, the advisory
  grammar (extracted to a shared validator, same rules), or anything about
  image generation (separate slice). Published contract SHAPES stay frozen
  and readable -- `editorial_audit.v1` and `.v2` both keep validating
  byte-for-byte.
- Must ALSO change (admitted in review, round 8 -- the original "must not
  change existing contracts/stages" was too narrow to hold, because
  readiness cannot be enforced without touching both):
  - **Audit version.** Binding approval to draft CONTENT requires a field
    on the audit. v2 is published and forbids extras, so the field cannot
    go there without making new audits unreadable to a v2 consumer or a
    rolled-back reader. Adds `editorial_audit.v3` (subclassing v2 so the
    promote gate and warning grammar cannot drift), re-freezes v2, and
    upgrades v1/v2 -> v3 in the runner's normalizer. Acceptance: a v2
    payload still validates; a v2 payload carrying the new field is
    REJECTED (proving the freeze); self-contradictory worker metadata is
    still refused rather than repaired.
  - **Store concurrency + write atomicity.** Readiness reads the job
    history and then writes it, so both must sit in one critical section.
    Canonical readiness inputs must come from the committed Git tree, never
    mutable worktree/index residue. Adds a re-entrant per-job `flock`,
    committed-artifact reads, and failure cleanup that restores the previous
    committed bytes/index state. Acceptance: readiness is decided with the
    lock HELD (mutation-checked) and released after; a failed commit is never
    readable as canonical source state, even if interruption prevents cleanup.
    A source fingerprint is also captured before worker dispatch and compared
    again under the lock, so a response can never be stamped against a
    same-revision draft that replaced the one present when work began.

## Scope (this PR)

Ownership lane: content-factory
Slice phase: vertical slice
Max files: 8

1. `atlas_brain/schemas/content_factory.py`:
   - `ChannelVariant` + `RepurposingPackage` (`repurposing.v1`): non-empty
     variants, non-blank channel/body, claim lineage, one channel per
     variant, and `ready_to_publish` gated on EVERY variant carrying a
     passing verdict (the variant-level analogue of the promote gate).
   - `ImagePrompt` + `ImagePromptSet` (`image_prompt.v1`): text prompts
     only, at least one prompt, gated prompt text.
   - `_validate_advisory_warnings` extracted so audit, variants, and image
     prompts share ONE bounded grammar and cannot drift.
2. `atlas_brain/services/content_factory_store.py`: `repurposing` and
   `image_prompt` stages admit their matching schemas.
3. `atlas_brain/services/content_factory_runner.py`: `_enforce_repurposing`
   recomputes each variant's verdict + checklist from its OWN body;
   `_enforce_image_prompts` gates the POSITIVE prompt text only, verifying
   each prompt independently (`negative_prompt` is an exclusion list and is
   deliberately excluded; joining items would synthesize cross-prompt
   claims). Prompt PII is stricter than body copy: international phonewords
   with bounded dial evidence and IDNA-equivalent email separators fail here,
   while detached numeric/art-direction prose remains admissible, including
   renderer nouns that trail an ambiguous candidate.
   Blank bodies/prompts fail closed via a shared `_deterministic_verdict`.
4. Proof: contract invariants both directions + real-entrypoint tests
   through `run_stage`.

### Review Contract

- Acceptance criteria:
  1. A worker asserting `ready_to_publish` on an overclaiming variant is
     rejected and NOTHING persists (runner recomputes, contract refuses).
  2. Clean variants persist with runner-computed verdicts and checklists;
     worker-supplied values are discarded.
  3. Image prompt text containing a banned claim or PII yields verdict
     `fail`; the verdict never carries the raw PII value.
  4. Contract invariants hold both ways: empty package, duplicate channel,
     blank body, and non-grammar advisory warnings are rejected; a
     not-ready package MAY carry a failing variant (legitimate
     intermediate state).
  5. Stage/schema mismatch still enforced for the new stages.
  6. Compatibility (round 8): `editorial_audit.v2` still validates
     unchanged and REJECTS the new fingerprint field; v1/v2 worker replies
     normalize to v3; contradictory `schema_version` still fails.
  7. Concurrency + atomicity (rounds 8-9): readiness is decided under the
     per-job lock and the lock is released afterwards; readiness reads the
     committed Git tree, so a failed/uncommitted artifact cannot authorize a
     later stage. An ordinary raised commit failure also restores the prior
     worktree bytes and index state.
  8. Dispatch binding (round 10): the committed draft fingerprint observed
     before the worker call must still match under the job lock before any
     audit, repurposing, or image-prompt response may persist, including an
     unready Phase 6 intermediate artifact.
- Reachability proof: `run_stage(job, "repurposing"|"image_prompt", ...)`,
  the same entrypoint the four existing stages use; artifacts land in the
  git-backed job folder.
- Affected surfaces: contracts, store stage map, store write path and
  locking, runner enforcement and audit normalization. The audit stage's
  persisted VERSION changes (v3); its decision behavior does not. No change
  to image generation.
- Risk areas: persisted-contract compatibility, PII false positives/negatives,
  and the per-job Git store's concurrency/crash boundary. No worker wrappers
  are wired to the Phase 6 stages yet (next slice), so those new stages cannot
  alter current pipeline behavior.
- Reviewer rules triggered: R1 (#2109 Phase 6), R2 (both-direction tests),
  R3 (gate cannot be self-reported), R5 (versioned audit compatibility),
  R6 (failed-write recovery), R8 (per-job serialization), R10 (one advisory
  grammar shared by three artifacts), R13 (class findings), R14.

### Decision-Seam Analysis

- **One decision:** `_phone_evidence` decides whether a mixed digit/letter
  candidate carries enough bounded mechanical evidence to be contact data
  rather than renderer prose.
- **Why the seam was wrong:** treating keypad-mappable spelling as sufficient
  evidence made `212 art deco` a phone number, while limiting the accepted
  structure to NANP membership missed explicit international phonewords.
  Earlier rounds also equated any attached letters with vanity and stopped
  before space-separated suffixes. These are both-direction failures at one
  admission seam, not separate spellings to enumerate.
- **Structural direction:** attached NANP vanity spelling is structural
  evidence. Detached spelling requires a preceding structural dial marker; an
  international spelling additionally requires an explicit `+`/`00` prefix and
  must map inside E.164's digit bound. No evidence defaults to ordinary renderer
  description. The oracle crosses renderer specifications, ordinary three-digit
  art directions, and trailing renderer nouns on the admit side, and separator
  partitions plus domestic/international prefixes on the reject side.

### Execution model

- **Selected closed-surface components:** Git's content-addressed committed
  tree is the canonical artifact source, and the operating system's `flock`
  supplies per-file mutual exclusion. Git closes the durable-read seam:
  worktree/index bytes do not exist to a canonical reader until a commit
  records them. `flock` closes the cooperative scheduling seam while the
  process is alive and the kernel releases it on process exit. Worker
  transport remains outside the lock; pre-dispatch snapshot/under-lock
  comparison closes that interval without holding a filesystem lock over a
  network call.
- **Admitted actors and invariant:** synchronous `run_stage` /
  `write_artifact` calls that use this store, possibly from multiple threads
  or processes, against one trusted local root. The lock identity is the
  resolved root plus job id. For every interleaving of those actors, only one
  call for that job may read committed readiness inputs and commit its result;
  a ready artifact therefore binds to the draft/audit pair from the committed
  tree observed inside its critical section. Source-derived stages also
  snapshot the committed draft immediately before dispatch and compare that
  snapshot under the lock before writing; a cooperative draft replacement
  during worker execution makes the response stale and unpersistable.
- **Failure boundary:** an ordinary raised Git failure restores the target's
  previous committed bytes and index entry. Abrupt process death or
  `BaseException` may leave mutable worktree/index residue, but the kernel
  releases the lock and subsequent readiness reads ignore that residue by
  reading Git `HEAD`. A later write to that stage overwrites the target from
  canonical input. Duplicate identical writes create no new commit;
  out-of-order ready writes fail the committed revision/fingerprint checks.
- **Assumptions:** the root, `.git` directory, and `.locks` directory are
  trusted same-user local state; all writers use this API; Git and POSIX
  `flock` are available; an external process that mutates the job repository
  while bypassing the lock is outside the model and unsupported.
- **Rejected component (hand-roll disclosure):** a database transaction would
  require replacing the accepted per-job Git artifact/audit surface with new
  schema and repository wiring. That is a storage migration, not a compatible
  primitive for this vertical slice. No lease, retry service, clock protocol,
  or cross-host coordinator is introduced.
- **One execution surface:** only the existing per-job Git artifact store is
  coordinated. Worker transport, image generation, and any future database
  storage remain outside this slice.

### Review round 1 (Codex)

Three findings, all fixed:

1. **P1 — orphan variants were representable.** `derived_from_claims` had a
   list default, so an omitted/empty lineage passed and a clean body could
   be marked `ready_to_publish`. Now `min_length=1` (required, non-empty),
   with negative coverage for omitted, empty, mixed, and blank-id lineage.
2. **MAJOR — the image gate failed on its second side.** `negative_prompt`
   was folded into the verified text, so naming a banned phrase in the
   EXCLUSION list -- the correct designer response to this module's own
   threat model -- tripped the gate. The verdict now covers `prompt_text`
   only; the negative prompt cannot cause rendering, so it cannot cause a
   failure. Both sides probed.
3. **MAJOR — the prompt set recorded a verdict nothing gated on.** Added
   `ready_to_generate`, the generation analogue of the package's
   `ready_to_publish`: it cannot be true while the deterministic verdict
   fails. A failing set still persists when not marked ready (legitimate
   intermediate state).

### Review round 2 (Codex)

Four findings, all fixed:

1. **P1 — lineage was non-blank, not REAL.** `derived_from_claims`
   accepted a fabricated id. `_enforce_lineage` now validates every cited
   id against the claim source ids in the job's draft artifact before a package may
   be `ready_to_publish`, and fails closed when the draft cannot be read.
   An unready package still skips the check (intermediate state).
2. **P1 — international phone PII passed the prompt gate.** `verify_copy`
   only fails on the US pattern (that scope was frozen deliberately in
   #2181). Prompt text now gets a stricter check -- an instruction about
   to be drawn into an image -- so `+44...` forms fail here without
   changing shared body-copy semantics.
3. **MAJOR — cross-item claim synthesis.** Prompts were joined before
   scanning, so "...guaranteed" + "savings..." across two prompts
   fabricated a hit. Each prompt is now verified independently and hits
   are prefixed with the offending prompt index.
4. **MAJOR — stale plan text.** The Scope section still described
   combined positive+negative gating; corrected above.

### Review round 3 (Codex)

Three findings, all fixed:

1. **P1 — the declared source revision was never checked.** A package could
   claim revision 1 against a revision-2 draft and ship whenever claim ids
   overlapped. `_enforce_lineage` now compares `source_draft_revision` with
   the draft on disk, for BOTH Phase 6 artifacts, before honouring either
   readiness flag.
2. **P1 — prose negation let banned words into renderer instructions.**
   "poster reading do not guarantee savings" passed, because the body-copy
   verifier treats that as a denial -- but the renderer still paints the
   words. Prompts now use `literal_claim_hits` (no negation suppression);
   body copy keeps prose semantics, so the #2181 contract is untouched.
3. **MAJOR — runner and schema disagreed on readiness.** The runner read raw
   dict truthiness, so a worker's `"false"` string looked ready while
   pydantic normalized it to False. The artifact is now validated first and
   the check branches on the normalized model value.

### Review round 4 (Codex)

Five findings, all fixed:

1. **P1 — readiness did not require an APPROVED draft.** Reading the draft
   proved it existed, not that anything cleared it. Readiness now requires
   the job's audit artifact to recommend `promote`, so unaudited or
   revise-state copy cannot ship or render.
2. **P1 — cross-project derivation.** A ready artifact could claim a draft
   from another project whenever revision and claim ids overlapped. The
   artifact's `project_id` must now match the draft's.
3. **P1 — the phone class was still open.** `0044 20 7946 0958` passed
   because only US and leading-plus forms were recognised. Replaced with a
   class-level rule: any run of 7+ digits under tolerant separators is
   contact-shaped. Ordinary short numbers in descriptions still pass.
4. **P1 — invisible-only text satisfied the non-blank invariant.** A
   zero-width space passed `NonEmptyStr` and earned a passing verdict.
   Added `VisibleStr` (rejects text whose characters are all Unicode C*/Z*
   categories) on the two fields that become shippable/renderable.
5. **MAJOR — the diff-budget justification was missing from the plan.**
   Added to "Why this slice exists" above, per the repository rule that it
   live in the plan rather than only in the commit message.

### Review round 5 (Codex)

Five findings, all fixed:

1. **P1 — the approving audit was not checked against the draft it
   approves.** A revision-1 audit authorized revision-2 copy (this repo's
   own test fixture created exactly that mismatch). The audit's
   `project_id` and `draft_revision` must now match the draft.
2. **P1 — a lone combining mark passed the visible-content rule.** U+FE0F
   is category Mn, so `VisibleStr` accepted it. M* now joins C*/Z* as
   non-standalone; real copy carrying a mark (emoji + VS16) still passes.
3. **P1 — the prompt phone check was wrong in BOTH directions.**
   `1-800-FLOWERS` passed (letters, so digit counting never saw it) while
   `calendar showing 2026-07-25` failed (a date is not a phone number).
   Replaced digit-counting with candidate -> reject known non-contact
   shapes (ISO/US dates, clock times) -> require dialable evidence (7-15
   digits, E.164 bound) plus an explicit vanity-number rule.
4. **P1 — internationalized email escaped the ASCII pattern.**
   `josé@example.com` and `user@例え.テスト` passed. The prompt path now
   uses a script-independent address shape.
5. **MAJOR — canonically equivalent channels were not duplicates.** NFC
   and NFD "cafe" both validated in one package; channels are NFKC-
   normalized before casefolding.

**Convergence note.** Phone detection has now been reworked in rounds 3, 4
and 5, and visible-text in rounds 4 and 5 -- the same non-convergent
pattern as #2181's advisory engine. The classifiers are now written as
two-directional decisions with both error sides pinned by parametrized
probes, which is the shape that finally held there.

### Review round 6 (Codex) — classifier redesign

One finding, and it named the real problem: the phone check was still a
"reject known-bad, admit the rest" enumeration. Evidence: `1-800-GOT-JUNK`
passed (hyphenated vanity spelling) while `RGB palette 255 255 255` was
rejected as PII.

**Redesigned as an EVIDENCE-GATED decision.** Rounds 3-6 each failed
because the rule asked "do these digits look like a phone number?", which
has no closed answer -- dates, RGB triples, dimensions and dialable numbers
share a digit grammar. It now requires positive evidence of contact intent,
of which there are exactly two kinds:

* **structural** — an unambiguous dialable form (E.164 `+`/`00` prefix, or
  the NANP 3-3-4 shape). Nothing describes artwork this way, so these fail
  with no other context.
* **lexical** — a dial-intent verb (call/text/dial/ring/reach...) near a
  dialable token. This is what makes "Call 1-800-GOT-JUNK" contact data
  when its digits alone are not.

Absent both, digits are description and the prompt passes. Token
continuation groups are restricted to digits or uppercase runs so a token
cannot swallow the following lowercase word.

**Generative oracle** replaces the hand-listed fixtures: 6 intents x 11
dialable forms (66 cases) must fail; 11 descriptive strings x 3 scene
templates (33 cases) must pass; structural forms must fail without any
intent word; and 4 email forms across scripts must fail. The hand-listed
phone tests from rounds 4-5 were removed as superseded.

### Review round 7 (Codex)

Two findings, both fixed:

1. **P1 — vanity recognition was casing-dependent.** `1-800-flowers`
   passed because continuation groups accepted uppercase only. The real
   distinction is ATTACHMENT, not casing: hyphen/dot-joined letters belong
   to the number (any case), a space-joined lowercase word is the next
   word. Oracle extended across casing modifiers, plus the other side
   (trailing words must not extend the token past the E.164 bound).
2. **P1 — approval was bound to a mutable revision label.** Rerunning the
   draft stage replaces the body while keeping revision 1, and the old
   audit still counted as approval. Approval now binds to draft CONTENT:
   the runner stamps `source_draft_fingerprint` (SHA-256 of the persisted
   draft bytes) onto the audit and verifies it at readiness. Worker-supplied
   values are overwritten, same discipline as the verdict.

### Review round 8 (Codex)

Three findings, all fixed. All three are defects this PR introduced, not
inherited: the read-validate-write sequence and the fingerprint field both
arrived in round 7, and the prompt classifier is Phase 6's own.

1. **P1 — the fingerprint check was not atomic with the write.** Round 7
   verified the draft fingerprint and then persisted; a concurrent draft
   rerun inside that window landed a `ready_to_publish` artifact beside copy
   the audit never covered. Validating before writing proves nothing when the
   thing validated against can move. `job_lock` (re-entrant per thread,
   `flock` across processes, lock file outside the job's git folder) now
   spans stamping, readiness and the commit.
2. **P1 — the fingerprint field silently rebroke the published v2 contract.**
   `editorial_audit.v2` shipped in #2181 and forbids extras, so adding a key
   to it made every newly written audit UNREADABLE to a v2 consumer or a
   rolled-back reader — the exact breakage v1's freeze note exists to
   prevent. Introduced `editorial_audit.v3` (subclassing v2 so the promote
   gate and warning grammar cannot drift), re-froze v2, and generalized the
   runner's normalizer to upgrade v1/v2 -> v3 while still refusing to launder
   self-contradictory worker metadata.
3. **P2 — dial intent was searched globally.** `a person calling across a
   room, RGB palette 255 255 255` was rejected as contact PII.

   Codex proposed scoping intent to a bounded candidate context. That does
   not work, and the counter-example is worth recording: in `a call center
   scene, 1920 1080 resolution` the gap from "call" to "1920" is two words —
   exactly the gap in `call me, 5551234567`. No window separates them.

   Replaced with a SHAPE-FIRST tier split. Unambiguous shapes (E.164, NANP
   3-3-4, [3,4] local, trunk prefix, vanity, and syntactically valid NANP
   runs) fail with no intent required. `unbroken` runs need a dial verb
   within 3 tokens, which may cross a comma. `grouped` shapes ([3,3,3] RGB,
   [4,4] resolution, [4,2,2] dates) need one within 2 tokens with no boundary
   crossed — which is what keeps compound nouns ("call center", "phone
   booth", "call sheet") out without enumerating any of them.

**Documented residual (deliberate, not missed):** an unbroken digit run that
is not a valid NANP number is shape-identical to a serial, and an earlier
round established that `serial 12345678 engraved on a plate` must render. So
that form fails only once a dial verb governs it. Shape alone cannot decide
between the two; NANP's own constraint (neither area nor exchange code may
start with 0 or 1) recovers the part that can be decided.

**Scope note:** this round added a locking primitive and a contract version
to what began as a contracts slice. Both are the hardened fix for defects
this PR introduced, so they belong here rather than in a follow-up.

### Review round 9 (Codex)

Four findings, all verified against the code before acting and all fixed.

1. **P1 — the vanity rule was too loose one way and too tight the other.**
   Treating ANY attached letter group as vanity rejected renderer specs:
   `16-bit-color`, `1920-1080-pixel`, `8-bit-style` and `32-bit-float` all
   failed (the last two were not in the report; found by probing the class).
   Meanwhile the separator grammar stops before a SPACE-joined letter group,
   so `Call 1 800 FLOWERS today` passed with `ready_to_generate=true`.

   Root fix, from the numbering plan rather than a word list: letters in a
   vanity number are DIGIT SUBSTITUTES, so the token must (a) lead with an
   area code -- exactly 3 digits, or "1" plus 3 -- and (b) map through the
   phone keypad to a syntactically valid NANP number. A spec's leading group
   is 2 or 4 digits, so the whole class renders. Space-joined suffixes extend
   only while the NANP symbol-count bound remains reachable, so every possible
   partition of a seven-letter suffix is covered without an arbitrary
   word-count cap.

2. **P1 — a failed commit left artifact residue the readiness gate trusted.**
   `write_artifact` wrote and staged the file before committing, so a commit
   failure left the job's modified draft artifact in the working tree with no
   commit recording it -- and the readiness gate read the working tree.
   Readiness now reads committed Git objects only, so uncommitted residue
   cannot become source state. Ordinary raised failures also restore the
   previous committed bytes/index entry; process-death residue remains
   non-canonical and is ignored.

3. **P2 — the governing contract did not cover round 8's changes.** The
   Problem-derived and Review Contracts still said "must not change existing
   contracts/stages" while the diff upgraded audits to v3 and added store
   locking. Both sections now carry those changes with their own acceptance
   criteria, rather than leaving them justified only by narrative.

4. **P2 — `channel` was non-blank but not VISIBLE.** A label made only of
   U+200B/U+200C/U+FE0F validated and persisted unroutable, and distinct
   invisible spellings evaded the duplicate check. `channel` is now
   `VisibleStr`, and duplicate detection compares on a routing key that drops
   format/control characters, so `email` and `email<ZWSP>` collide.

Both P1 fixes are mutation-checked: reverting either makes the new tests
fail, so they are load-bearing rather than vacuous. The execution model above
records the current-main `AGENTS.md` 3k.4 boundary.

### Review round 10 (Codex)

Five findings, all verified against the code before acting and fixed at their
shared decision/execution seams.

1. **P1 — international phonewords bypassed a NANP-only decision.** An
   explicit `+`/`00` prefix plus dial intent and keypad-mappable spelling now
   admits a bounded 7-15 digit international candidate without pretending it
   belongs to NANP.
2. **P2 — detached prose was treated as vanity spelling.** Space-joined alpha
   extensions no longer prove contact data by themselves. They require nearby
   dial intent, so ordinary three-digit renderer prose such as `room 212 art
   deco sign` remains admissible. Attached NANP vanity syntax stays structural
   evidence.
3. **P2 — default-ignorable marks split routing identities.** The channel
   routing key now drops the Unicode default-ignorable class, including
   variation selectors and combining grapheme joiners, as well as format and
   control characters. Visible labels that differ only by those marks collide.
4. **P1 — post-worker fingerprinting attested to the wrong draft.** The runner
   now snapshots committed draft identity before dispatch and compares it
   under the per-job lock before persisting any source-derived response.
   Audit, repurposing, and image-prompt artifacts all reject a same-revision
   replacement that lands while their worker is running.
5. **P1 — IDNA-equivalent email separators bypassed the prompt gate.** Email
   admission normalizes U+3002, U+FF0E, and U+FF61 to the ASCII domain-label
   separator before matching; findings remain redacted.

The phone oracle covers both directions from grammars: domestic separator
partitions and international prefix/spelling combinations reject under dial
evidence, while three-digit values crossed with art-direction phrases admit.
The dispatch tests replace the committed draft inside the worker boundary, so
moving the snapshot back after dispatch makes them fail.

### Review round 11 (Codex, follow-up PR #2201)

Two findings, both verified against the published follow-up before acting and
fixed at the generating operation rather than at the reported strings.

1. **P1 — removing ignorables after NFKC left composition-sensitive aliases.**
   A combining grapheme joiner between a base and combining mark could block
   canonical composition before being removed, so `é` and
   `e<U+034F><U+0301>` remained distinct routing keys. Routing now removes the
   full default-ignorable/format/control class before NFKC. Generated proofs
   cross four canonical compositions with four ignorable marks.
2. **P1 — the numeric separator grammar accepted one whitespace code point.**
   Extra formatting whitespace split an international phoneword before its
   explicit prefix and keypad spelling could be evaluated together. Numeric
   groups now consume a whitespace run as one separator. The existing group
   cap, compact E.164 symbol bound, explicit international prefix, and bounded
   intent gate remain unchanged; generated proofs vary prefixes, whitespace
   classes, and run widths, with detached prose proving the passing side.

### Review round 12 (Codex, follow-up PR #2201)

Three findings, all verified against the published follow-up and fixed at the
shared parsing, evidence, and source-construction boundaries.

1. **P1 — compatibility/separator spelling changed the phone verdict.** Phone
   admission now classifies an NFKC-normalized view and shares one dial
   separator grammar across tokenization, splitting, and compaction. Generated
   proofs cross compatibility forms and separator choices, including slash,
   against the same semantic oracle.
2. **P1 — proximity to an open intent vocabulary was not structural evidence.**
   Detached phonewords now require a finite bridge between the dial marker and
   candidate. The oracle crosses direct/functional bridge evidence against
   descriptive intervening words; ambiguous prose defaults to admissible.
3. **P1 — the pre-dispatch fingerprint did not bind caller-built prompt text.**
   Source-bound stages now take a prompt builder. `run_stage` reads committed
   draft bytes once, builds the prompt from their parsed document, and hashes
   those same bytes before dispatch. The existing under-lock re-read still
   rejects a replacement during the worker call.

### Review round 13 (Codex, follow-up PR #2201)

Three findings, all verified against the published follow-up and corrected at
the same classifier and source-construction seams.

1. **P1 — slash-delimited numeric renderer values entered the unconditional
   E.164 shortcut.** Compact explicit-prefix digits remain strong evidence;
   slash-delimited numeric shapes now pass through the structural-intent
   decision. Generated proofs keep slash phonewords and structurally governed
   numbers failing while renderer dimensions and dates remain admissible.
2. **P1 — line-break formatting split direct dial syntax.** The finite bridge
   now admits exactly one logical LF, CR, or CRLF while continuing to reject
   paragraph breaks and descriptive intervening words.
3. **P1 — an arbitrary prompt callback could ignore its draft argument.**
   Source-bound stages now accept no caller prompt payload. The runner owns the
   fixed stage instruction and deterministic serialization of the committed
   draft snapshot, then hashes the same source bytes and retains the under-lock
   comparison.

### Review round 14 (Codex, follow-up PR #2201)

Three findings, all verified against the published follow-up and corrected at
the same classifier and source-construction seams.

1. **P1 - trailing renderer nouns were reverse dial intent.** Ambiguous
   detached candidates now require structural dial evidence before the
   candidate; renderer nouns such as `contact sheet` and `text treatment` after
   `212 art deco` remain admissible prose.
2. **P1 - parenthesized phoneword groups missed the shared separator grammar.**
   Parentheses are parsed by the bounded dial grammar, so `Call +44 (800)
   FLOWERS` receives the same verdict as its unparenthesized and numeric
   keypad-equivalent forms.
3. **P1 - a missing committed draft became JSON null.** Source-bound stages now
   require a valid committed draft before constructing the runner-owned prompt
   or dispatching the worker.

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

Two artifacts, two stages, one enforcement idea reused: the deterministic
verdict is always computed by the runner from the artifact's own text and
overwrites whatever the worker claimed. Variants get a per-variant verdict
because each ships independently; image prompts are checked independently and
their redacted hits aggregate into the prompt-set verdict. Readiness consumes
committed Git objects inside one per-job critical section, and source identity
is snapshotted before worker dispatch then rechecked inside that section, so
mutable residue or a mid-worker draft replacement cannot become approval input.
Source-bound prompts require a committed draft object before dispatch; missing
source is not a serializable stage input.

## Intentional

- **Per-variant verdicts, not one package verdict** -- variants ship
  separately, so a failing LinkedIn variant must not block a clean X
  variant from being fixed independently.
- **`ready_to_publish` is a package-level gate** requiring ALL variants to
  pass: partial shipping is a human decision, not a model's.
- **Image prompts are gated on text** because a diffusion model renders
  banned copy INTO the image, past every downstream text check.
- **The offending prompt/body still persists** on a failing artifact --
  the verdict is redacted, but the payload must stay visible or a human
  cannot fix it (same as an existing draft body).
- **No generation, no worker wrappers here** -- generation is
  human-triggered and VRAM-guarded, and on this box a 30B model (~20 GB)
  plus ComfyUI FLUX (~12 GB) exceeds the 24 GB card, so the generator
  slice must handle model eviction explicitly.

## Deferred

- Image generation via the ComfyUI MCP (human-triggered, VRAM-guarded,
  must address LLM/FLUX co-residency on a 24 GB card).
- OWUI worker wrappers for the two new stages (`cf-repurposer`,
  `cf-image-prompt`) with §8 tool scoping.
- Phase 7 manifest entries for the new stages.

Parked hardening: none new.

## Verification

    python -m pytest tests/test_content_factory_schemas.py \
        tests/test_content_factory_runner.py \
        tests/test_content_factory_store.py \
        tests/test_content_factory_copy_verification.py \
        tests/test_leads_intake.py -q
    # -> 956 passed (incl. the round-6 contact oracle, round-7
    #    content-binding probes, round-8 descriptive-number boundary, and
    #    rounds 9-10 separator-partition, international/detached-prose,
    #    IDNA, dispatch-binding, and committed-residue proofs, plus round-11
    #    canonical-composition/whitespace-run proofs and round-12 normalized
    #    parser and structural-bridge proofs, round-13 slash/line-break
    #    boundaries and runner-owned prompt-construction proofs, and round-14
    #    parenthesized phonewords, trailing renderer nouns, and missing-source
    #    pre-dispatch proofs)
    #
    # Mutation check on the round-8 lock test: removing `with job_lock(...)`
    # from run_stage makes test_run_stage_holds_job_lock_across_validation_
    # and_write FAIL, so the assertion is load-bearing rather than vacuous.
    #
- Ruff, `python -m py_compile`, and `git diff --check` passed on the touched
  Python/diff surface.
- Guard class-closure advisory and plan shape/files/diff-size/rule audits
  passed against current `origin/main`.
- NOT run: live worker pass (no wrappers wired yet, by design).

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/schemas/content_factory.py` | 60 |
| `atlas_brain/services/content_factory_runner.py` | 365 |
| `atlas_brain/services/content_factory_store.py` | 102 |
| `plans/PR-CF-Phase6-Repurposing-Contracts.md` | 285 |
| `plans/PR-CF-Phase6-Round10-Remediation.md` | 296 |
| `tests/test_content_factory_runner.py` | 686 |
| `tests/test_content_factory_schemas.py` | 68 |
| **Total** | **1862** |
