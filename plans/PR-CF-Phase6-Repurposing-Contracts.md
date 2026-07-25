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

**Diff-budget overage (~1,190 LOC vs the 400 soft cap) — why this slice is
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

Roughly 60% of the LOC is tests (four review rounds of adversarial
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
- Must not change: existing contracts/stages, the editorial gate's
  behavior, the advisory grammar (extracted to a shared validator, same
  rules), or anything about image generation (separate slice).

## Scope (this PR)

Ownership lane: content-factory
Slice phase: vertical slice

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
   claims). Prompt PII is stricter than body copy: international phone
   forms fail here.
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
- Reachability proof: `run_stage(job, "repurposing"|"image_prompt", ...)`,
  the same entrypoint the four existing stages use; artifacts land in the
  git-backed job folder.
- Affected surfaces: contracts, store stage map, runner enforcement. No
  change to existing stages or to image generation.
- Risk areas: none live -- no worker wrappers are wired to these stages
  yet (next slice), so this cannot alter current pipeline behavior.
- Reviewer rules triggered: R1 (#2109 Phase 6), R2 (both-direction tests),
  R3 (gate cannot be self-reported), R5 (no existing-stage behavior
  change), R10 (one advisory grammar shared by three artifacts), R14.

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

### Files touched

- `atlas_brain/schemas/content_factory.py`
- `atlas_brain/services/content_factory_copy_verification.py`
- `atlas_brain/services/content_factory_runner.py`
- `atlas_brain/services/content_factory_store.py`
- `plans/PR-CF-Phase6-Repurposing-Contracts.md`
- `tests/test_content_factory_runner.py`
- `tests/test_content_factory_schemas.py`

## Mechanism

Two artifacts, two stages, one enforcement idea reused: the deterministic
verdict is always computed by the runner from the artifact's own text and
overwrites whatever the worker claimed. Variants get a per-variant verdict
because each ships independently; the prompt set gets one verdict over all
prompt text because it is rendered as a unit.

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
    # -> 523 passed (214 new; incl. the round-6 generative oracle, the
    #    round-7 casing/content-binding probes, and the round-8
    #    descriptive-numbers x dial-words class-closure oracle)
    #
    # Mutation check on the round-8 lock test: removing `with job_lock(...)`
    # from run_stage makes test_run_stage_holds_job_lock_across_validation_
    # and_write FAIL, so the assertion is load-bearing rather than vacuous.
    #
    # Repo-wide: 20155 passed / 169 failed / 8 collection errors. Every
    # failure and error reproduces identically on a stashed (clean) tree --
    # they are pre-existing local dependency issues in invoicing / mcp /
    # scheduler / reasoning, and none are in a file this diff touches.

- `python -m py_compile` clean (SyntaxWarning as error) on touched modules.
- NOT run: live worker pass (no wrappers wired yet, by design).

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/schemas/content_factory.py` | 246 |
| `atlas_brain/services/content_factory_copy_verification.py` | 20 |
| `atlas_brain/services/content_factory_runner.py` | 496 |
| `atlas_brain/services/content_factory_store.py` | 71 |
| `plans/PR-CF-Phase6-Repurposing-Contracts.md` | 388 |
| `tests/test_content_factory_runner.py` | 834 |
| `tests/test_content_factory_schemas.py` | 269 |
| **Total** | **2324** |
