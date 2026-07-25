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
    # -> 351 passed (42 new: 18 contract invariants, 24 run_stage gates)

- `python -m py_compile` clean (SyntaxWarning as error) on touched modules.
- NOT run: live worker pass (no wrappers wired yet, by design).

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/schemas/content_factory.py` | 159 |
| `atlas_brain/services/content_factory_copy_verification.py` | 20 |
| `atlas_brain/services/content_factory_runner.py` | 193 |
| `atlas_brain/services/content_factory_store.py` | 4 |
| `plans/PR-CF-Phase6-Repurposing-Contracts.md` | 209 |
| `tests/test_content_factory_runner.py` | 396 |
| `tests/test_content_factory_schemas.py` | 206 |
| **Total** | **1187** |
