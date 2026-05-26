# PR: Support-Ticket Blog Small Upload Trim

## Why this slice exists

PR #985 proved the live support-ticket blog path can save a source-truthful
descriptive draft from the packaged CSV. That validation also parked a product
polish issue: tiny support-ticket uploads can still produce long, repetitive
articles because the prompt's support-ticket descriptive path allows several
overlapping sections while the general blog rule still asks for 1500-2200 words.

This slice promotes that parked item and fixes the prompt contract at the
source, so small no-outcome/no-resolution support-ticket uploads produce compact
review drafts instead of expensive, repetitive long-form posts.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-provider-blog-polish
Slice phase: Product polish

1. Tighten the host blog prompt's no-outcome/no-resolution support-ticket path
   for small uploaded-ticket batches.
2. Keep the extracted blog prompt copy in sync with the host prompt.
3. Add a prompt-contract test so future edits keep the small-upload guidance in
   both prompt copies.
4. Remove the promoted parked item from `ATLAS-HARDENING.md`.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Support-Ticket-Blog-Small-Upload-Trim.md` | Plan doc for this polish slice. |
| `atlas_brain/skills/digest/blog_post_generation.md` | Tighten small-upload support-ticket descriptive blog guidance. |
| `extracted_content_pipeline/skills/digest/blog_post_generation.md` | Keep extracted skill prompt aligned with the host copy. |
| `tests/test_atlas_content_ops_infrastructure.py` | Pin the prompt contract in both prompt copies. |
| `ATLAS-HARDENING.md` | Remove the promoted parked item after fixing it. |

## Mechanism

The prompt keeps the existing truthfulness contract: uploaded tickets can show
repeated questions and FAQ opportunities, but cannot prove outcomes or concrete
answers without explicit evidence. This slice adds a narrower shape for small
uploads when both measured outcomes and resolution evidence are absent:

- use a compact descriptive article, not the default long-form blog length
- cap the structure to a small number of H2 sections
- avoid repeating the same cluster explanation across multiple sections
- keep draft FAQ answers as verification placeholders

The test reads the host and extracted prompt files directly and asserts both
copies include the same small-upload length and section guidance.

## Intentional

- This does not add another generated-content detector. The current issue is
  output ergonomics, not a missed truthfulness blocker.
- This does not rerun live Haiku. The previous live run already proved the route
  saves truth-safe content; this slice pins the deterministic prompt contract and
  avoids another model-cost validation run for a copy-shape change.
- This leaves large support-ticket uploads on the normal article path. The
  compact rule is only for small uploads where the source material cannot
  support a long article without repetition.

## Deferred

- A future live validation slice can re-run the packaged CSV and record actual
  token/length behavior after this prompt change if we want model-output proof.
- Broader product work to promote customer language into generated blog/landing
  keywords and to define standalone FAQ Article output remains outside this PR.
- Parked hardening: none added. This PR promotes and closes `Support-ticket
  descriptive blog output is long and repetitive on tiny uploads`.

## Verification

- Command: python -m pytest tests/test_atlas_content_ops_infrastructure.py -q
  - `12 passed`
- Command: python -m pytest tests/test_extracted_blog_generation.py tests/test_evaluate_support_ticket_generated_content.py -q
  - `99 passed`
- Command: bash scripts/validate_extracted_content_pipeline.sh
  - passed
- Command: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
  - passed
- Command: python scripts/audit_extracted_standalone.py --fail-on-debt
  - passed
- Command: bash scripts/check_ascii_python.sh
  - passed
- Command: bash extracted/_shared/scripts/sync_extracted.sh extracted_content_pipeline
  - passed; extracted copy unchanged beyond the intended prompt update
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/pr-support-ticket-blog-small-upload-trim-body.md
  - passed

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~80 |
| Prompt updates | ~20 |
| Test coverage | ~20 |
| Hardening cleanup | ~10 |
| **Total** | **~130** |
