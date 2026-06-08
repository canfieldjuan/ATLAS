# PR-Gate-A-Live-Convergence-Proof

## Why this slice exists

The #1360 Gate A live output-quality proof failed on the support-ticket SaaS
demo payload: blog variants collapsed, one blog variant was blocked, landing
variants were not meaningfully distinct enough, sales briefs drifted from the
requested brief type, and brand voice missed second-person in some outputs.

The fix wave has since landed: blog persistence, sales-brief type lock/prompt
steering, second-person brand voice, blog publishable-prose prevention,
landing-page variant distinctness, and the #1378 harness `--outputs` selector.
The operator explicitly assigned the next step in #1378: run the live
convergence proof for `landing_page`, `blog_post`, and `sales_brief`, commit
the generated artifacts, and let the reviewer judge the real samples.

The first #1383 proof run then surfaced an actionable gap: the support-ticket
blog debug-source detector missed the exact live phrase "rows were included
for generation", and the live smoke seed still exposed that debug wording in
the source summary. The same review cycle also exposed a second-person sales
brief instability where the shared brand-voice prompt was too implicit. This
PR now closes those narrow gates and refreshes the proof artifacts.

This PR may exceed the 400 LOC soft cap because the raw JSON outputs are the
point of the slice. The artifacts are indivisible evidence, while the code
changes are the minimum needed to make the proof exercise the intended gates.

## Scope (this PR)

Ownership lane: content-ops/gate-a-output-quality
Slice phase: Functional validation

1. Run `scripts/smoke_content_ops_gate_a_live_quality.py` against the real
   local database/model route with `--outputs landing_page,blog_post,sales_brief`.
2. Disable local Ollama fallback with
   `EXTRACTED_CAMPAIGN_LLM_AUTO_ACTIVATE_OLLAMA=false` and use the configured
   Sonnet route from `.env` / `.env.local`.
3. Commit the exact run artifacts under `docs/extraction/validation/fixtures/`
   plus a markdown report with the command, resolved model, structural result,
   and sample pointers.
4. Harden the support-ticket blog debug-source detector for the exact live
   phrase and remove debug "included for generation" wording from the smoke
   seed blueprint.
5. Make the shared second-person brand-voice prompt explicit enough for
   sales-brief generation to satisfy the existing enforcement gate.
6. Do not self-certify product acceptance. The reviewer owns the GOOD-bar
   judgment against the exported drafts.

### Review Contract

- Acceptance criteria:
  - [ ] The committed command includes
        `--outputs landing_page,blog_post,sales_brief`.
  - [ ] The artifact set includes `execution-result.json`,
        `review-results.json`, `summary.json`, and per-output exports for each
        selected output that the harness produced.
  - [ ] The report records the resolved model route and that local Ollama
        fallback was disabled.
  - [ ] The report surfaces real samples and structural results without
        claiming product pass/fail beyond the harness result.
  - [ ] The run excludes `email_campaign` so its unverified input-fit cannot
        mask the three-generator convergence signal.
  - [ ] The refreshed run proves the bad blog prose candidate is blocked with
        `support_ticket_generated_content:debug_source_narration`.
  - [ ] Tests cover the exact live debug phrases and allowed near-misses.
- Affected surfaces: validation artifacts, support-ticket blog generation
  gates, support-ticket live-smoke seed copy, shared brand-voice prompt
  guidance, and focused tests.
- Risk areas: accidental local model fallback, missing artifacts, conflating
  structural harness `ok` with human product acceptance, over-broad debug
  prose detection, and weakening brand-voice enforcement instead of improving
  prompt compliance.
- Reviewer rules triggered: R1, R2, R10.

### Files touched

- `docs/extraction/validation/content_ops_gate_a_reconfirm_2026-06-08.md`
- `docs/extraction/validation/fixtures/content_ops_gate_a_reconfirm_20260608/execution-result.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_reconfirm_20260608/export-blog_post.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_reconfirm_20260608/export-landing_page.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_reconfirm_20260608/export-sales_brief.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_reconfirm_20260608/review-results.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_reconfirm_20260608/summary.json`
- `extracted_content_pipeline/blog_generation.py`
- `extracted_content_pipeline/brand_voice.py`
- `plans/PR-Gate-A-Live-Convergence-Proof.md`
- `scripts/smoke_content_ops_live_generation.py`
- `tests/test_extracted_blog_generation.py`
- `tests/test_extracted_brand_voice.py`
- `tests/test_extracted_sales_brief_generation.py`
- `tests/test_smoke_content_ops_live_generation.py`

## Mechanism

The script executes the real Content Ops service builder, persists generated
drafts to the configured Postgres database, reviews the saved ids, exports the
reviewed drafts exactly, and writes JSON artifacts to the selected output
directory.

This slice runs the script from the PR worktree with absolute `--env-file`
paths to the primary checkout's existing `.env` and `.env.local`. That avoids
copying secrets into the worktree while keeping the run equivalent to a clean
main checkout with those env files present.

After the run, the artifacts are copied into
`docs/extraction/validation/fixtures/content_ops_gate_a_reconfirm_20260608/`
and summarized in
`docs/extraction/validation/content_ops_gate_a_reconfirm_2026-06-08.md`.

Review-response mechanics:

- `extracted_content_pipeline/blog_generation.py` expands the existing
  support-ticket debug-source narration patterns to catch the exact live
  phrases from review, while paired tests preserve allowed near-misses.
- `scripts/smoke_content_ops_live_generation.py` rewrites the seeded blog
  blueprint summaries so the model no longer sees "rows were included for
  generation" or "uploaded ticket CSV can produce" as source copy to repeat.
- `extracted_content_pipeline/brand_voice.py` keeps the existing audit/blocker
  contract but makes second-person prompt guidance explicit (`you` / `your`)
  before sales-brief generation, so compliant drafts are more likely to pass
  the already-enforced gate.

## Intentional

- Exclude `email_campaign` from this convergence run. #1378 made it selectable,
  but its support-ticket input-fit is intentionally separate from this
  three-generator proof.
- Use Sonnet via the configured cloud/OpenRouter route, not Haiku and not local
  Ollama, so the run compares the fix wave against the #1360 baseline without
  changing the model variable.
- Do not write a product-quality verdict. The report can state structural
  counts and sample excerpts, but the GOOD-bar judgment remains reviewer-owned.
- The refreshed live run still permits a blocked variant. The blocked
  `pain_led` blog candidate is evidence that the #1373-style gate fired, not a
  saved customer-facing draft.

## Deferred

- Separate `--outputs email_campaign` live validation if the operator wants to
  prove campaign input-fit on the support-ticket payload.
- Separate Haiku / cheaper-model ship-readiness pass after the Sonnet
  convergence signal is reviewed.
- Landing-page prose-prevention and section dedup remain separate product
  hardening if the reviewer wants those rough edges promoted after judging the
  refreshed proof.
- Same-lane `HARDENING.md` items considered:
  - "Blog output uses debug-style source narration instead of publishable
    prose" was promoted into this PR and addressed by the detector/source-seed
    hardening plus refreshed proof artifacts.
  - "Brand-voice second-person guidance is not consistently honored" was
    promoted for the sales-brief instability found during this review and
    addressed with explicit second-person prompt guidance.
  - "Landing-page variants pass audits but are not meaningfully distinct" was
    addressed by the already-landed #1375 fix and rechecked here with three
    exported landing variants; the separate section-title prose and section
    dedup rough edges stay parked for a future landing polish slice.
  - "Gate A needs a messy-ticket grounding rerun" stays parked because this
    proof intentionally reruns the clean SaaS fixture that failed in #1360 to
    isolate fix-wave convergence before changing the input distribution.

Parked hardening:

- `HARDENING.md` - Gate A needs a messy-ticket grounding rerun.
- Landing-page section-title prose prevention and section dedup from the #1383
  reviewer notes.

## Verification

- `EXTRACTED_CAMPAIGN_LLM_AUTO_ACTIVATE_OLLAMA=false python scripts/smoke_content_ops_gate_a_live_quality.py --account-id 2b2b950d-f64b-4852-bc30-f92a34cdf169 --user-id 11111111-1111-4111-8111-111111111111 --support-ticket-csv extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv --env-file /home/juan-canfield/Desktop/Atlas/.env --env-file /home/juan-canfield/Desktop/Atlas/.env.local --output-dir tmp/content_ops_gate_a_reconfirm_20260608 --outputs landing_page,blog_post,sales_brief --variant-count 3 --quality-repair-attempts 1 --max-cost-usd 20.00 --json` - PASS (`summary.json` reports `ok=true`; selected outputs only).
- `python -m pytest tests/test_extracted_blog_generation.py tests/test_extracted_brand_voice.py tests/test_extracted_sales_brief_generation.py tests/test_smoke_content_ops_live_generation.py tests/test_smoke_content_ops_gate_a_live_quality.py -q` - PASS (193 passed).
- `for f in docs/extraction/validation/fixtures/content_ops_gate_a_reconfirm_20260608/*.json; do python -m json.tool "$f" >/dev/null || exit 1; done` - PASS.
- `jq` model-route check over `docs/extraction/validation/fixtures/content_ops_gate_a_reconfirm_20260608/export-blog_post.json`, `docs/extraction/validation/fixtures/content_ops_gate_a_reconfirm_20260608/export-landing_page.json`, and `docs/extraction/validation/fixtures/content_ops_gate_a_reconfirm_20260608/export-sales_brief.json` with filter `map([.. | objects | .generation_model? // empty]) | add | unique` - PASS (`anthropic/claude-sonnet-4-5`).
- `rg` check for known bad phrases in `docs/extraction/validation/fixtures/content_ops_gate_a_reconfirm_20260608/export-blog_post.json` - PASS (no matches for `rows were included for generation`, `uploaded ticket CSV can produce`, or `Your uploaded tickets contain`).
- `bash` `scripts/validate_extracted_content_pipeline.sh` - PASS.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` - PASS.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - PASS.
- `bash` `scripts/check_ascii_python.sh` - PASS.
- `bash` `scripts/run_extracted_pipeline_checks.sh` - PASS (`extracted_reasoning_core`: 295 passed; `extracted_content_pipeline`: 3370 passed, 10 skipped, 1 known `pynvml` warning).
- `bash scripts/push_pr.sh tmp/gate-a-live-convergence-proof-pr-body.md --force-with-lease origin claude/pr-gate-a-live-convergence-proof` - runs the managed pre-push hook once before pushing.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/extraction/validation/content_ops_gate_a_reconfirm_2026-06-08.md` | 124 |
| `docs/extraction/validation/fixtures/content_ops_gate_a_reconfirm_20260608/execution-result.json` | 433 |
| `docs/extraction/validation/fixtures/content_ops_gate_a_reconfirm_20260608/export-blog_post.json` | 831 |
| `docs/extraction/validation/fixtures/content_ops_gate_a_reconfirm_20260608/export-landing_page.json` | 1258 |
| `docs/extraction/validation/fixtures/content_ops_gate_a_reconfirm_20260608/export-sales_brief.json` | 367 |
| `docs/extraction/validation/fixtures/content_ops_gate_a_reconfirm_20260608/review-results.json` | 39 |
| `docs/extraction/validation/fixtures/content_ops_gate_a_reconfirm_20260608/summary.json` | 214 |
| `extracted_content_pipeline/blog_generation.py` | 8 |
| `extracted_content_pipeline/brand_voice.py` | 16 |
| `plans/PR-Gate-A-Live-Convergence-Proof.md` | 200 |
| `scripts/smoke_content_ops_live_generation.py` | 6 |
| `tests/test_extracted_blog_generation.py` | 11 |
| `tests/test_extracted_brand_voice.py` | 1 |
| `tests/test_extracted_sales_brief_generation.py` | 33 |
| `tests/test_smoke_content_ops_live_generation.py` | 7 |
| **Total** | **3548** |
