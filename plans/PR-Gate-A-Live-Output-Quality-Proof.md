# PR-Gate-A-Live-Output-Quality-Proof

## Why this slice exists

Issue #1357 stops net-new feature work until the recent brand-voice and
output-variation burst is proven on real data, real model output, and real
Postgres persistence. The last several slices added brand voice, variants, and
multi-output generation behavior, but the dated validation folder has no
06-06/06-07 artifact proving the combined path produces content a marketer
would actually ship.

This is a functional-validation slice. It does not add product behavior. It
adds the missing live proof: an adversarial support-ticket run with inline brand
voice, `variant_count > 1`, `blog_post + landing_page + sales_brief`, generated
asset review, export, and a human quality verdict that is allowed to fail.

The diff may exceed the 400 LOC soft cap because issue #1357 explicitly
requires the actual generated samples to be committed under
`docs/extraction/validation/` so output quality is reviewable rather than only
asserted.

## Scope (this PR)

Ownership lane: content-ops/live-output-quality-proof
Slice phase: Functional validation

1. Add a thin issue-specific live validation harness that reuses the existing
   Content Ops host wiring, support-ticket input provider, generated-asset
   review API/repositories, and export helpers.
2. Run the harness against the local real Postgres database and configured
   OpenRouter/Claude route with local Ollama fallback disabled.
3. Use intentionally adversarial inputs: sparse/mixed support tickets, a sharp
   brand voice with banned terms, and at least three variant angles.
4. Generate `landing_page`, `blog_post`, and `sales_brief` in one execution
   request with `variant_count > 1`.
5. Push the generated drafts through the generated-asset review queue and export
   exact saved draft ids from real Postgres.
6. Commit a dated validation artifact under `docs/extraction/validation/` with
   the actual generated samples, brand-voice audit observations, variant
   distinctness review, grounding review, cost/budget notes, and an explicit
   human quality verdict.
7. If the output is mediocre, samey, ungrounded, or otherwise not chargeable,
   report that as the result instead of polishing the artifact into a demo.

### Review Contract

- Acceptance criteria:
  - [x] The plan/artifact names issue #1357 and the live account/run settings.
  - [x] The run uses real Postgres and the configured cloud/OpenRouter model
        route with `EXTRACTED_CAMPAIGN_LLM_AUTO_ACTIVATE_OLLAMA=false`.
  - [x] The run includes inline brand voice and `variant_count > 1`.
  - [x] The run includes `blog_post`, `landing_page`, and `sales_brief` in the
        same execution result.
  - [x] Saved ids are reviewed/approved and exported from real Postgres.
  - [x] The artifact includes real generated samples, not only summaries.
  - [x] The artifact gives a human quality verdict for on-voice, distinctness,
        grounding, and usability, and surfaces the worst examples plainly.
  - [x] The artifact records any live failure (budget gate, all-variants-fail,
        review/export failure, bad quality) as the headline result.
- Affected surfaces: validation harness, live validation docs/fixtures.
- Risk areas: accidentally running a local model, cherry-picking flattering
  samples, hiding weak output, artifact claims not matching raw samples,
  unbounded secrets/logging.
- Reviewer rules triggered: R1, R2, R6, R10, R11.

### Files touched

- `scripts/smoke_content_ops_gate_a_live_quality.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_smoke_content_ops_gate_a_live_quality.py`
- `docs/extraction/validation/content_ops_gate_a_brand_voice_variants_live_quality_2026-06-07.md`
- `docs/extraction/validation/fixtures/content_ops_gate_a_brand_voice_variants_2026-06-07/execution-result.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_brand_voice_variants_2026-06-07/export-blog_post.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_brand_voice_variants_2026-06-07/export-landing_page.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_brand_voice_variants_2026-06-07/export-sales_brief.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_brand_voice_variants_2026-06-07/review-results.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_brand_voice_variants_2026-06-07/summary.json`
- `plans/PR-Gate-A-Live-Output-Quality-Proof.md`
- `HARDENING.md`

## Mechanism

The harness is a small orchestration layer over existing primitives:

1. Load `.env` / `.env.local` via the existing dotenv pattern and initialize the
   real DB pool.
2. Build `ContentOpsExecutionServices(enable_db_services=True)` from
   `atlas_brain._content_ops_services`.
3. Build a support-ticket input package from an adversarial CSV and merge it
   into one execution payload:

   ```json
   {
     "outputs": ["landing_page", "blog_post", "sales_brief"],
     "variant_count": 3,
     "inputs": {"brand_voice": {...}, "source_material": [...]}
   }
   ```

4. Execute through `execute_content_ops_from_mapping(...)`, not service mocks.
5. Extract exact `saved_ids` per output/variant from the execution result.
6. Use the generated-asset review/update path to approve those exact ids for
   the same tenant scope.
7. Export the exact ids through the existing output export helpers and write a
   run bundle under `tmp/`.
8. The committed artifact summarizes the run bundle and includes representative
   raw samples so reviewers can audit quality by eye.

## Intentional

- This is proof, not polish. The harness records bad output and exits non-zero
  on structural live failures, but a mediocre human-quality verdict can still
  be committed as a failed gate artifact.
- The brand voice is inline for this run. Stored profile CRUD is already tested;
  issue #1357 asks whether generation obeys a real profile, not whether profile
  lookup works again.
- The script masks no secrets because it does not print environment values or
  LLM keys.
- The artifact may include long generated samples. That is intentional because
  issue #1357 requires samples to be eyeballed.

## Deferred

- Any quality failures found by the run become follow-up hardening/product
  slices after this gate, not hidden fixes in the validation PR.
- Hosted/browser UI validation remains outside this issue unless review/export
  fails structurally.

Parked hardening:

- `HARDENING.md`: Blog post variants collapse to one persisted draft id.
- `HARDENING.md`: Brand-voice second-person guidance is not consistently honored.
- `HARDENING.md`: Sales brief live generation drifts from requested renewal brief type.
- `HARDENING.md`: Landing-page variants pass audits but are not meaningfully distinct.
- `HARDENING.md`: Blog output uses debug-style source narration instead of publishable prose.
- `HARDENING.md`: Gate A needs a messy-ticket grounding rerun.

## Verification

- `python -m pytest tests/test_smoke_content_ops_gate_a_live_quality.py -q`:
  6 passed.
- `EXTRACTED_CAMPAIGN_LLM_AUTO_ACTIVATE_OLLAMA=false python scripts/smoke_content_ops_gate_a_live_quality.py --account-id 2b2b950d-f64b-4852-bc30-f92a34cdf169 --user-id 11111111-1111-4111-8111-111111111111 --support-ticket-csv extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv --env-file /home/juan-canfield/Desktop/Atlas/.env --env-file /home/juan-canfield/Desktop/Atlas/.env.local --output-dir tmp/content_ops_gate_a_brand_voice_variants_20260607 --variant-count 3 --quality-repair-attempts 1 --max-cost-usd 20.00 --json`:
  execution/review/export completed; human Gate A verdict failed due blog
  variant persistence collapse and brand-voice misses. The harness now records
  the duplicate blog saved id as a structural error instead of a false-green
  summary.
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/gate-a-live-output-quality-proof-pr-body.md`:
  passed.
- `bash scripts/run_extracted_pipeline_checks.sh`: 3248 passed, 10 skipped.

## Estimated diff size

| File | LOC |
|---|---:|
| `scripts/smoke_content_ops_gate_a_live_quality.py` | 643 |
| `tests/test_smoke_content_ops_gate_a_live_quality.py` | 164 |
| `docs/extraction/validation/content_ops_gate_a_brand_voice_variants_live_quality_2026-06-07.md` | 154 |
| `docs/extraction/validation/fixtures/content_ops_gate_a_brand_voice_variants_2026-06-07/` | 2,791 generated sample lines |
| `HARDENING.md` | +56 |
| `plans/PR-Gate-A-Live-Output-Quality-Proof.md` | 161 |
| **Total** | **3,970, justified by issue-required real samples and live harness** |
