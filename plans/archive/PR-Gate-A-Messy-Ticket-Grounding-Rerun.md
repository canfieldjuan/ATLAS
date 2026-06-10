# PR-Gate-A-Messy-Ticket-Grounding-Rerun

## Why this slice exists

#1383 closed the clean Gate A support-ticket proof for `landing_page`,
`blog_post`, and `sales_brief`: all three selected generators saved 3/3
variants, and the refreshed blog samples no longer exposed upload/source-row
mechanics. The remaining same-lane hardening item is stricter: the clean SaaS
fixture has exactly balanced clusters and well-formed rows, so it does not
stress grounding on noisy support data.

This slice drains `HARDENING.md`'s "Gate A needs a messy-ticket grounding
rerun" item by proving the same Gate A path on a deliberately messier checked
fixture: lopsided cluster counts, blank rows, missing optional fields, duplicate
wording, inconsistent timestamps, mixed customer wording, and support-ticket
rows that should be ignored or handled safely by the package.

This PR may exceed the 400 LOC soft cap because the raw JSON validation
artifacts are the deliverable. The code/test surface should stay narrow; the
artifact overage is the evidence.

## Scope (this PR)

Ownership lane: content-ops/gate-a-output-quality
Slice phase: Functional validation

1. Add a checked messy support-ticket CSV fixture for Gate A validation.
2. Run `scripts/smoke_content_ops_gate_a_live_quality.py` against the real
   local database/model route with
   `--outputs landing_page,blog_post,sales_brief`.
3. Disable local Ollama fallback with
   `EXTRACTED_CAMPAIGN_LLM_AUTO_ACTIVATE_OLLAMA=false` and use the configured
   Sonnet route from `.env` / `.env.local`.
4. Commit the exact run artifacts under
   `docs/extraction/validation/fixtures/` plus a markdown report with the
   command, resolved model, structural result, sample pointers, and the messy
   fixture summary.
5. Keep this as validation unless the messy run exposes a correctness failure
   that must be fixed to make the proof meaningful.
6. Do not self-certify product acceptance. The reviewer owns the GOOD-bar
   judgment against the exported drafts.

### Review Contract

- Acceptance criteria:
  - [ ] The committed command includes
        `--outputs landing_page,blog_post,sales_brief`.
  - [ ] The committed source fixture is visibly messier than
        `support_ticket_saas_demo_sources.csv`.
  - [ ] The artifact set includes `execution-result.json`,
        `review-results.json`, `summary.json`, and per-output exports for each
        selected output that the harness produced.
  - [ ] The report records the resolved model route and that local Ollama
        fallback was disabled.
  - [ ] The report surfaces real samples and structural results without
        claiming product pass/fail beyond the harness result.
  - [ ] The run excludes `email_campaign` so its separate input-fit questions
        cannot mask the three-generator grounding signal.
  - [ ] Blog exports are checked for the source-mechanics phrases closed in
        #1383.
- Affected surfaces: validation fixture, validation artifacts, support-ticket
  live proof documentation, and focused fixture/packaging checks if needed.
- Risk areas: accidentally making the messy fixture too tidy, leaking
  source/upload mechanics back into blog copy, over-claiming product acceptance,
  and allowing junk rows to dominate the proof.
- Reviewer rules triggered: R1, R10.

### Files touched

- `extracted_content_pipeline/examples/support_ticket_messy_grounding_sources.csv`
- `docs/extraction/validation/content_ops_gate_a_messy_ticket_grounding_2026-06-08.md`
- `docs/extraction/validation/fixtures/content_ops_gate_a_messy_ticket_grounding_20260608/execution-result.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_messy_ticket_grounding_20260608/export-blog_post.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_messy_ticket_grounding_20260608/export-landing_page.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_messy_ticket_grounding_20260608/export-sales_brief.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_messy_ticket_grounding_20260608/review-results.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_messy_ticket_grounding_20260608/summary.json`
- `HARDENING.md`
- `plans/PR-Gate-A-Messy-Ticket-Grounding-Rerun.md`
- `tests/test_smoke_content_ops_gate_a_live_quality.py`

## Mechanism

The fixture will use the existing support-ticket CSV ingestion path. The live
proof script packages the rows through the support-ticket input package, seeds
the selected content inputs, runs the configured real model route, reviews the
saved ids, exports the reviewed drafts, and writes the same artifact envelope
used by #1383.

The report will compare the messy fixture shape to the clean 36-row SaaS demo
fixture at a high level: row count, included ticket count, cluster distribution,
junk/blank/partial rows, and whether the exported drafts stayed grounded in
observed ticket evidence.

## Intentional

- Keep the first messy proof on Sonnet, matching #1383 and the #1360 baseline.
- Do not include `email_campaign`; this rerun isolates the same three
  generators that just cleared the clean fixture.
- Do not weaken any quality gates to make the messy proof pass. If a selected
  output is blocked, the artifact/report should show why.
- Keep the fixture synthetic/non-sensitive but messy enough to exercise the
  real parser and grounding boundaries.

## Deferred

- Cheaper-model readiness remains separate after Sonnet messy-input behavior is
  reviewed.
- Product polish for list-dumpy blog prose, templated titles, or landing-page
  section dedup remains separate unless it blocks this proof.
- Live validation on customer-provided private data is out of scope; this slice
  uses a checked synthetic fixture that is safe to commit.

Parked hardening:

- None. This PR drains the "Gate A needs a messy-ticket grounding rerun" item
  from `HARDENING.md`.

## Verification

- `python -m pytest tests/test_smoke_content_ops_gate_a_live_quality.py tests/test_smoke_content_ops_live_generation.py -q`
  - Result: `52 passed in 0.31s`.
- `EXTRACTED_CAMPAIGN_LLM_AUTO_ACTIVATE_OLLAMA=false python scripts/smoke_content_ops_gate_a_live_quality.py --account-id 3e4f1b6c-1a92-4b8a-9d7e-5f2a0e8c7b91 --user-id 11111111-1111-4111-8111-111111111111 --support-ticket-csv extracted_content_pipeline/examples/support_ticket_messy_grounding_sources.csv --env-file /home/juan-canfield/Desktop/Atlas/.env --env-file /home/juan-canfield/Desktop/Atlas/.env.local --output-dir tmp/content_ops_gate_a_messy_ticket_grounding_20260608 --outputs landing_page,blog_post,sales_brief --variant-count 3 --quality-repair-attempts 1 --max-cost-usd 20.00 --json`
  - Result: `passed`; exports contain 3 `landing_page`, 3 `blog_post`, and 3
    `sales_brief` rows.
- JSON validation over committed artifacts.
  - Result: all six committed JSON artifacts parsed successfully.
- Model-route check over the three committed export files.
  - Result: all nine generated rows record
    `generation_model=anthropic/claude-sonnet-4-5`.
- Blog source-mechanics phrase check over committed blog exports.
  - Result: no matches.
- `bash` with `scripts/validate_extracted_content_pipeline.sh`
  - Result: passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline`
  - Result: clean.
- `python scripts/audit_extracted_standalone.py --fail-on-debt`
  - Result: `Atlas runtime import findings: 0`.
- `bash` with `scripts/check_ascii_python.sh`
  - Result: passed.
- `bash` with `scripts/run_extracted_pipeline_checks.sh`
  - Result: `3391 passed, 10 skipped, 1 warning in 58.88s`.
- `bash` with `scripts/local_pr_review.sh`
  - Result: passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/examples/support_ticket_messy_grounding_sources.csv` | 45 |
| `docs/extraction/validation/content_ops_gate_a_messy_ticket_grounding_2026-06-08.md` | 195 |
| `docs/extraction/validation/fixtures/content_ops_gate_a_messy_ticket_grounding_20260608/*.json` | 3623 |
| `HARDENING.md` | -9 |
| `plans/PR-Gate-A-Messy-Ticket-Grounding-Rerun.md` | 160 |
| `tests/test_smoke_content_ops_gate_a_live_quality.py` | 29 |
| **Total** | **4048 / -9** |
