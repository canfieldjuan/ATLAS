# PR-Deflection-Measured-Repetition

## Why this slice exists

#1481 (spotted by the operator on the first real full-volume funnel run) and
#1460 (deep-dive P1): the deflection report never measured question-level
repetition. Grouping was one FAQ per topic/intent bucket for rows without
resolution evidence, so `repeat_ticket_count` counted bucket membership -
on the live 35,386-row CFPB upload the buyer-visible snapshot asserted ALL
35,386 tickets were "repeat-ticket hits" and priced them at $477,711
($5.7M/yr run-rate), with rank #75/#76 rendering "1 repeat tickets". The
product's own copy defines a repeat question as one customers asked more
than once, and its landing page models industry repeat rates at ~40%.

This slice implements #1460's prescribed fix (operator-selected on #1481):
sub-cluster within each topic/intent bucket by question similarity, and
count only dense sub-clusters as repeat questions. On the same CFPB upload
the measured numbers become 12,858 repeat-ticket hits (36% - in line with
the industry benchmark) across 3,649 real question clusters, $173,583
support tax, with the top question a genuine 253-ticket cluster.

The diff is far over the 400-LOC soft cap (~1.7k insertions over 40 files)
because the semantics change forces an atomic re-baseline: ~120 tests, the
shared fixtures they exercise, the frontend contract docs and regenerated
examples, two smoke harnesses, and one recorded live-proof golden all
encode the old one-FAQ-per-topic contract. Splitting would leave CI red
between slices. The production core is ~220 LOC in two modules.

## Scope (this PR)

Ownership lane: deflection-full-50k-e2e-proof
Slice phase: Production hardening

1. Deterministic question sub-clustering in `ticket_faq_markdown.py`:
   MinHash-LSH (16 fixed permutations, 4 bands x 4 rows) over question-gist
   token sets with exact-Jaccard >= 1/3 verification and union-find, applied
   to every topic-degraded group. Only sub-clusters with >= 2 tickets remain
   FAQ groups; resolution-scoped groups are untouched.
2. Excluded one-off tickets are counted (`non_repeat_ticket_count`,
   `non_repeat_question_count` on `TicketFAQMarkdownResult` and report
   summaries), surfaced via a `non_repeat_tickets_excluded` warning and a
   report sentence, and counted as covered by the `condensed` output check.
3. `repeat_ticket_count` (snapshot summary + paid-report Support Tax
   section) sums only items with >= 2 tickets; snapshot summary gains
   `non_repeat_ticket_count`.
4. Re-baseline of every consumer of the old semantics: tests, shared
   fixtures (reworded so intended repeats genuinely repeat), frontend
   contract docs + regenerated examples, smoke harnesses, and the
   2026-06-09 resolution-evidence live-proof golden (consciously, with a
   re-baseline note in the validation doc).

### Review Contract

- Acceptance criteria:
  - [ ] A mixed bucket of genuinely different questions produces multiple
        distinct FAQ items, not one mega-topic (#1460 acceptance).
  - [ ] A question asked once is never counted as a repeat: it is excluded
        from items, counted in non_repeat_* fields, and surfaced in the
        warning and report text.
  - [ ] `repeat_ticket_count` <= included tickets, and
        repeat + non_repeat == covered ticket sources (the `condensed`
        check enforces the accounting).
  - [ ] Clustering is deterministic across runs/processes (fixed hash
        constants; no randomness or time dependence).
  - [ ] Resolution-scoped grouping and drafted-answer behavior unchanged.
- Affected surfaces: FAQ grouping engine, report/snapshot metrics, frontend
  contract docs, fixtures/goldens, smoke harnesses, tests.
- Risk areas: correctness of the repeat definition, determinism,
  performance at full volume, golden/contract drift.
- Reviewer rules triggered: R1, R2, R9, R10, R12.

### Files touched

- `.github/workflows/atlas_content_ops_input_provider_checks.yml`
- `atlas-intel-ui/scripts/content-ops-deflection-report-ui.test.mjs`
- `docs/extraction/validation/deflection_resolution_evidence_live_proof_2026-06-09.md`
- `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/report.md`
- `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/result.json`
- `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/source.csv`
- `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/summary.json`
- `docs/frontend/content_ops_faq_deflection_checkout_contract.md`
- `docs/frontend/content_ops_faq_deflection_report_example.json`
- `docs/frontend/content_ops_faq_deflection_snapshot_example.json`
- `docs/frontend/content_ops_faq_report_contract.md`
- `docs/frontend/content_ops_faq_report_example.json`
- `extracted_content_pipeline/examples/support_ticket_bundle.json`
- `extracted_content_pipeline/examples/support_ticket_saas_demo_faq.md`
- `extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv`
- `extracted_content_pipeline/examples/support_ticket_sources.csv`
- `extracted_content_pipeline/faq_deflection_report.py`
- `extracted_content_pipeline/ticket_faq_markdown.py`
- `plans/PR-Deflection-Measured-Repetition.md`
- `scripts/smoke_content_ops_faq_output_proof.py`
- `scripts/smoke_extracted_content_ops_execution.py`
- `tests/test_atlas_content_ops_execution_services.py`
- `tests/test_check_content_ops_faq_search_route_contract.py`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_content_ops_faq_report_contract_docs.py`
- `tests/test_extracted_campaign_source_adapters.py`
- `tests/test_extracted_content_control_surface_api.py`
- `tests/test_extracted_content_deflection_submit.py`
- `tests/test_extracted_content_ops_execution.py`
- `tests/test_extracted_content_ops_live_execute_harness.py`
- `tests/test_extracted_support_ticket_input_provider.py`
- `tests/test_extracted_ticket_faq_markdown.py`
- `tests/test_smoke_content_ops_cfpb_faq_markdown.py`
- `tests/test_smoke_content_ops_faq_scale_run.py`
- `tests/test_support_ticket_provider_landing_blog_execute.py`

## Mechanism

`_question_gist_tokens` reduces each row to the token set of its extracted
question sentence (or its opening gist), redaction tokens stripped.
`_question_subclusters` runs deterministic MinHash-LSH over those gists:
identical gists merge via an exact-match fast path; otherwise each row's
16-value signature is banded (4 x 4) and candidates are verified against
the band bucket's first member with exact Jaccard >= 1/3 before union-find
merges them. Recall is approximate by design (#1460 names MinHash/LSH);
determinism comes from literal permutation constants and insertion-ordered
processing. Work is bounded: per-row cost is O(gist tokens x 16) plus a
constant number of representative verifications - no pairwise scan (the
#1454 lesson).

After grouping, topic-degraded groups are replaced by their dense
sub-clusters keyed `(topic, "topic:<t>:question:<i>")`; singleton clusters
increment the non-repeat counters instead of becoming items. The
`condensed` output check treats excluded one-offs as covered
(`rendered + non_repeat == ticket sources`), so the report cannot silently
drop tickets - the exclusion is part of the accounting and the rendered
text.

Measured on the live CFPB sample: FAQ build 38 s at 35,386 rows; the full
3,649-item report renders to a 2.6 MB PDF in 24 s, inside the email
delivery limits, so the complete measured backlog ships without a shape
change.

## Intentional

- The >= 2 threshold and >= 1/3 Jaccard are the repeat definition for this
  slice; both agent-assisted re-baseline rounds were barred from relaxing
  them, and no production defect was found during re-baselining.
- LSH recall is approximate, and the *effective* per-pair threshold is
  stricter than the declared 1/3: the current 4-band x 4-row shape places
  the candidate-recall S-curve midpoint at J ~= 0.71, so a pair only has
  ~4% chance of being verified at J=0.33, ~22% at J=0.50, and ~70% at
  J=0.70. Moderate-similarity rewordings (J~0.5, e.g. "How do I reset my
  password?" vs "How can I reset my password please, account locked") are
  mostly never verified against the 1/3 gate; transitive chaining through
  exact duplicates partially rescues recall on real data (CFPB still landed
  at the ~36% benchmark). This under-counts repeats rather than over-counts
  - the conservative, buyer-protecting direction for a billing-adjacent
  metric - so it ships as-is, with the 8-band x 2-row re-band (which aligns
  the empirical curve with the declared 1/3 at no extra hashing cost, but
  shifts ~120 baselines) tracked as a follow-up in #1504. Fixtures that must
  merge use wording whose gists match exactly or overwhelmingly.
- The 12-row demo fixture family was rebuilt to contain genuinely repeated
  questions (it previously repeated themes, not questions, and would
  honestly produce an empty snapshot under measured repetition).
- The 2026-06-09 resolution-evidence live-proof artifacts were regenerated
  with reworded unresolved-lane rows and are no longer the byte-output of
  the original dated run; the validation doc carries an explicit
  re-baseline note. Headline counts are unchanged (12/4/2/2).
- Smoke-harness payload changes (`smoke_extracted_content_ops_execution.py`,
  `smoke_content_ops_faq_output_proof.py`) only add repeat partners so the
  harnesses keep producing items under the new semantics.

## Deferred

- The remaining #1440 live-run legs (pay -> report email + PDF) follow this
  slice deploying, so the first paid artifact carries the honest numbers.
- The portfolio results-page copy ("repeat-ticket hits") needs no code
  change - the numbers flow from the snapshot - but a copy line showing the
  excluded one-off count on the page is a portfolio follow-up.
- Rarity-weighted anchor redesign for category-less cluster previews
  (#1454 option 3) remains separate.

Parked hardening: none.

## Verification

- Passed: bash scripts/run_extracted_pipeline_checks.sh
  - 3741 passed, 10 skipped, 0 failed (full extracted gauntlet).
- Passed: bash scripts/check_ascii_python.sh
- Passed: python scripts/audit_extracted_standalone.py --fail-on-debt
- Real-data validation (35,386-row CFPB sample, the #1440 live input):
  - repeat_ticket_count 35,386 -> 12,858 (36%); non_repeat 22,528;
    support tax $477,711 -> $173,583; 3,649 measured question clusters;
    top question a 253-ticket dense cluster; `condensed` True.
  - FAQ build 38 s; full-report PDF 2,614,779 bytes in 24.4 s via
    `render_deflection_full_report_pdf`.
- Determinism: fixed constants; both re-baseline rounds verified borderline
  fixture pairs through the real `_question_subclusters` path.
- Pending before push:
  - python scripts/sync_pr_plan.py plans/PR-Deflection-Measured-Repetition.md --check
  - bash scripts/local_pr_review.sh (via push_pr.sh with the body file)

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_content_ops_input_provider_checks.yml` | 3 |
| `atlas-intel-ui/scripts/content-ops-deflection-report-ui.test.mjs` | 2 |
| `docs/extraction/validation/deflection_resolution_evidence_live_proof_2026-06-09.md` | 13 |
| `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/report.md` | 6 |
| `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/result.json` | 2 |
| `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/source.csv` | 6 |
| `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/summary.json` | 2 |
| `docs/frontend/content_ops_faq_deflection_checkout_contract.md` | 5 |
| `docs/frontend/content_ops_faq_deflection_report_example.json` | 32 |
| `docs/frontend/content_ops_faq_deflection_snapshot_example.json` | 5 |
| `docs/frontend/content_ops_faq_report_contract.md` | 14 |
| `docs/frontend/content_ops_faq_report_example.json` | 83 |
| `extracted_content_pipeline/examples/support_ticket_bundle.json` | 23 |
| `extracted_content_pipeline/examples/support_ticket_saas_demo_faq.md` | 157 |
| `extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv` | 74 |
| `extracted_content_pipeline/examples/support_ticket_sources.csv` | 4 |
| `extracted_content_pipeline/faq_deflection_report.py` | 37 |
| `extracted_content_pipeline/ticket_faq_markdown.py` | 202 |
| `plans/PR-Deflection-Measured-Repetition.md` | 236 |
| `scripts/smoke_content_ops_faq_output_proof.py` | 25 |
| `scripts/smoke_extracted_content_ops_execution.py` | 41 |
| `tests/test_atlas_content_ops_execution_services.py` | 88 |
| `tests/test_check_content_ops_faq_search_route_contract.py` | 10 |
| `tests/test_content_ops_deflection_report.py` | 33 |
| `tests/test_content_ops_faq_report_contract_docs.py` | 20 |
| `tests/test_extracted_campaign_source_adapters.py` | 6 |
| `tests/test_extracted_content_control_surface_api.py` | 15 |
| `tests/test_extracted_content_deflection_submit.py` | 6 |
| `tests/test_extracted_content_ops_execution.py` | 32 |
| `tests/test_extracted_content_ops_live_execute_harness.py` | 33 |
| `tests/test_extracted_support_ticket_input_provider.py` | 11 |
| `tests/test_extracted_ticket_faq_markdown.py` | 1348 |
| `tests/test_smoke_content_ops_cfpb_faq_markdown.py` | 11 |
| `tests/test_smoke_content_ops_faq_scale_run.py` | 37 |
| `tests/test_support_ticket_provider_landing_blog_execute.py` | 2 |
| **Total** | **2624** |
