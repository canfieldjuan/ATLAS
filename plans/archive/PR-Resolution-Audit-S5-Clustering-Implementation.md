# PR-Resolution-Audit-S5-Clustering-Implementation

## Why this slice exists

Issue #1993 S5 now has a merged current-code spike (#2032) proving three buyer
visible clustering defects: same-intent SSO/SAML tickets can be hidden below the
repeat gate, cancel-subscription and cancel-order tickets can merge into one
row, and equal-score output/source ordering can follow upload order. This slice
turns that spike into the first upstream fix. The diff is over the soft 400 LOC
target because the same slice must convert the merged spike test file, add the
new class probes, update the remediation tracker, and archive the completed
spike plan; the runtime code change remains limited to the extracted
support-ticket grouping path.

Diff-budget override: This implementation is indivisible because the root fix,
S5 acceptance-fixture conversion, class-regression probes, generated SaaS demo
golden refresh, tracker update, and merged-plan archive must land together for
CI to enforce the corrected clustering contract.

### Problem-derived contract

- Root cause: auto-generated `support_ticket_cluster` labels are treated as hard
  FAQ topics before question similarity and embedding logic can compare across
  token-set partitions. Inside a topic, question clustering also accepts
  single-token transitive bridges, so a shared action word such as `cancel` can
  join distinct workflows. The first advisory pass also split generated rows
  away from matching hard-topic rows before singleton filtering, hiding exact
  repeated questions in mixed uploads.
- Correct fix must touch/change: `ticket_faq_markdown` must carry cluster
  source/key metadata, treat generated token clusters as advisory instead of
  hard topics, join advisory rows into matching hard-topic buckets before
  singleton filtering, pool unmatched auto-token rows before question
  clustering, reject one-token non-exact lexical bridges, and add stable
  group/source ordering. It must not fabricate a canonical question when
  actual ticket wording does not support one. Tests must convert the S5 spike
  into corrected acceptance behavior and prove the review-finding classes.
- Must not change: report/snapshot/email/PDF/landing structure, pricing,
  checkout, stored paid report schema, support-ticket package metadata shape,
  explicit/provided/keyword cluster semantics, or the token-set skip warning.

## Scope (this PR)

Ownership lane: resolution-audit-csv
Slice phase: Vertical slice

1. Use token-generated support-ticket clusters as advisory FAQ grouping signals
   while keeping explicit/provided/keyword clusters as hard topics.
2. Let the existing embedding booster see rows that were previously separated
   only by generated token-set partitions.
3. Tighten lexical question merging so a single shared token cannot bridge
   distinct intents.
4. Make equal-score FAQ group and source-id ordering deterministic.
5. Convert the S5 spike assertions into corrected acceptance assertions and
   update the Resolution Audit CSV tracker.
6. Archive the merged S5 spike plan in this branch.

### Files touched

- `docs/audits/resolution-audit-csv/CURRENT_CODE_REMEDIATION_ARC.md`
- `docs/audits/resolution-audit-csv/S5_CLUSTERING_SPIKE.md`
- `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/report.md`
- `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/result.json`
- `extracted_content_pipeline/examples/support_ticket_saas_demo_faq.md`
- `extracted_content_pipeline/faq_deflection_report.py`
- `extracted_content_pipeline/support_ticket_clustering.py`
- `extracted_content_pipeline/ticket_faq_markdown.py`
- `plans/INDEX.md`
- `plans/PR-Resolution-Audit-S5-Clustering-Implementation.md`
- `plans/archive/PR-Resolution-Audit-S5-Clustering-Spike.md`
- `tests/test_content_ops_deflection_resolution_live_proof.py`
- `tests/test_extracted_ticket_faq_markdown.py`
- `tests/test_resolution_audit_s5_clustering_spike.py`

### Review Contract

- Acceptance criteria:
  - SSO/SAML rows no longer render as one fabricated canned question when the
    actual ticket wording does not support that repeated question.
  - The embedding port receives all generated-token candidate rows in the
    reachability fixture rather than only one hard `support_ticket_cluster`
    partition.
  - One-token non-exact bridges such as cancel, refund, and update no longer
    merge distinct repeated workflows.
  - Mixed hard/advisory duplicates can still render as one repeated customer
    question before singleton filtering.
  - Cancel-subscription and cancel-order rows no longer produce one mixed item.
  - Nested opportunities with evidence-level cluster labels and inherited
    generated cluster sources still treat the labels as advisory.
  - Equal-score item order and source-id order are stable under input reversal.
  - The token-set skip boundary remains visible and unchanged.
- Affected surfaces:
  - Extracted support-ticket FAQ grouping and tests only.
- Risk areas:
  - Do not broaden explicit/provided/keyword cluster semantics.
  - Do not change buyer-facing report/snapshot/email/PDF/landing shape.
  - Do not hide the large-upload token-set skip warning.
- Reviewer rules triggered: R1 requirements match, R2 test evidence, R10
  documentation drift, R13 class coverage, R14 codebase verification.

## Mechanism

`ticket_faq_markdown` will carry cluster source/key metadata into its internal
rows. Generated token clusters (`token_set`/`token_anchor`) become advisory for
display/context, while explicit/provided/keyword clusters still supply hard
topics. Advisory rows first try to join a matching hard-topic bucket when their
real question gist matches a hard row, preferring cluster-label/topic matches;
unmatched auto-token rows are pooled for question clustering so the existing
embedding booster can compare across their former partitions. Lexical merging
keeps exact duplicate gist joins, but every non-exact join needs at least two
shared tokens so one action word cannot single-link unrelated workflows. FAQ
item rows are sorted by stable source key, and equal-score group sorting adds a
stable question/source tie-breaker. Advisory generated clusters never
substitute a canned question; if the wording does not form a repeat, the rows
stay non-repeat unless the real semantic merge path joins them.

`support_ticket_clustering` receives only the minimal SSO terminology fold
needed to keep identity-provider/SAML wording in the same support-ticket
cluster family. The public package metadata keys stay unchanged.

## Intentional

- This is a first vertical fix, not a full semantic-clustering rewrite. The
  optional embedding path remains the semantic rescue mechanism; this PR makes
  it reachable across generated token partitions.
- Explicit/provided support-ticket clusters remain hard topic boundaries because
  they come from upstream data or operator taxonomy, not the lossy token preview.
- No new report model fields, hosted payload fields, snapshot fields, email/PDF
  sections, landing copy, or pricing/checkout changes.

## Deferred

- F7/service-level embedding-port forwarding remains separate unless this
  implementation proves a runtime path still discards a configured port.
- Wider calibrated semantic thresholds and real corpus evaluation remain a
  later robust-testing slice.
- S6 text/comment/outcome hygiene and S7 date-window policy remain separate
  #1993 slices.

Parked hardening: none.

## Verification

- Passed: `pytest tests/test_extracted_ticket_faq_markdown.py tests/test_resolution_audit_s5_clustering_spike.py tests/test_extracted_support_ticket_input_package.py -q` (523 passed)
- Passed: `pytest tests/test_resolution_audit_s5_clustering_spike.py -q` (7 passed)
- Passed: `pytest tests/test_extracted_ticket_faq_markdown.py tests/test_resolution_audit_s5_clustering_spike.py tests/test_extracted_support_ticket_input_package.py tests/test_content_ops_faq_saas_demo_corpus.py -q` (558 passed)
- Passed: `pytest tests/test_content_ops_deflection_report.py::test_csv_product_gap_owner_lane_vertical_routes_login_gap tests/test_content_ops_deflection_resolution_live_proof.py::test_resolution_live_proof_artifacts_show_publishable_and_gap_lanes tests/test_content_ops_deflection_resolution_live_proof.py::test_resolution_live_proof_regenerates_from_committed_csv -q` (3 passed)
- Passed: `pytest tests/test_content_ops_deflection_report.py tests/test_content_ops_deflection_resolution_live_proof.py -q` (184 passed, 4 skipped)
- Passed: `pytest tests/test_resolution_audit_s5_clustering_spike.py tests/test_extracted_ticket_faq_markdown.py tests/test_content_ops_deflection_report.py::test_csv_product_gap_owner_lane_vertical_routes_login_gap tests/test_content_ops_deflection_resolution_live_proof.py tests/test_content_ops_faq_saas_demo_corpus.py -q` (469 passed)
- Passed: `scripts/run_extracted_pipeline_checks.sh` via bash (5163 passed, 21 skipped)
- Passed: `python scripts/audit_extracted_pipeline_ci_enrollment.py --atlas-brain-tests-from origin/main`
- Passed: `scripts/validate_extracted_content_pipeline.sh` via bash
- Passed: `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline`
- Passed: `python scripts/audit_extracted_standalone.py --fail-on-debt`
- Passed: `scripts/check_ascii_python.sh` via bash
- Passed: `extracted/_shared/scripts/sync_extracted.sh extracted_content_pipeline` via bash
- Planned: `bash scripts/push_pr.sh /tmp/resolution-audit-s5-implementation-pr-body.md`

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/audits/resolution-audit-csv/CURRENT_CODE_REMEDIATION_ARC.md` | 17 |
| `docs/audits/resolution-audit-csv/S5_CLUSTERING_SPIKE.md` | 9 |
| `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/report.md` | 28 |
| `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/result.json` | 8 |
| `extracted_content_pipeline/examples/support_ticket_saas_demo_faq.md` | 8 |
| `extracted_content_pipeline/faq_deflection_report.py` | 3 |
| `extracted_content_pipeline/support_ticket_clustering.py` | 5 |
| `extracted_content_pipeline/ticket_faq_markdown.py` | 247 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Resolution-Audit-S5-Clustering-Implementation.md` | 176 |
| `plans/archive/PR-Resolution-Audit-S5-Clustering-Spike.md` | 0 |
| `tests/test_content_ops_deflection_resolution_live_proof.py` | 10 |
| `tests/test_extracted_ticket_faq_markdown.py` | 44 |
| `tests/test_resolution_audit_s5_clustering_spike.py` | 153 |
| **Total** | **711** |
