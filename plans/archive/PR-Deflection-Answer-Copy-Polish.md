## Why this slice exists

`PR-Deflection-Resolution-Copy-Polish` removed the internal
"Use the uploaded resolution evidence:" scaffold from proven FAQ steps. The
reviewer then flagged the next buyer-visible polish gaps: proven-answer
summaries still start with "Customers mention:", single-resolution drafts add
an internal reviewer instruction as step 2, and long resolution excerpts can
truncate mid-word. Those issues make otherwise resolution-backed drafts look
less publishable even though the underlying evidence scoping is correct.

## Scope (this PR)

Ownership lane: content-ops/faq-deflection-backend

Slice phase: Product polish

1. Replace the proven FAQ item `answer` summary with clean, evidence-scoped
   copy that does not start with "Customers mention:".
2. Avoid mid-word truncation when long `resolution_text` excerpts are shortened
   for draft steps.
3. Remove the internal single-resolution "Confirm the answer..." reviewer step
   from proven drafts; keep fail-closed review steps for unproven drafts.
4. Refresh report/macro/docs expectations for the buyer-visible copy.
5. Align the compact FAQ output proof with the new proven-draft minimum of one
   verified resolution step plus the support fallback.

### Files touched

- `extracted_content_pipeline/ticket_faq_markdown.py`
- `scripts/smoke_content_ops_faq_output_proof.py`
- `tests/test_extracted_ticket_faq_markdown.py`
- `tests/test_extracted_ticket_faq_macro_writeback.py`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_content_ops_faq_report_contract_docs.py`
- `tests/test_smoke_content_ops_faq_output_proof.py`
- `docs/frontend/content_ops_faq_deflection_report_example.json`
- `docs/frontend/content_ops_faq_report_example.json`
- `HARDENING.md`
- `plans/PR-Deflection-Answer-Copy-Polish.md`

## Mechanism

The item builder already knows the selected question, source count, and whether
resolution evidence exists. This slice uses that state to emit a short
evidence-scoped answer summary for proven drafts while leaving unproven drafts
clearly marked as needing review.

For resolution-backed steps, the existing first-sentence excerpt logic remains
the source of truth. The shortening branch changes from a raw character slice
to a word-boundary slice so generated steps do not end mid-word. When only one
resolution excerpt exists, the proven draft now returns that excerpt plus the
support step instead of inserting an internal reviewer instruction.

The compact output proof checker now treats two steps as sufficient coverage:
one verified resolution step and one support fallback. Its negative fixture
still uses one step, so the `action_step_coverage` detection branch remains
proven.

## Intentional

- This does not synthesize new troubleshooting instructions. Proven drafts
  still use only uploaded `resolution_text` excerpts for action steps.
- This does not polish unproven draft-review steps. Those are intentionally
  internal because the item is not publishable without verified resolution
  evidence.
- This does not change grouping, evidence scoping, ranking, paywall behavior, or
  macro approval gates.
- Cross-layer caller hints were inspected: `faq_deflection_report.py` is covered
  by the focused deflection-report tests, and `ticket_faq_search.py` references
  its own answer-summary helper rather than the new ticket FAQ summary helper.

## Deferred

- Follow-up slice: add the stricter artifact/eval gate that fails if an item
  cites evidence outside its evidence scope.
- Follow-up slice: richer final help-center prose over verified resolution
  evidence once the copy surface is clean and guarded.
- Considered hardening: the existing `HARDENING.md` FAQ UI dependency-audit
  entry is unrelated to this backend copy surface and remains parked.
- Parked hardening: `SaaS demo preflight subprocess reloads live repo dotenv`
  records the local `.env` isolation issue that still affects full local mirror
  runs in provisioned checkouts.

## Verification

- `pytest tests/test_extracted_ticket_faq_markdown.py tests/test_extracted_ticket_faq_macro_writeback.py tests/test_content_ops_deflection_report.py tests/test_extracted_ticket_faq_output_ingestion.py tests/test_content_ops_faq_report_contract_docs.py -q` -- 190 passed.
- `pytest tests/test_smoke_content_ops_faq_output_proof.py -q` -- 4 passed.
- `python scripts/build_content_ops_deflection_report.py /home/juan-canfield/Desktop/saas-deflection-large-sample.csv --source-format csv --max-items 8 --result-output /tmp/deflection-large-saas-answer-copy-polish.json --summary-output /tmp/deflection-large-saas-answer-copy-polish-summary.json --require-output-checks --json` -- passed; generated 8 opportunities, 7 proven drafts, 1 review-needed bucket.
- Direct 420-row generator check -- 0 occurrences of "Customers mention:",
  "Confirm the answer matches", "Use the uploaded resolution evidence", and
  the prior `for se...` mid-word truncation; all 7 proven answers start with
  "Verified resolution evidence".
- `scripts/validate_extracted_content_pipeline.sh` -- passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` -- passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` -- passed.
- `scripts/check_ascii_python.sh` -- passed.
- Full extracted mirror with live deflection env vars blanked before
  `scripts/run_extracted_pipeline_checks.sh` -- local checkout caveat: 1
  unrelated SaaS demo preflight subprocess test failed because it reloads this
  repo's `.env`; 2888 passed, 10 skipped.
- `scripts/local_pr_review.sh` with `--allow-dirty` and the staged PR body file -- passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| `extracted_content_pipeline/ticket_faq_markdown.py` | ~35 |
| Output proof checker | ~2 |
| Tests | ~56 |
| Frontend JSON examples | ~268 |
| Hardening note | ~11 |
| Plan doc | ~118 |
| **Total** | **~491** |

Over the 400 LOC soft cap because the frontend contract examples are
regenerated JSON fixtures. The behavioral diff is intentionally narrow: answer
summary phrasing, word-boundary excerpting, and removal of one internal
single-resolution step.
