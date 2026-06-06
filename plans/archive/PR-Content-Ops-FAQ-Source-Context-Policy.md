# PR-Content-Ops-FAQ-Source-Context-Policy

## Why this slice exists

Running real CFPB complaint rows from
`/home/juan-canfield/Downloads/archive (1)/rows.csv` through the generic FAQ
Markdown path exposed a source-context loss. The source adapter preserves raw
provider fields such as `Product`, `Issue`, and `Sub-issue`, but the FAQ intent
classifier only uses source titles, evidence text, and pain points. That lets
generic SaaS rules fire on consumer complaint narratives:

- credit-report rows can become `reporting friction`;
- debt-collection rows can become account/profile or reporting topics;
- mortgage rows fall to generic billing or workflow guidance.

The fix belongs in the generic FAQ classifier, not in a CFPB-only exporter or
dataset-specific script.

## Scope (this PR)

1. Include generic source-context fields in FAQ intent classification.
2. Add mortgage-servicing FAQ policy alongside the existing credit-report,
   debt-collection, and billing policies.
3. Reject complaint-process boilerplate as FAQ questions so rows such as
   "submitted several complaints through the CFPB" fall back to the applicable
   source policy instead of becoming the rendered FAQ heading.
4. Add regression coverage using CFPB-shaped CSV rows through the real source
   adapter plus direct FAQ builder tests for the new mortgage policy.
5. Update the AI Content Ops backlog with the real-output finding and closeout.
6. Replace the stale in-flight FAQ complaint-source row with this slice.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-FAQ-Source-Context-Policy.md` | Plan doc for this slice. |
| `docs/extraction/coordination/inflight.md` | Claim this AI Content Ops slice and remove the merged stale FAQ row. |
| `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md` | Record the real CFPB output-quality follow-up. |
| `extracted_content_pipeline/ticket_faq_markdown.py` | Use source-context fields in intent classification and add mortgage policy. |
| `tests/test_extracted_ticket_faq_markdown.py` | Regression tests for source-context classification and mortgage actions. |

## Mechanism

The FAQ classifier will build intent text from the existing evidence text,
source title, pain points, and generic source-context fields such as `product`,
`category`, `issue`, and `sub_issue`. The lookup uses the existing tolerant
field normalization helper, so provider exports with `Product`, `Sub-product`,
`Sub Issue`, or snake_case variants work without dataset-specific branches.

Mortgage servicing gets the same treatment as credit reports and debt
collection: a policy question, domain-specific action steps, and escalation
guidance. Mortgage intent classification requires mortgage-specific anchors
such as `mortgage`, `home loan`, `foreclosure`, or `mortgage servicer`; broader
servicing language such as `loan modification` appears in the mortgage guidance
only after a row is already classified as mortgage. Complaint-process
boilerplate is treated the same way as vague openers like "Need help?" so the
renderer does not turn filing-channel text into the FAQ question. The source
adapter remains generic and unchanged.

## Intentional

- No LLM calls. FAQ output stays deterministic and audit-friendly.
- No CFPB-only branch. CFPB is the real public fixture that exposed the gap,
  but the implementation reads generic source-context fields.
- No full-corpus CSV import in tests. Tests use small CFPB-shaped fixtures so
  CI stays fast and independent of local downloads.

## Deferred

- Better representative-evidence ranking for very long complaint narratives.
  Current output uses the first evidence row in each topic; this slice fixes
  wrong topic/action policy first.
- Larger source-specific policy packs for other financial products such as
  student loans, credit cards, or bank accounts. Add only when real output
  samples show the need.

## Verification

Passed locally:

```bash
pytest tests/test_extracted_ticket_faq_markdown.py tests/test_smoke_content_ops_cfpb_faq_markdown.py -q
# 58 passed

python -m py_compile extracted_content_pipeline/ticket_faq_markdown.py tests/test_extracted_ticket_faq_markdown.py

python scripts/build_extracted_ticket_faq_markdown.py /tmp/atlas_cfpb_samples/debt_collection_150.csv --source-format csv --title "Debt Collection Complaint FAQ" --max-items 8 --max-evidence-per-item 3 --support-contact https://example.com/support --require-output-checks --output /tmp/atlas_cfpb_samples/debt_collection_150_after.md
python scripts/build_extracted_ticket_faq_markdown.py /tmp/atlas_cfpb_samples/credit_reporting_150.csv --source-format csv --title "Credit Reporting Complaint FAQ" --max-items 8 --max-evidence-per-item 3 --support-contact https://example.com/support --require-output-checks --output /tmp/atlas_cfpb_samples/credit_reporting_150_after.md
python scripts/build_extracted_ticket_faq_markdown.py /tmp/atlas_cfpb_samples/mortgage_150.csv --source-format csv --title "Mortgage Complaint FAQ" --max-items 8 --max-evidence-per-item 3 --support-contact https://example.com/support --require-output-checks --output /tmp/atlas_cfpb_samples/mortgage_150_after.md

bash scripts/run_extracted_pipeline_checks.sh
# 1560 passed, 1 existing torch/pynvml warning
```

`bash scripts/local_pr_review.sh` runs after commit.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~85 |
| FAQ classifier/policy | ~45 |
| Tests | ~110 |
| Docs/coordination | ~20 |
| Total | ~260 |
