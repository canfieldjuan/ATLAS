# PR-Content-Selling-Context-Inputs

## Why this slice exists

PR-Content-Ops-Real-Asset-Url-CTA proved that live campaign drafts can use a real booking URL, but the URL is only wired through the review-source Postgres smoke script. The product execution path still has no first-class way for a host/API caller to pass selling context at request time. This slice makes `inputs.selling.booking_url` a real Content Ops execution input for email campaign generation while preserving the placeholder-URL fail-close gate already merged in `CampaignGenerationService`.

## Scope (this PR)

1. Thread request-level campaign opportunity defaults from `ContentOpsRequest.inputs` into the email campaign dispatcher.
2. Teach `CampaignGenerationService.generate` to merge those defaults into each normalized opportunity without overwriting row-provided data.
3. Regression-test both seams: dispatcher input propagation and real campaign prompt/source metadata visibility.

### Files touched

- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Selling-Context-Inputs.md`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/content_ops_execution.py`
- `extracted_content_pipeline/campaign_generation.py`
- `tests/test_extracted_content_ops_execution.py`
- `tests/test_extracted_campaign_generation.py`

## Mechanism

The API already accepts arbitrary bounded `inputs`; no route model change is required. `content_ops_execution._dispatch_email_campaign` extracts an opportunity-defaults mapping from `request.inputs`, currently limited to:

```python
{"selling": request.inputs["selling"]}
```

or a flat `booking_url` / `selling_booking_url` convenience input promoted to:

```python
{"selling": {"booking_url": "..."}}
```

The dispatcher passes that mapping as `opportunity_defaults` to `CampaignGenerationService.generate`. The service merges defaults into each opportunity after the repository read and before reasoning, channel shaping, prompt rendering, quality revalidation, and metadata persistence. Row fields win over defaults; nested `selling` mappings are merged so a row-specific `selling.booking_url` is not replaced by a request-level fallback.

## Intentional

- No new FastAPI model field: `inputs` is already the product extension point and is depth/size bounded by the API validator.
- No prompt change: the packaged campaign skill already consumes `opportunity_json`, and `selling.booking_url` is already part of that contract.
- No URL validation beyond the existing placeholder gate: hosts can pass customer-owned booking/asset URLs, and the persistence boundary still rejects placeholder URLs emitted by the LLM.
- Only email campaign execution uses `opportunity_defaults` in this slice. Other opportunity-driven assets can adopt the same seam later if they need selling context.

## Deferred

- Generic CLI sugar for `--booking-url` outside the review-source smoke script. The core product/API path lands here first; source conversion/import CLIs can add convenience flags in a follow-up without changing generation behavior.
- Richer selling context schema validation. For now the product preserves host-provided mappings instead of inventing a premature schema.

## Verification

Commands run:

```bash
pytest tests/test_extracted_content_ops_execution.py tests/test_extracted_campaign_generation.py  # 94 passed
python -m py_compile extracted_content_pipeline/content_ops_execution.py extracted_content_pipeline/campaign_generation.py tests/test_extracted_content_ops_execution.py tests/test_extracted_campaign_generation.py
bash scripts/validate_extracted_content_pipeline.sh
python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
python scripts/audit_extracted_standalone.py --fail-on-debt
bash scripts/check_ascii_python.sh
bash scripts/local_pr_review.sh --allow-dirty
```

## Estimated diff size

| File | LOC churn (approx) |
|---|---:|
| `docs/extraction/coordination/inflight.md` | 3 |
| `plans/PR-Content-Selling-Context-Inputs.md` | 72 |
| `extracted_content_pipeline/README.md` | 17 |
| `extracted_content_pipeline/content_ops_execution.py` | 50 |
| `extracted_content_pipeline/campaign_generation.py` | 37 |
| `tests/test_extracted_content_ops_execution.py` | 106 |
| `tests/test_extracted_campaign_generation.py` | 72 |
| **Total** | **357** |

Below the 400 LOC review budget.
