# Gate A Email Campaign Input-Fit Proof - 2026-06-08

## Purpose

This validation closes issue #1376's deferred email-campaign Gate A coverage:
prove the live-quality harness can adapt support-ticket source material into
the sequence-shaped `email_campaign` path, import campaign opportunities, run
the real Content Ops execution services, and persist generated campaign drafts.

This report records the harness result and sample pointers only. It does not
self-certify the generated copy as product-accepted; the reviewer owns that
judgment.

## Input Fixture

Fixture:
`extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv`

The harness converted the uploaded support-ticket rows into
`campaign_opportunities` before execution.

| Import metric | Value |
|---|---:|
| Inserted opportunities | 36 |
| Skipped opportunities | 0 |
| Import source | `gate_a_support_ticket_csv` |
| Target mode | `vendor_retention` |
| Filter topic type | `support_ticket_faq_gap_live_gate_a` |

## Command

Local Ollama fallback was disabled for the run.

```bash
EXTRACTED_CAMPAIGN_LLM_AUTO_ACTIVATE_OLLAMA=false python scripts/smoke_content_ops_gate_a_live_quality.py \
  --account-id 5b2f2a9c-6d1e-4f2c-9a87-31e64d42a901 \
  --user-id 11111111-1111-4111-8111-111111111111 \
  --support-ticket-csv extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv \
  --env-file /home/juan-canfield/Desktop/Atlas/.env \
  --env-file /home/juan-canfield/Desktop/Atlas/.env.local \
  --output-dir tmp/content_ops_gate_a_email_campaign_input_fit_20260608 \
  --outputs email_campaign \
  --variant-count 3 \
  --quality-repair-attempts 1 \
  --max-cost-usd 20.00 \
  --json
```

## Result

Harness status: `passed`

Committed artifacts:

- `docs/extraction/validation/fixtures/content_ops_gate_a_email_campaign_input_fit_20260608/execution-result.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_email_campaign_input_fit_20260608/opportunity-import.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_email_campaign_input_fit_20260608/review-results.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_email_campaign_input_fit_20260608/summary.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_email_campaign_input_fit_20260608/export-email_campaign.json`

Structural output counts:

| Output | Requested opportunities | Generated drafts | Exported rows | Saved IDs | Errors |
|---|---:|---:|---:|---:|---:|
| `email_campaign` | 1 | 2 | 2 | 2 | 0 |

Resolved generation route:

- `deterministic/single-support-ticket` is recorded as `llm_model` on both
  committed campaign rows.
- The sparse one-ticket support-ticket path does not call a model and records
  `generation_parse_attempts=0`. This prevents the fabricated count/volume
  claims exposed during review while preserving the same review/export
  pipeline.
- Both rows record `campaign_revalidation.audit.status=pass` with empty
  `unsupported_numeric_claims`, `unsupported_scan_claims`, and
  `unsupported_aggregate_claims`.

Shared prompt prevention:

- `apply_brand_voice_to_system_prompt(...)` now injects a shared
  `## Grounding contract` for campaign, landing page, blog post, sales brief,
  and social-post rewrite prompts, even when no brand voice profile is
  selected.
- The broad no-fabrication rule is no longer campaign-only; focused tests
  assert the shared system prompt prevents invented counts, percentages,
  statistics, scan/research claims, unsupported entity names, and ungrounded
  proof points across the generator seam.

Placeholder URL guard:

- The first attempted run proved the placeholder guard still blocks
  `example.com` output.
- The passing run threads a real `selling.affiliate_url` into the prompt
  context and both committed drafts use
  `https://finetunelab.ai/systems/ai-content-ops/intake`.

## Generated Samples

Email campaign samples:

| Channel | ID | Recipient | Subject | CTA |
|---|---|---|---|---|
| `email_cold` | `13c6c335-f14e-41e1-adbd-687d723e3d6c` | `billing@silverline.example` | Usage overage question from your team | Worth a look? |
| `email_followup` | `b26be0f8-6c54-4519-8b26-a2debb0a6f9d` | `billing@silverline.example` | Follow-up: usage overage question | Worth a look? |

Grounding sample:

- Target ID: `saas-demo-036`
- Company: `Silverline Studio`
- Vendor: `FlowPilot`
- Evidence: `How do I see which workspace caused a usage overage?`
- Pain category: `billing and plan management`

## Verification Commands

Focused tests:

```bash
python -m pytest tests/test_extracted_brand_voice.py tests/test_extracted_campaign_generation.py tests/test_extracted_campaign_skill_registry.py tests/test_extracted_landing_page_generation.py tests/test_extracted_blog_generation.py tests/test_extracted_sales_brief_generation.py tests/test_extracted_social_post_generation.py -q
```

Result: `280 passed in 0.57s`.

Artifact JSON validation:

```bash
python - <<'PY'
from pathlib import Path
import json
root = Path('docs/extraction/validation/fixtures/content_ops_gate_a_email_campaign_input_fit_20260608')
for path in sorted(root.glob('*.json')):
    with path.open(encoding='utf-8') as handle:
        json.load(handle)
    print(path)
PY
```

Result: all committed artifact JSON files parsed successfully.

Generation route and generated-row check:

```bash
python - <<'PY'
from pathlib import Path
import json
root = Path('docs/extraction/validation/fixtures/content_ops_gate_a_email_campaign_input_fit_20260608')
payload = json.loads((root / 'export-email_campaign.json').read_text(encoding='utf-8'))
models = sorted({row.get('llm_model') for row in payload.get('rows', [])})
channels = sorted({row.get('channel') for row in payload.get('rows', [])})
parse_attempts = sorted({row.get('generation_parse_attempts') for row in payload.get('rows', [])})
print(f'rows={payload.get("count")}; models={models}; channels={channels}; parse_attempts={parse_attempts}')
assert payload.get('count') == 2
assert models == ['deterministic/single-support-ticket']
assert channels == ['email_cold', 'email_followup']
assert parse_attempts == [0]
PY
```

Result: two generated rows, one cold email and one follow-up, both using the
deterministic sparse-ticket scaffold.

Full extracted package checks:

```bash
bash scripts/run_extracted_pipeline_checks.sh
```

Result: `3433 passed, 10 skipped, 1 warning in 52.02s`.
