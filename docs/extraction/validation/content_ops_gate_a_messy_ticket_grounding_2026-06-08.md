# Gate A Messy Ticket Grounding Rerun - 2026-06-08

## Purpose

This validation reruns the Gate A support-ticket proof from #1383 on a noisier
checked fixture. The goal is to prove the same live path can ingest messy,
lopsided support-ticket data and generate the selected outputs without falling
back to local models or reintroducing blog source-mechanics narration.

This report records the harness result and sample pointers only. It does not
self-certify the outputs as product-accepted; the reviewer owns that judgment.

## Input Fixture

Fixture:
`extracted_content_pipeline/examples/support_ticket_messy_grounding_sources.csv`

Package summary from the focused fixture test:

| Metric | Value |
|---|---:|
| Source rows | 44 |
| Included ticket rows | 42 |
| Skipped ticket rows | 2 |
| Question-like tickets | 39 |
| Dated window detected | false |
| Parser warnings | 2 x `ticket_row_missing_text` |

Top ticket clusters:

| Cluster | Count |
|---|---:|
| reporting export | 11 |
| dashboard freshness | 7 |
| sso setup | 6 |
| billing and plan management | 4 |
| data import | 3 |
| api and webhooks | 3 |
| workflow automation | 2 |
| permissions and seats | 2 |
| integration sync | 2 |
| Weekly account digest | 1 |
| Customer said thanks | 1 |

The fixture intentionally includes duplicate wording, blank/missing ticket text,
missing optional fields, inconsistent timestamp shapes, and lopsided cluster
counts.

## Command

Local Ollama fallback was disabled for the run.

```bash
EXTRACTED_CAMPAIGN_LLM_AUTO_ACTIVATE_OLLAMA=false python scripts/smoke_content_ops_gate_a_live_quality.py \
  --account-id 3e4f1b6c-1a92-4b8a-9d7e-5f2a0e8c7b91 \
  --user-id 11111111-1111-4111-8111-111111111111 \
  --support-ticket-csv extracted_content_pipeline/examples/support_ticket_messy_grounding_sources.csv \
  --env-file /home/juan-canfield/Desktop/Atlas/.env \
  --env-file /home/juan-canfield/Desktop/Atlas/.env.local \
  --output-dir tmp/content_ops_gate_a_messy_ticket_grounding_20260608 \
  --outputs landing_page,blog_post,sales_brief \
  --variant-count 3 \
  --quality-repair-attempts 1 \
  --max-cost-usd 20.00 \
  --json
```

## Result

Harness status: `passed`

Committed artifacts:

- `docs/extraction/validation/fixtures/content_ops_gate_a_messy_ticket_grounding_20260608/execution-result.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_messy_ticket_grounding_20260608/review-results.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_messy_ticket_grounding_20260608/summary.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_messy_ticket_grounding_20260608/export-landing_page.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_messy_ticket_grounding_20260608/export-blog_post.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_messy_ticket_grounding_20260608/export-sales_brief.json`

Structural output counts:

| Output | Exported rows | Saved IDs | Errors |
|---|---:|---:|---:|
| `landing_page` | 3 | 3 | 0 |
| `blog_post` | 3 | 3 | 0 |
| `sales_brief` | 3 | 3 | 0 |

Resolved model route:

- `anthropic/claude-sonnet-4-5` is recorded as `generation_model` on all
  committed generated rows.

Blog source-mechanics check:

- No committed blog export matched the source-mechanics phrases checked for
  #1383 regressions: uploaded tickets, uploaded set, source rows, usable rows,
  included rows, rows were included, we ingested, you sent, support ticket
  export, or source mechanics.

## Generated Samples

Landing page samples:

| Variant | ID | Title | Headline |
|---|---|---|---|
| social proof | `12ccdfa0-af71-43ed-898b-3b7371164b7c` | Support Ticket FAQ Gap Audit - Turn Repeat Questions Into Approved Answers | Turn Repeat Support Tickets Into Approved FAQ Answers |
| outcome led | `5bc0bc6c-5717-4ab0-af61-1f77f17d0284` | Cut Repeat Support Tickets with FAQ Gap Audit | Turn Repeat Tickets Into Approved FAQ Answers |
| pain led | `cf198a6e-fb29-4df2-8066-37d096894067` | Stop Answering the Same Support Questions Every Week | Your team answers the same questions every week |

Blog post samples:

| Variant | ID | Title | First H2 |
|---|---|---|---|
| social proof | `ed3e1ecc-7106-428d-bb59-ffbc07738b3d` | Support Ticket FAQ Gaps: What Repeat Tickets Reveal Before Renewal | What repeat support questions show |
| outcome led | `d087572c-206c-444b-99b2-07477bfcba26` | Support Ticket FAQ Gaps: What Repeat Tickets Reveal Before Renewal | What repeat support questions show |
| pain led | `d787d807-7d97-4587-a75e-6405b7f1cb67` | Support Ticket FAQ Gaps: What Repeat Tickets Reveal Before Renewal | What repeat support questions show |

Sales brief samples:

| Variant | ID | Brief type | Headline |
|---|---|---|---|
| social proof | `6e7fb6bf-edd9-4b34-a756-ffef38713e8b` | renewal | Your RevOps lead hit a wall exporting attribution reports right before the board packet deadline. Support ticket shows reporting friction. |
| outcome led | `f4ac7467-911e-4840-86ca-d451d5ed45b7` | renewal | Your RevOps lead hit a reporting wall 30 days before board packet deadline. Export friction is renewal exposure. |
| pain led | `42d3e666-5bab-4581-9e88-ede18cd629c7` | renewal | Your RevOps lead can't export attribution reports. Board packet deadline is forcing workarounds. Renewal window is your fix moment. |

## Verification Commands

Focused tests:

```bash
python -m pytest tests/test_smoke_content_ops_gate_a_live_quality.py tests/test_smoke_content_ops_live_generation.py -q
```

Result: `52 passed in 0.31s`.

Artifact JSON validation:

```bash
python - <<'PY'
from pathlib import Path
import json
root = Path('docs/extraction/validation/fixtures/content_ops_gate_a_messy_ticket_grounding_20260608')
for path in sorted(root.glob('*.json')):
    with path.open(encoding='utf-8') as handle:
        json.load(handle)
    print(path)
PY
```

Result: all committed artifact JSON files parsed successfully.

Model route check:

```bash
python - <<'PY'
from pathlib import Path
import json
root = Path('docs/extraction/validation/fixtures/content_ops_gate_a_messy_ticket_grounding_20260608')
for path in sorted(root.glob('export-*.json')):
    payload = json.loads(path.read_text(encoding='utf-8'))
    models = sorted({row.get('metadata', {}).get('generation_model') for row in payload.get('rows', [])})
    print(f'{path.name}: {payload.get("count")} rows; models={models}')
    assert models == ['anthropic/claude-sonnet-4-5']
PY
```

Result: all three export files recorded three rows using
`anthropic/claude-sonnet-4-5`.

Blog source-mechanics regression check:

```bash
if rg -n -i "uploaded tickets?|uploaded ticket set|uploaded set|source rows|usable rows|included rows|rows were included|we ingested|you sent|support ticket export|source mechanics" docs/extraction/validation/fixtures/content_ops_gate_a_messy_ticket_grounding_20260608/export-blog_post.json; then exit 1; fi
```

Result: no matches.

Full extracted package checks:

```bash
bash scripts/validate_extracted_content_pipeline.sh
python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
python scripts/audit_extracted_standalone.py --fail-on-debt
bash scripts/check_ascii_python.sh
bash scripts/run_extracted_pipeline_checks.sh
```

Results:

- `validate_extracted_content_pipeline.sh`: passed.
- `forbid_atlas_reasoning_imports.py`: clean.
- `audit_extracted_standalone.py --fail-on-debt`: `Atlas runtime import findings: 0`.
- `check_ascii_python.sh`: passed.
- `run_extracted_pipeline_checks.sh`: `3391 passed, 10 skipped, 1 warning in 58.88s`.
