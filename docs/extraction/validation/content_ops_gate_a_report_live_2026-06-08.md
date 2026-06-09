# Gate A Report Live Coverage Proof - 2026-06-08

## Purpose

This validation closes the remaining Gate A live-coverage gap: `report` was
wired through the Content Ops execution stack, but it had not been exercised
end-to-end by the live-quality harness.

This report records the harness result and sample pointers only. It does not
self-certify the generated report as product-accepted; the reviewer owns that
judgment against the exported row.

## Input Fixture

Fixture:
`extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv`

The harness converted the uploaded support-ticket rows into
`campaign_opportunities`, then bound the report request to the first imported
target id so preview and execution read the same seeded opportunity.

| Import metric | Value |
|---|---:|
| Inserted opportunities | 36 |
| Skipped opportunities | 0 |
| Import source | `gate_a_support_ticket_csv` |
| Target mode | `vendor_retention` |
| Bound report target id | `saas-demo-001` |

Bound source row:

| Field | Value |
|---|---|
| Account | `Atlas Supply` |
| Vendor | `FlowPilot` |
| Contact | `Morgan Reyes`, Revenue Operations Lead |
| Subject | `Attribution export blocked` |
| Evidence | `How do I export attribution reports before our board meeting?` |
| Pain category | `reporting export` |

## Command

Local Ollama fallback was disabled for the run.

```bash
EXTRACTED_CAMPAIGN_LLM_AUTO_ACTIVATE_OLLAMA=false python scripts/smoke_content_ops_gate_a_live_quality.py \
  --account-id 7d9c8e6a-5f42-4c91-9237-1394a0f2b681 \
  --user-id 11111111-1111-4111-8111-111111111111 \
  --support-ticket-csv extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv \
  --env-file /home/juan-canfield/Desktop/Atlas/.env \
  --env-file /home/juan-canfield/Desktop/Atlas/.env.local \
  --output-dir tmp/content_ops_gate_a_report_live_20260608 \
  --outputs report \
  --variant-count 3 \
  --quality-repair-attempts 1 \
  --max-cost-usd 20.00 \
  --json
```

## Result

Harness status: `passed`

Committed artifacts:

- `docs/extraction/validation/fixtures/content_ops_gate_a_report_live_20260608/execution-result.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_report_live_20260608/opportunity-import.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_report_live_20260608/review-results.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_report_live_20260608/summary.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_report_live_20260608/export-report.json`

Structural output counts:

| Output | Requested opportunities | Generated drafts | Exported rows | Saved IDs | Errors |
|---|---:|---:|---:|---:|---:|
| `report` | 1 | 1 | 1 | 1 | 0 |

Resolved generation route:

- `anthropic/claude-sonnet-4-5` is recorded as `generation_model` on the
  committed report row.
- `generation_parse_attempts=1`.
- Local Ollama fallback was disabled:
  `EXTRACTED_CAMPAIGN_LLM_AUTO_ACTIVATE_OLLAMA=false`.
- Budget cap `--max-cost-usd 20.00` was not triggered.

## Generated Sample

Report sample:

| ID | Title | Sections | References |
|---|---|---:|---:|
| `c8b7105a-392e-4641-90a2-bb9110ae580e` | FlowPilot Attribution Export Limitation Impacting Atlas Supply Operations | 3 | 1 |

Summary:

> Atlas Supply's Revenue Operations Lead, Morgan Reyes, has encountered a
> critical limitation in FlowPilot's reporting capabilities.

Sections:

| Section | Evidence ids |
|---|---|
| Export Limitation Context | `saas-demo-001` |
| Operational Impact on Revenue Operations | `saas-demo-001` |
| Retention Risk Considerations | `saas-demo-001` |

Reference ids: `saas-demo-001`

## Verification Commands

Focused harness tests:

```bash
pytest tests/test_smoke_content_ops_gate_a_live_quality.py
```

Result: `23 passed in 0.21s`.

Artifact JSON validation:

```bash
python - <<'PY'
from pathlib import Path
import json
root = Path('docs/extraction/validation/fixtures/content_ops_gate_a_report_live_20260608')
for path in sorted(root.glob('*.json')):
    with path.open(encoding='utf-8') as handle:
        json.load(handle)
    print(path)
PY
```

Result: all five committed artifact JSON files parsed successfully.

Generation route and generated-row check:

```bash
python - <<'PY'
from pathlib import Path
import json
root = Path('docs/extraction/validation/fixtures/content_ops_gate_a_report_live_20260608')
payload = json.loads((root / 'export-report.json').read_text(encoding='utf-8'))
row = payload['rows'][0]
model = row.get('metadata', {}).get('generation_model')
references = row.get('reference_ids')
sections = row.get('sections') or []
print(f'rows={payload.get("count")}; model={model}; references={references}; sections={len(sections)}')
assert payload.get('count') == 1
assert model == 'anthropic/claude-sonnet-4-5'
assert references == ['saas-demo-001']
assert len(sections) == 3
PY
```

Result: one generated report row using `anthropic/claude-sonnet-4-5`, reference
`saas-demo-001`, and three sections.

Full extracted package checks:

```bash
bash scripts/run_extracted_pipeline_checks.sh
```

Result: `3475 passed, 10 skipped, 1 warning in 57.26s`; wrapper completed all
extracted checks.
