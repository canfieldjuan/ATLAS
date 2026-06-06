# PR-Content-Ops-Real-Asset-Url-CTA

## Why this slice exists

PR #641 correctly blocks placeholder URLs before campaign drafts are saved, but
the live G2 provider proof now fail-closes because the smoke input does not
provide a real selling or booking URL. The next production step is to let host
operators provide a real CTA URL on the live smoke path so provider-generated
drafts can pass with useful, non-fabricated links.

This slice should not relax the placeholder URL gate. It gives the existing
campaign prompt a real `selling.booking_url` value through the same opportunity
JSON contract it already consumes.

## Scope (this PR)

1. Add a narrow `--booking-url` option to the review-source Postgres smoke.
2. Inject the URL as source-row fallback metadata only for the smoke's imported
   opportunities, preserving existing `--default-field` behavior.
3. Add tests proving the booking URL reaches generation through the imported
   opportunity payload and that blank values are ignored.

### Files touched

- `scripts/smoke_content_ops_review_source_postgres.py`
- `tests/test_smoke_content_ops_review_source_postgres.py`
- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Ops-Real-Asset-Url-CTA.md`

## Mechanism

`scripts/smoke_content_ops_review_source_postgres.py` already parses
repeatable `--default-field key=value` metadata and passes it through
the ingestion inspector and source opportunity loader.
This PR should add a first-class `--booking-url` operator flag that injects a
nested `{"selling": {"booking_url": "<url>"}}` value into the same defaults map
before ingestion and import.

The campaign prompt already receives the normalized opportunity as
`{opportunity_json}` in `CampaignGenerationService._generate_one()`, and the
outreach skill already instructs the model to include `selling.booking_url`.

## Intentional

- This is smoke/operator wiring, not a new campaign-generation abstraction.
  The generation service should keep accepting arbitrary opportunity metadata.
- The placeholder URL gate remains fail-closed. Hosts must provide a real URL
  for live CTA links.
- This slice only covers the review-source Postgres smoke path that exposed the
  bug. Broader asset URL UX is separate.

## Deferred

- General Content Ops UI/API fields for account-level selling assets.
- Automatic lookup of published blog posts or booking URLs from host settings.
- URL validation beyond blank/nonblank input. The existing placeholder URL gate
  remains the enforcement point before persistence.

## Verification

Local checks:

```bash
pytest tests/test_smoke_content_ops_review_source_postgres.py
# 13 passed

python -m py_compile scripts/smoke_content_ops_review_source_postgres.py tests/test_smoke_content_ops_review_source_postgres.py
# passed

bash scripts/local_pr_review.sh
# passed

python scripts/smoke_content_ops_review_source_postgres.py \
  --source g2 --vendor Slack --limit 1 \
  --account-id content_ops_live_g2_real_url_<timestamp> \
  --llm pipeline --channels email_cold,email_followup --min-drafts 2 \
  --default-field 'company_name=Acme Logistics' \
  --default-field 'contact_email=ops@example.com' \
  --default-field 'contact_name=Jordan Lee' \
  --booking-url https://juancanfield.com/ --json
# live provider/Postgres path passed: 2 generated, 2 saved, 0 skipped
```

If the focused checks pass, rerun the G2 live-provider smoke with a real
booking URL and confirm it persists at least two drafts without
`placeholder_url` errors.

## Estimated diff size

| File | LOC churn (approx) |
|---|---:|
| `scripts/smoke_content_ops_review_source_postgres.py` | 24 |
| `tests/test_smoke_content_ops_review_source_postgres.py` | 41 |
| `docs/extraction/coordination/inflight.md` | 3 |
| `plans/PR-Content-Ops-Real-Asset-Url-CTA.md` | 96 |
| **Total** | **164** |

Below the 400 LOC review budget.
