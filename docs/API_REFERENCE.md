# ATLAS API Inventory

Exhaustive route inventory generated from FastAPI router decorators in `atlas_brain/api/` and app mounts in `atlas_brain/main.py`.

- Total discovered routes: **476**
- Most application APIs are mounted under `/api/v1`.
- Root-level exceptions include OpenAI compatibility (`/v1/...`), Ollama compatibility (`/`, `/api/...`), and webhooks (`/webhooks/...`).

## Route groups

| Module | Routes |
|---|---:|
| `admin_costs` | 26 |
| `alerts` | 12 |
| `auth` | 7 |
| `autonomous` | 10 |
| `b2b_affiliates` | 7 |
| `b2b_campaigns` | 35 |
| `b2b_crm_events` | 7 |
| `b2b_evidence` | 7 |
| `b2b_reviews` | 1 |
| `b2b_scrape` | 17 |
| `b2b_tenant_dashboard` | 53 |
| `b2b_vendor_briefing` | 13 |
| `b2b_win_loss` | 5 |
| `billing` | 4 |
| `blog_admin` | 9 |
| `blog_public` | 2 |
| `campaign_webhooks` | 2 |
| `comms.call_actions` | 10 |
| `comms.management` | 9 |
| `comms.webhooks` | 12 |
| `consumer_dashboard` | 35 |
| `contacts` | 2 |
| `devices.control` | 5 |
| `edge.websocket` | 2 |
| `email_actions` | 5 |
| `email_drafts` | 8 |
| `health` | 2 |
| `identity` | 5 |
| `inbox_rules` | 7 |
| `intelligence` | 9 |
| `invoicing.actions` | 4 |
| `llm` | 6 |
| `ollama_compat` | 4 |
| `openai_compat` | 1 |
| `orchestrated.websocket` | 1 |
| `pipeline_visibility` | 14 |
| `presence` | 2 |
| `proactive_actions` | 4 |
| `prospects` | 9 |
| `query.text` | 1 |
| `reasoning` | 4 |
| `recognition` | 18 |
| `security` | 6 |
| `seller_campaigns` | 9 |
| `session` | 6 |
| `settings` | 14 |
| `speaker` | 8 |
| `system` | 1 |
| `universal_scrape` | 7 |
| `vendor_targets` | 7 |
| `video` | 6 |
| `vision` | 16 |

## `admin_costs` (26 routes)

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/v1/admin/costs/b2b-efficiency` | `b2b_efficiency` |
| `GET` | `/api/v1/admin/costs/burn-dashboard` | `burn_dashboard` |
| `GET` | `/api/v1/admin/costs/by-model` | `cost_by_model` |
| `GET` | `/api/v1/admin/costs/by-operation` | `cost_by_operation` |
| `GET` | `/api/v1/admin/costs/by-provider` | `cost_by_provider` |
| `GET` | `/api/v1/admin/costs/by-vendor` | `cost_by_vendor` |
| `GET` | `/api/v1/admin/costs/by-workflow` | `cost_by_workflow` |
| `GET` | `/api/v1/admin/costs/cache-health` | `cache_health` |
| `GET` | `/api/v1/admin/costs/daily` | `cost_daily` |
| `GET` | `/api/v1/admin/costs/error-timeline` | `error_timeline` |
| `GET` | `/api/v1/admin/costs/generic-reasoning` | `generic_reasoning` |
| `GET` | `/api/v1/admin/costs/reasoning-activity` | `reasoning_activity` |
| `GET` | `/api/v1/admin/costs/recent` | `recent_calls` |
| `GET` | `/api/v1/admin/costs/reconciliation` | `cost_reconciliation` |
| `GET` | `/api/v1/admin/costs/runs/{run_id}` | `cost_run_detail` |
| `GET` | `/api/v1/admin/costs/scraping/details` | `scraping_details` |
| `GET` | `/api/v1/admin/costs/scraping/reddit/by-subreddit` | `reddit_by_subreddit` |
| `GET` | `/api/v1/admin/costs/scraping/reddit/overview` | `reddit_overview` |
| `GET` | `/api/v1/admin/costs/scraping/reddit/per-vendor` | `reddit_per_vendor` |
| `GET` | `/api/v1/admin/costs/scraping/reddit/signal-breakdown` | `reddit_signal_breakdown` |
| `GET` | `/api/v1/admin/costs/scraping/runs/{run_id}/pages` | `scraping_run_pages` |
| `GET` | `/api/v1/admin/costs/scraping/summary` | `scraping_summary` |
| `GET` | `/api/v1/admin/costs/scraping/top-posts` | `scraping_top_posts` |
| `GET` | `/api/v1/admin/costs/summary` | `cost_summary` |
| `GET` | `/api/v1/admin/costs/system-resources` | `system_resources` |
| `GET` | `/api/v1/admin/costs/task-health` | `task_health` |

## `alerts` (12 routes)

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/v1/alerts` | `get_alerts` |
| `POST` | `/api/v1/alerts/acknowledge-all` | `acknowledge_all_alerts` |
| `DELETE` | `/api/v1/alerts/cleanup` | `cleanup_old_alerts` |
| `GET` | `/api/v1/alerts/rules` | `list_alert_rules` |
| `POST` | `/api/v1/alerts/rules` | `create_alert_rule` |
| `DELETE` | `/api/v1/alerts/rules/{rule_name}` | `delete_alert_rule` |
| `POST` | `/api/v1/alerts/rules/{rule_name}/disable` | `disable_alert_rule` |
| `POST` | `/api/v1/alerts/rules/{rule_name}/enable` | `enable_alert_rule` |
| `GET` | `/api/v1/alerts/stats` | `get_alert_stats` |
| `POST` | `/api/v1/alerts/test` | `trigger_test_alert` |
| `GET` | `/api/v1/alerts/unacknowledged/count` | `get_unacknowledged_count` |
| `POST` | `/api/v1/alerts/{alert_id}/acknowledge` | `acknowledge_alert` |

## `auth` (7 routes)

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/v1/auth/change-password` | `change_password` |
| `POST` | `/api/v1/auth/forgot-password` | `forgot_password` |
| `POST` | `/api/v1/auth/login` | `login` |
| `GET` | `/api/v1/auth/me` | `me` |
| `POST` | `/api/v1/auth/refresh` | `refresh` |
| `POST` | `/api/v1/auth/register` | `register` |
| `POST` | `/api/v1/auth/reset-password` | `reset_password` |

## `autonomous` (10 routes)

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/v1/autonomous/` | `list_tasks` |
| `POST` | `/api/v1/autonomous/` | `create_task` |
| `GET` | `/api/v1/autonomous/status/summary` | `get_status_summary` |
| `DELETE` | `/api/v1/autonomous/{task_id}` | `delete_task` |
| `GET` | `/api/v1/autonomous/{task_id}` | `get_task` |
| `PUT` | `/api/v1/autonomous/{task_id}` | `update_task` |
| `POST` | `/api/v1/autonomous/{task_id}/disable` | `disable_task` |
| `POST` | `/api/v1/autonomous/{task_id}/enable` | `enable_task` |
| `GET` | `/api/v1/autonomous/{task_id}/executions` | `get_executions` |
| `POST` | `/api/v1/autonomous/{task_id}/run` | `run_task_now` |

## `b2b_affiliates` (7 routes)

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/v1/b2b/tenant/affiliates/clicks` | `record_click` |
| `GET` | `/api/v1/b2b/tenant/affiliates/clicks/summary` | `click_summary` |
| `GET` | `/api/v1/b2b/tenant/affiliates/opportunities` | `list_opportunities` |
| `GET` | `/api/v1/b2b/tenant/affiliates/partners` | `list_partners` |
| `POST` | `/api/v1/b2b/tenant/affiliates/partners` | `create_partner` |
| `DELETE` | `/api/v1/b2b/tenant/affiliates/partners/{partner_id}` | `delete_partner` |
| `PATCH` | `/api/v1/b2b/tenant/affiliates/partners/{partner_id}` | `update_partner` |

## `b2b_campaigns` (35 routes)

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/v1/b2b/campaigns` | `list_campaigns` |
| `GET` | `/api/v1/b2b/campaigns/analytics/by-company` | `analytics_by_company` |
| `GET` | `/api/v1/b2b/campaigns/analytics/by-vendor` | `analytics_by_vendor` |
| `GET` | `/api/v1/b2b/campaigns/analytics/funnel` | `analytics_funnel` |
| `GET` | `/api/v1/b2b/campaigns/analytics/timeline` | `analytics_timeline` |
| `POST` | `/api/v1/b2b/campaigns/bulk-approve` | `bulk_approve` |
| `POST` | `/api/v1/b2b/campaigns/bulk-reject` | `bulk_reject` |
| `GET` | `/api/v1/b2b/campaigns/company-timeline` | `company_timeline` |
| `GET` | `/api/v1/b2b/campaigns/export` | `export_campaigns` |
| `POST` | `/api/v1/b2b/campaigns/generate` | `generate_campaigns_endpoint` |
| `GET` | `/api/v1/b2b/campaigns/quality-diagnostics` | `campaign_quality_diagnostics` |
| `GET` | `/api/v1/b2b/campaigns/quality-trends` | `campaign_quality_trends` |
| `GET` | `/api/v1/b2b/campaigns/review-candidates` | `review_candidates` |
| `GET` | `/api/v1/b2b/campaigns/review-candidates/summary` | `review_candidates_summary` |
| `GET` | `/api/v1/b2b/campaigns/review-queue` | `review_queue` |
| `GET` | `/api/v1/b2b/campaigns/review-queue/summary` | `review_queue_summary` |
| `GET` | `/api/v1/b2b/campaigns/sequences` | `list_sequences` |
| `GET` | `/api/v1/b2b/campaigns/sequences/{sequence_id}` | `get_sequence` |
| `GET` | `/api/v1/b2b/campaigns/sequences/{sequence_id}/audit-log` | `sequence_audit_log` |
| `GET` | `/api/v1/b2b/campaigns/sequences/{sequence_id}/outcome` | `get_outcome` |
| `POST` | `/api/v1/b2b/campaigns/sequences/{sequence_id}/outcome` | `record_outcome` |
| `POST` | `/api/v1/b2b/campaigns/sequences/{sequence_id}/pause` | `pause_sequence` |
| `POST` | `/api/v1/b2b/campaigns/sequences/{sequence_id}/resume` | `resume_sequence` |
| `POST` | `/api/v1/b2b/campaigns/sequences/{sequence_id}/set-recipient` | `set_recipient` |
| `GET` | `/api/v1/b2b/campaigns/stats` | `campaign_stats` |
| `GET` | `/api/v1/b2b/campaigns/suppressions` | `list_suppressions` |
| `POST` | `/api/v1/b2b/campaigns/suppressions` | `create_suppression` |
| `GET` | `/api/v1/b2b/campaigns/suppressions/check` | `check_suppression` |
| `DELETE` | `/api/v1/b2b/campaigns/suppressions/{suppression_id}` | `delete_suppression` |
| `GET` | `/api/v1/b2b/campaigns/{campaign_id:uuid}` | `get_campaign` |
| `PATCH` | `/api/v1/b2b/campaigns/{campaign_id:uuid}` | `update_campaign` |
| `POST` | `/api/v1/b2b/campaigns/{campaign_id:uuid}/approve` | `approve_campaign` |
| `GET` | `/api/v1/b2b/campaigns/{campaign_id:uuid}/audit-log` | `campaign_audit_log` |
| `POST` | `/api/v1/b2b/campaigns/{campaign_id:uuid}/cancel` | `cancel_campaign` |
| `POST` | `/api/v1/b2b/campaigns/{campaign_id:uuid}/queue-send` | `queue_campaign_for_send` |

## `b2b_crm_events` (7 routes)

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/v1/b2b/crm/events` | `list_crm_events` |
| `POST` | `/api/v1/b2b/crm/events` | `ingest_crm_event` |
| `POST` | `/api/v1/b2b/crm/events/batch` | `ingest_crm_events_batch` |
| `GET` | `/api/v1/b2b/crm/events/enrichment-stats` | `get_enrichment_stats` |
| `POST` | `/api/v1/b2b/crm/events/hubspot` | `ingest_hubspot_webhook` |
| `POST` | `/api/v1/b2b/crm/events/pipedrive` | `ingest_pipedrive_webhook` |
| `POST` | `/api/v1/b2b/crm/events/salesforce` | `ingest_salesforce_webhook` |

## `b2b_evidence` (7 routes)

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/v1/b2b/evidence/annotations` | `list_annotations` |
| `POST` | `/api/v1/b2b/evidence/annotations` | `set_annotation` |
| `POST` | `/api/v1/b2b/evidence/annotations/remove` | `remove_annotations` |
| `GET` | `/api/v1/b2b/evidence/trace` | `get_trace` |
| `GET` | `/api/v1/b2b/evidence/vault` | `get_vault` |
| `GET` | `/api/v1/b2b/evidence/witnesses` | `list_witnesses` |
| `GET` | `/api/v1/b2b/evidence/witnesses/{witness_id}` | `get_witness` |

## `b2b_reviews` (1 route)

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/v1/b2b/reviews/import` | `import_b2b_reviews` |

## `b2b_scrape` (17 routes)

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/v1/b2b/scrape/logs` | `list_logs` |
| `POST` | `/api/v1/b2b/scrape/run-all` | `trigger_scrape_all` |
| `GET` | `/api/v1/b2b/scrape/targets` | `list_targets` |
| `POST` | `/api/v1/b2b/scrape/targets` | `create_target` |
| `GET` | `/api/v1/b2b/scrape/targets/coverage-plan` | `coverage_plan` |
| `POST` | `/api/v1/b2b/scrape/targets/coverage-plan/disable-poor-fit` | `disable_poor_fit_targets` |
| `POST` | `/api/v1/b2b/scrape/targets/coverage-plan/seed-conditional-probation` | `seed_conditional_probation_targets` |
| `POST` | `/api/v1/b2b/scrape/targets/coverage-plan/seed-missing-core` | `seed_missing_core_targets` |
| `POST` | `/api/v1/b2b/scrape/targets/onboard-vendor` | `onboard_vendor_targets` |
| `GET` | `/api/v1/b2b/scrape/targets/probation-telemetry` | `probation_telemetry` |
| `POST` | `/api/v1/b2b/scrape/targets/probation-telemetry/disable-low-yield` | `disable_low_yield_probation_targets` |
| `POST` | `/api/v1/b2b/scrape/targets/probation-telemetry/promote` | `promote_probation_targets` |
| `POST` | `/api/v1/b2b/scrape/targets/run-probation-batch` | `run_probation_batch` |
| `POST` | `/api/v1/b2b/scrape/targets/source-yield/disable-low-yield` | `disable_low_yield_source_targets` |
| `DELETE` | `/api/v1/b2b/scrape/targets/{target_id}` | `delete_target` |
| `PATCH` | `/api/v1/b2b/scrape/targets/{target_id}` | `update_target` |
| `POST` | `/api/v1/b2b/scrape/targets/{target_id}/run` | `trigger_scrape` |

## `b2b_tenant_dashboard` (53 routes)

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/v1/b2b/tenant/accounts-in-motion-feed` | `list_tenant_accounts_in_motion_feed` |
| `GET` | `/api/v1/b2b/tenant/campaigns` | `list_tenant_campaigns` |
| `POST` | `/api/v1/b2b/tenant/campaigns/generate` | `generate_campaigns` |
| `PATCH` | `/api/v1/b2b/tenant/campaigns/{campaign_id}` | `update_campaign` |
| `GET` | `/api/v1/b2b/tenant/compare-vendor-periods` | `compare_tenant_vendor_periods` |
| `GET` | `/api/v1/b2b/tenant/competitive-sets` | `list_competitive_sets` |
| `POST` | `/api/v1/b2b/tenant/competitive-sets` | `create_competitive_set` |
| `DELETE` | `/api/v1/b2b/tenant/competitive-sets/{competitive_set_id}` | `delete_competitive_set` |
| `PUT` | `/api/v1/b2b/tenant/competitive-sets/{competitive_set_id}` | `update_competitive_set` |
| `GET` | `/api/v1/b2b/tenant/competitive-sets/{competitive_set_id}/plan` | `preview_competitive_set_plan` |
| `POST` | `/api/v1/b2b/tenant/competitive-sets/{competitive_set_id}/run` | `run_competitive_set_now` |
| `GET` | `/api/v1/b2b/tenant/displacement` | `competitor_displacement` |
| `GET` | `/api/v1/b2b/tenant/export/high-intent` | `export_tenant_high_intent` |
| `GET` | `/api/v1/b2b/tenant/export/reviews` | `export_tenant_reviews` |
| `GET` | `/api/v1/b2b/tenant/export/signals` | `export_tenant_signals` |
| `GET` | `/api/v1/b2b/tenant/export/source-health` | `export_tenant_source_health` |
| `GET` | `/api/v1/b2b/tenant/high-intent` | `list_tenant_high_intent` |
| `GET` | `/api/v1/b2b/tenant/leads` | `list_leads` |
| `GET` | `/api/v1/b2b/tenant/leads/{company}` | `get_lead_detail` |
| `GET` | `/api/v1/b2b/tenant/opportunity-dispositions` | `list_opportunity_dispositions` |
| `POST` | `/api/v1/b2b/tenant/opportunity-dispositions` | `set_opportunity_disposition` |
| `POST` | `/api/v1/b2b/tenant/opportunity-dispositions/bulk` | `bulk_set_opportunity_dispositions` |
| `POST` | `/api/v1/b2b/tenant/opportunity-dispositions/remove` | `remove_opportunity_dispositions` |
| `GET` | `/api/v1/b2b/tenant/overview` | `dashboard_overview` |
| `GET` | `/api/v1/b2b/tenant/pain-trends` | `pain_trends` |
| `GET` | `/api/v1/b2b/tenant/pipeline` | `get_tenant_pipeline_status` |
| `POST` | `/api/v1/b2b/tenant/push-to-crm` | `push_to_crm` |
| `GET` | `/api/v1/b2b/tenant/report-subscriptions/{scope_type}/{scope_key}` | `get_report_subscription` |
| `PUT` | `/api/v1/b2b/tenant/report-subscriptions/{scope_type}/{scope_key}` | `upsert_report_subscription` |
| `GET` | `/api/v1/b2b/tenant/reports` | `list_tenant_reports` |
| `POST` | `/api/v1/b2b/tenant/reports/battle-card` | `generate_tenant_battle_card_report` |
| `POST` | `/api/v1/b2b/tenant/reports/company-deep-dive` | `generate_tenant_account_deep_dive_report` |
| `POST` | `/api/v1/b2b/tenant/reports/compare` | `generate_tenant_comparison_report` |
| `POST` | `/api/v1/b2b/tenant/reports/compare-companies` | `generate_tenant_account_comparison_report` |
| `GET` | `/api/v1/b2b/tenant/reports/{report_id}` | `get_tenant_report` |
| `GET` | `/api/v1/b2b/tenant/reviews` | `list_tenant_reviews` |
| `GET` | `/api/v1/b2b/tenant/reviews/{review_id}` | `get_tenant_review` |
| `GET` | `/api/v1/b2b/tenant/signals` | `list_tenant_signals` |
| `GET` | `/api/v1/b2b/tenant/signals/{vendor_name}` | `get_vendor_detail` |
| `GET` | `/api/v1/b2b/tenant/slow-burn-watchlist` | `list_tenant_slow_burn_watchlist` |
| `GET` | `/api/v1/b2b/tenant/vendor-history` | `get_tenant_vendor_history` |
| `GET` | `/api/v1/b2b/tenant/vendors` | `list_tracked_vendors` |
| `POST` | `/api/v1/b2b/tenant/vendors` | `add_tracked_vendor` |
| `GET` | `/api/v1/b2b/tenant/vendors/search` | `search_available_vendors` |
| `DELETE` | `/api/v1/b2b/tenant/vendors/{vendor_name}` | `remove_tracked_vendor` |
| `GET` | `/api/v1/b2b/tenant/watchlist-views` | `list_watchlist_views` |
| `POST` | `/api/v1/b2b/tenant/watchlist-views` | `create_watchlist_view` |
| `DELETE` | `/api/v1/b2b/tenant/watchlist-views/{view_id}` | `delete_watchlist_view` |
| `PUT` | `/api/v1/b2b/tenant/watchlist-views/{view_id}` | `update_watchlist_view` |
| `GET` | `/api/v1/b2b/tenant/watchlist-views/{view_id}/alert-email-log` | `list_watchlist_alert_email_log` |
| `GET` | `/api/v1/b2b/tenant/watchlist-views/{view_id}/alert-events` | `list_watchlist_alert_events` |
| `POST` | `/api/v1/b2b/tenant/watchlist-views/{view_id}/alert-events/deliver-email` | `deliver_watchlist_alert_email` |
| `POST` | `/api/v1/b2b/tenant/watchlist-views/{view_id}/alert-events/evaluate` | `evaluate_watchlist_alert_events` |

## `b2b_vendor_briefing` (13 routes)

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/v1/b2b/briefings` | `list_briefings` |
| `POST` | `/api/v1/b2b/briefings/bulk-approve` | `bulk_approve_briefings` |
| `POST` | `/api/v1/b2b/briefings/bulk-reject` | `bulk_reject_briefings` |
| `POST` | `/api/v1/b2b/briefings/checkout` | `vendor_checkout` |
| `GET` | `/api/v1/b2b/briefings/checkout-session` | `checkout_session_info` |
| `GET` | `/api/v1/b2b/briefings/export` | `export_briefings` |
| `POST` | `/api/v1/b2b/briefings/gate` | `briefing_gate` |
| `POST` | `/api/v1/b2b/briefings/generate` | `generate_briefing` |
| `POST` | `/api/v1/b2b/briefings/preview` | `preview_briefing` |
| `GET` | `/api/v1/b2b/briefings/report-data` | `report_data` |
| `GET` | `/api/v1/b2b/briefings/review-queue` | `briefing_review_queue` |
| `GET` | `/api/v1/b2b/briefings/review-queue/summary` | `briefing_review_summary` |
| `POST` | `/api/v1/b2b/briefings/send-batch` | `send_batch` |

## `b2b_win_loss` (5 routes)

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/v1/b2b/predict/win-loss` | `predict_win_loss` |
| `POST` | `/api/v1/b2b/predict/win-loss/compare` | `compare_win_loss` |
| `GET` | `/api/v1/b2b/predict/win-loss/recent` | `list_recent_predictions` |
| `GET` | `/api/v1/b2b/predict/win-loss/{prediction_id}` | `get_prediction` |
| `GET` | `/api/v1/b2b/predict/win-loss/{prediction_id}/csv` | `export_prediction_csv` |

## `billing` (4 routes)

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/v1/billing/checkout` | `create_checkout` |
| `POST` | `/api/v1/billing/portal` | `create_portal` |
| `GET` | `/api/v1/billing/status` | `billing_status` |
| `POST` | `/webhooks/stripe` | `stripe_webhook` |

## `blog_admin` (9 routes)

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/v1/admin/blog/drafts` | `list_drafts` |
| `GET` | `/api/v1/admin/blog/drafts/summary` | `draft_summary` |
| `GET` | `/api/v1/admin/blog/drafts/{draft_id}` | `get_draft` |
| `PATCH` | `/api/v1/admin/blog/drafts/{draft_id}` | `update_draft` |
| `GET` | `/api/v1/admin/blog/drafts/{draft_id}/evidence` | `get_draft_evidence` |
| `POST` | `/api/v1/admin/blog/drafts/{draft_id}/publish` | `publish_draft` |
| `POST` | `/api/v1/admin/blog/generate` | `generate_post` |
| `GET` | `/api/v1/admin/blog/quality-diagnostics` | `blog_quality_diagnostics` |
| `GET` | `/api/v1/admin/blog/quality-trends` | `blog_quality_trends` |

## `blog_public` (2 routes)

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/v1/blog/published` | `list_published_posts` |
| `GET` | `/api/v1/blog/published/{slug}` | `get_published_post` |

## `campaign_webhooks` (2 routes)

| Method | Path | Handler |
|---|---|---|
| `POST` | `/webhooks/campaign-email` | `campaign_email_webhook` |
| `GET` | `/webhooks/unsubscribe` | `unsubscribe` |

## `comms.call_actions` (10 routes)

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/v1/comms/call-actions/{transcript_id}/approve-plan` | `approve_plan` |
| `POST` | `/api/v1/comms/call-actions/{transcript_id}/book` | `book_appointment` |
| `POST` | `/api/v1/comms/call-actions/{transcript_id}/discard` | `discard_draft` |
| `POST` | `/api/v1/comms/call-actions/{transcript_id}/draft-email` | `draft_email` |
| `POST` | `/api/v1/comms/call-actions/{transcript_id}/draft-sms` | `draft_sms_confirmation` |
| `POST` | `/api/v1/comms/call-actions/{transcript_id}/reject-plan` | `reject_plan` |
| `POST` | `/api/v1/comms/call-actions/{transcript_id}/send-email` | `send_drafted_email` |
| `POST` | `/api/v1/comms/call-actions/{transcript_id}/send-sms` | `send_drafted_sms` |
| `POST` | `/api/v1/comms/call-actions/{transcript_id}/sms` | `send_sms` |
| `GET` | `/api/v1/comms/call-actions/{transcript_id}/view` | `view_transcript` |

## `comms.management` (9 routes)

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/v1/comms/appointments` | `book_appointment` |
| `DELETE` | `/api/v1/comms/appointments/{appointment_id}` | `cancel_appointment` |
| `POST` | `/api/v1/comms/availability` | `check_availability` |
| `POST` | `/api/v1/comms/calls` | `make_call` |
| `GET` | `/api/v1/comms/contexts` | `list_contexts` |
| `GET` | `/api/v1/comms/contexts/{context_id}` | `get_context` |
| `POST` | `/api/v1/comms/recordings/reconcile` | `reconcile_recordings` |
| `POST` | `/api/v1/comms/sms` | `send_sms` |
| `GET` | `/api/v1/comms/status` | `get_comms_status` |

## `comms.webhooks` (12 routes)

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/v1/comms/voice/conversation` | `handle_conversation` |
| `POST` | `/api/v1/comms/voice/dial-status` | `handle_dial_status` |
| `POST` | `/api/v1/comms/voice/inbound` | `handle_inbound_call` |
| `POST` | `/api/v1/comms/voice/outbound` | `handle_outbound_call` |
| `POST` | `/api/v1/comms/voice/recording-status` | `handle_recording_status` |
| `POST` | `/api/v1/comms/voice/sip-outbound-swml` | `handle_sip_outbound_swml` |
| `POST` | `/api/v1/comms/voice/sms/inbound` | `handle_inbound_sms` |
| `POST` | `/api/v1/comms/voice/sms/status` | `handle_sms_status` |
| `POST` | `/api/v1/comms/voice/status` | `handle_call_status` |
| `WS` | `/api/v1/comms/voice/stream/{call_sid}` | `handle_audio_stream` |
| `POST` | `/api/v1/comms/voice/swml-debug` | `swml_debug` |
| `POST` | `/api/v1/comms/voice/voicemail` | `handle_voicemail` |

## `consumer_dashboard` (35 routes)

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/v1/consumer/dashboard/asins` | `list_tracked_asins` |
| `POST` | `/api/v1/consumer/dashboard/asins` | `add_tracked_asin` |
| `GET` | `/api/v1/consumer/dashboard/asins/search` | `search_available_asins` |
| `DELETE` | `/api/v1/consumer/dashboard/asins/{asin}` | `remove_tracked_asin` |
| `GET` | `/api/v1/consumer/dashboard/brand-correlation` | `get_brand_correlation` |
| `GET` | `/api/v1/consumer/dashboard/brand-history` | `brand_history` |
| `GET` | `/api/v1/consumer/dashboard/brand-registry` | `list_brand_registry` |
| `POST` | `/api/v1/consumer/dashboard/brand-registry` | `add_brand_to_registry_endpoint` |
| `POST` | `/api/v1/consumer/dashboard/brand-registry/{brand}/aliases` | `add_brand_alias_endpoint` |
| `GET` | `/api/v1/consumer/dashboard/brands` | `list_brands` |
| `GET` | `/api/v1/consumer/dashboard/brands/compare` | `compare_brands` |
| `GET` | `/api/v1/consumer/dashboard/brands/{brand_name}` | `get_brand_detail` |
| `GET` | `/api/v1/consumer/dashboard/brands/{brand_name}/pdf` | `export_brand_report_pdf` |
| `GET` | `/api/v1/consumer/dashboard/categories` | `list_categories` |
| `GET` | `/api/v1/consumer/dashboard/change-events` | `list_change_events` |
| `GET` | `/api/v1/consumer/dashboard/change-events/summary` | `change_events_summary` |
| `GET` | `/api/v1/consumer/dashboard/concurrent-events` | `list_concurrent_events` |
| `GET` | `/api/v1/consumer/dashboard/corrections` | `list_consumer_corrections` |
| `POST` | `/api/v1/consumer/dashboard/corrections` | `create_consumer_correction` |
| `GET` | `/api/v1/consumer/dashboard/corrections/stats` | `consumer_correction_stats` |
| `GET` | `/api/v1/consumer/dashboard/corrections/{correction_id}` | `get_consumer_correction` |
| `POST` | `/api/v1/consumer/dashboard/corrections/{correction_id}/revert` | `revert_consumer_correction` |
| `GET` | `/api/v1/consumer/dashboard/displacement-edges` | `list_displacement_edges` |
| `GET` | `/api/v1/consumer/dashboard/displacement-history` | `get_displacement_history` |
| `GET` | `/api/v1/consumer/dashboard/export/brands` | `export_brands` |
| `GET` | `/api/v1/consumer/dashboard/export/pain-points` | `export_pain_points` |
| `GET` | `/api/v1/consumer/dashboard/export/reviews` | `export_reviews` |
| `GET` | `/api/v1/consumer/dashboard/features` | `get_feature_gaps` |
| `GET` | `/api/v1/consumer/dashboard/flows` | `get_competitive_flows` |
| `GET` | `/api/v1/consumer/dashboard/fuzzy-brand-search` | `fuzzy_brand_search` |
| `GET` | `/api/v1/consumer/dashboard/pipeline` | `get_pipeline_status` |
| `GET` | `/api/v1/consumer/dashboard/reports/{report_id}/pdf` | `export_market_report_pdf` |
| `GET` | `/api/v1/consumer/dashboard/reviews` | `search_reviews` |
| `GET` | `/api/v1/consumer/dashboard/reviews/{review_id}` | `get_review` |
| `GET` | `/api/v1/consumer/dashboard/safety` | `get_safety_signals` |

## `contacts` (2 routes)

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/v1/comms/calls/search` | `search_calls` |
| `GET` | `/api/v1/contacts/{contact_id}/timeline` | `contact_timeline` |

## `devices.control` (5 routes)

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/v1/devices/` | `list_devices` |
| `POST` | `/api/v1/devices/intent` | `execute_intent` |
| `GET` | `/api/v1/devices/{device_id}` | `get_device` |
| `POST` | `/api/v1/devices/{device_id}/action` | `execute_device_action` |
| `GET` | `/api/v1/devices/{device_id}/state` | `get_device_state` |

## `edge.websocket` (2 routes)

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/v1/ws/edge/status` | `get_edge_status` |
| `WS` | `/api/v1/ws/edge/{location_id}` | `edge_websocket` |

## `email_actions` (5 routes)

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/v1/email/actions/{gmail_message_id}/archive` | `archive_email` |
| `POST` | `/api/v1/email/actions/{gmail_message_id}/escalate` | `escalate_email` |
| `POST` | `/api/v1/email/actions/{gmail_message_id}/quote` | `generate_quote` |
| `POST` | `/api/v1/email/actions/{gmail_message_id}/send-info` | `send_info` |
| `POST` | `/api/v1/email/actions/{gmail_message_id}/slots` | `show_slots` |

## `email_drafts` (8 routes)

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/v1/email/drafts/` | `list_drafts` |
| `POST` | `/api/v1/email/drafts/generate/{gmail_message_id}` | `generate_draft` |
| `GET` | `/api/v1/email/drafts/{draft_id}` | `get_draft` |
| `POST` | `/api/v1/email/drafts/{draft_id}/approve` | `approve_draft` |
| `POST` | `/api/v1/email/drafts/{draft_id}/edit` | `edit_draft` |
| `POST` | `/api/v1/email/drafts/{draft_id}/redraft` | `redraft` |
| `POST` | `/api/v1/email/drafts/{draft_id}/reject` | `reject_draft` |
| `POST` | `/api/v1/email/drafts/{draft_id}/skip` | `skip_draft` |

## `health` (2 routes)

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/v1/health` | `health` |
| `GET` | `/api/v1/ping` | `ping` |

## `identity` (5 routes)

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/v1/identity/` | `list_identities` |
| `POST` | `/api/v1/identity/` | `create_identity` |
| `GET` | `/api/v1/identity/names` | `get_identity_names` |
| `DELETE` | `/api/v1/identity/{name}` | `delete_person` |
| `DELETE` | `/api/v1/identity/{name}/{modality}` | `delete_identity` |

## `inbox_rules` (7 routes)

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/v1/email/inbox-rules/` | `list_rules` |
| `POST` | `/api/v1/email/inbox-rules/` | `create_rule` |
| `POST` | `/api/v1/email/inbox-rules/reorder` | `reorder_rules` |
| `POST` | `/api/v1/email/inbox-rules/test` | `test_rules` |
| `DELETE` | `/api/v1/email/inbox-rules/{rule_id}` | `delete_rule` |
| `GET` | `/api/v1/email/inbox-rules/{rule_id}` | `get_rule` |
| `PUT` | `/api/v1/email/inbox-rules/{rule_id}` | `update_rule` |

## `intelligence` (9 routes)

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/v1/intelligence/approvals` | `list_pending_approvals` |
| `GET` | `/api/v1/intelligence/approvals/{approval_id}` | `get_approval` |
| `POST` | `/api/v1/intelligence/approvals/{approval_id}/approve` | `approve_intervention` |
| `POST` | `/api/v1/intelligence/approvals/{approval_id}/reject` | `reject_intervention` |
| `POST` | `/api/v1/intelligence/intervention` | `run_intervention` |
| `GET` | `/api/v1/intelligence/pressure` | `list_pressure_baselines` |
| `POST` | `/api/v1/intelligence/report` | `generate_report` |
| `GET` | `/api/v1/intelligence/reports` | `list_reports` |
| `GET` | `/api/v1/intelligence/reports/{report_id}` | `get_report` |

## `invoicing.actions` (4 routes)

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/v1/invoicing/{invoice_id}` | `view_invoice` |
| `POST` | `/api/v1/invoicing/{invoice_id}/mark-paid` | `mark_paid` |
| `POST` | `/api/v1/invoicing/{invoice_id}/send` | `send_invoice` |
| `POST` | `/api/v1/invoicing/{invoice_id}/send-reminder` | `send_reminder` |

## `llm` (6 routes)

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/v1/llm/activate` | `activate_llm` |
| `GET` | `/api/v1/llm/available` | `list_available` |
| `POST` | `/api/v1/llm/chat` | `chat` |
| `POST` | `/api/v1/llm/deactivate` | `deactivate_llm` |
| `POST` | `/api/v1/llm/generate` | `generate_text` |
| `GET` | `/api/v1/llm/status` | `get_status` |

## `ollama_compat` (4 routes)

| Method | Path | Handler |
|---|---|---|
| `GET` | `/` | `root` |
| `POST` | `/api/chat` | `chat` |
| `GET` | `/api/tags` | `list_models` |
| `GET` | `/api/version` | `version` |

## `openai_compat` (1 route)

| Method | Path | Handler |
|---|---|---|
| `POST` | `/v1/chat/completions` | `chat_completions` |

## `orchestrated.websocket` (1 route)

| Method | Path | Handler |
|---|---|---|
| `WS` | `/api/v1/ws/orchestrated` | `orchestrated_websocket` |

## `pipeline_visibility` (14 routes)

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/v1/pipeline/visibility/attempts` | `get_artifact_attempts` |
| `GET` | `/api/v1/pipeline/visibility/dedup-decisions` | `get_dedup_decisions` |
| `GET` | `/api/v1/pipeline/visibility/events` | `get_visibility_events` |
| `GET` | `/api/v1/pipeline/visibility/extraction-health` | `get_extraction_health` |
| `GET` | `/api/v1/pipeline/visibility/quarantines` | `get_enrichment_quarantines` |
| `GET` | `/api/v1/pipeline/visibility/queue` | `get_visibility_queue` |
| `GET` | `/api/v1/pipeline/visibility/review-actions` | `get_review_actions` |
| `POST` | `/api/v1/pipeline/visibility/reviews/{review_id}/resolve` | `resolve_review` |
| `GET` | `/api/v1/pipeline/visibility/summary` | `get_visibility_summary` |
| `GET` | `/api/v1/pipeline/visibility/synthesis-validation` | `get_synthesis_validation_results` |
| `GET` | `/api/v1/pipeline/visibility/watchlist-delivery` | `get_watchlist_delivery_ops` |
| `GET` | `/api/v1/pipeline/visibility/watchlist-delivery/views/{view_id}` | `get_watchlist_delivery_view_detail` |
| `POST` | `/api/v1/pipeline/visibility/watchlist-delivery/views/{view_id}/deliver-now` | `deliver_watchlist_view_now` |
| `POST` | `/api/v1/pipeline/visibility/watchlist-delivery/views/{view_id}/disable-email` | `disable_watchlist_view_email` |

## `presence` (2 routes)

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/v1/presence/history` | `get_presence_history` |
| `GET` | `/api/v1/presence/status` | `get_presence_status` |

## `proactive_actions` (4 routes)

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/v1/actions/` | `list_actions` |
| `DELETE` | `/api/v1/actions/{action_id}` | `delete_action` |
| `POST` | `/api/v1/actions/{action_id}/dismiss` | `dismiss_action` |
| `POST` | `/api/v1/actions/{action_id}/done` | `mark_action_done` |

## `prospects` (9 routes)

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/v1/b2b/prospects` | `list_prospects` |
| `GET` | `/api/v1/b2b/prospects/company-overrides` | `list_company_overrides` |
| `POST` | `/api/v1/b2b/prospects/company-overrides` | `create_or_update_company_override` |
| `POST` | `/api/v1/b2b/prospects/company-overrides/bootstrap` | `bootstrap_company_overrides` |
| `DELETE` | `/api/v1/b2b/prospects/company-overrides/{override_id}` | `remove_company_override` |
| `GET` | `/api/v1/b2b/prospects/export` | `export_prospects` |
| `GET` | `/api/v1/b2b/prospects/manual-queue` | `list_manual_prospect_queue` |
| `POST` | `/api/v1/b2b/prospects/manual-queue/{queue_id}/resolve` | `resolve_manual_prospect_queue` |
| `GET` | `/api/v1/b2b/prospects/stats` | `prospect_stats` |

## `query.text` (1 route)

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/v1/query/text` | `query_text` |

## `reasoning` (4 routes)

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/v1/reasoning/events` | `list_events` |
| `GET` | `/api/v1/reasoning/locks` | `list_locks` |
| `POST` | `/api/v1/reasoning/process/{event_id}` | `process_event` |
| `GET` | `/api/v1/reasoning/queue` | `list_queue` |

## `recognition` (18 routes)

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/v1/recognition/enroll/face` | `enroll_face` |
| `POST` | `/api/v1/recognition/enroll/gait/cancel` | `cancel_gait_enrollment` |
| `POST` | `/api/v1/recognition/enroll/gait/complete` | `complete_gait_enrollment` |
| `POST` | `/api/v1/recognition/enroll/gait/frame` | `add_gait_frame` |
| `POST` | `/api/v1/recognition/enroll/gait/start` | `start_gait_enrollment` |
| `GET` | `/api/v1/recognition/enroll/gait/status` | `get_gait_enrollment_status` |
| `GET` | `/api/v1/recognition/events` | `get_recognition_events` |
| `POST` | `/api/v1/recognition/identify/combined` | `identify_combined` |
| `POST` | `/api/v1/recognition/identify/face` | `identify_face` |
| `POST` | `/api/v1/recognition/identify/gait/frame` | `add_gait_identify_frame` |
| `POST` | `/api/v1/recognition/identify/gait/match` | `match_gait` |
| `POST` | `/api/v1/recognition/identify/gait/start` | `start_gait_identification` |
| `GET` | `/api/v1/recognition/persons` | `list_persons` |
| `POST` | `/api/v1/recognition/persons` | `create_person` |
| `DELETE` | `/api/v1/recognition/persons/{person_id}` | `delete_person` |
| `GET` | `/api/v1/recognition/persons/{person_id}` | `get_person` |
| `PATCH` | `/api/v1/recognition/persons/{person_id}` | `update_person` |
| `GET` | `/api/v1/recognition/persons/{person_id}/embeddings` | `get_person_embeddings` |

## `security` (6 routes)

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/v1/security/assets` | `list_security_assets` |
| `POST` | `/api/v1/security/assets/observe` | `observe_security_asset` |
| `GET` | `/api/v1/security/assets/persisted` | `list_persisted_security_assets` |
| `GET` | `/api/v1/security/assets/telemetry` | `get_security_asset_telemetry` |
| `GET` | `/api/v1/security/status` | `get_security_status` |
| `GET` | `/api/v1/security/threats/summary` | `get_security_threat_summary` |

## `seller_campaigns` (9 routes)

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/v1/seller/campaigns` | `list_seller_campaigns` |
| `POST` | `/api/v1/seller/campaigns/generate` | `trigger_generation` |
| `GET` | `/api/v1/seller/intelligence` | `list_category_intelligence` |
| `POST` | `/api/v1/seller/intelligence/refresh` | `refresh_category_intelligence` |
| `GET` | `/api/v1/seller/targets` | `list_seller_targets` |
| `POST` | `/api/v1/seller/targets` | `create_seller_target` |
| `DELETE` | `/api/v1/seller/targets/{target_id}` | `delete_seller_target` |
| `GET` | `/api/v1/seller/targets/{target_id}` | `get_seller_target` |
| `PATCH` | `/api/v1/seller/targets/{target_id}` | `update_seller_target` |

## `session` (6 routes)

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/v1/session/continue` | `continue_session` |
| `POST` | `/api/v1/session/create` | `create_session` |
| `GET` | `/api/v1/session/status/db` | `get_db_status` |
| `GET` | `/api/v1/session/{session_id}` | `get_session` |
| `POST` | `/api/v1/session/{session_id}/close` | `close_session` |
| `GET` | `/api/v1/session/{session_id}/history` | `get_session_history` |

## `settings` (14 routes)

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/v1/settings/daily` | `get_daily_settings` |
| `PATCH` | `/api/v1/settings/daily` | `update_daily_settings` |
| `GET` | `/api/v1/settings/email` | `get_email_settings` |
| `PATCH` | `/api/v1/settings/email` | `update_email_settings` |
| `GET` | `/api/v1/settings/integrations` | `get_integration_settings` |
| `PATCH` | `/api/v1/settings/integrations` | `update_integration_settings` |
| `GET` | `/api/v1/settings/intelligence` | `get_intelligence_settings` |
| `PATCH` | `/api/v1/settings/intelligence` | `update_intelligence_settings` |
| `GET` | `/api/v1/settings/llm` | `get_llm_settings` |
| `PATCH` | `/api/v1/settings/llm` | `update_llm_settings` |
| `GET` | `/api/v1/settings/notifications` | `get_notification_settings` |
| `PATCH` | `/api/v1/settings/notifications` | `update_notification_settings` |
| `GET` | `/api/v1/settings/voice` | `get_voice_settings` |
| `PATCH` | `/api/v1/settings/voice` | `update_voice_settings` |

## `speaker` (8 routes)

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/v1/speaker/enroll/cancel` | `cancel_enrollment` |
| `POST` | `/api/v1/speaker/enroll/complete` | `complete_enrollment` |
| `POST` | `/api/v1/speaker/enroll/sample` | `add_enrollment_sample` |
| `POST` | `/api/v1/speaker/enroll/start` | `start_enrollment` |
| `GET` | `/api/v1/speaker/enrolled` | `list_enrolled_users` |
| `GET` | `/api/v1/speaker/status` | `get_status` |
| `POST` | `/api/v1/speaker/verify` | `verify_speaker` |
| `DELETE` | `/api/v1/speaker/{user_id}` | `delete_enrollment` |

## `system` (1 route)

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/v1/system/stats` | `get_system_stats` |

## `universal_scrape` (7 routes)

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/v1/scraper/jobs` | `list_jobs` |
| `POST` | `/api/v1/scraper/jobs` | `create_job` |
| `POST` | `/api/v1/scraper/jobs/from-file` | `create_job_from_file` |
| `DELETE` | `/api/v1/scraper/jobs/{job_id}` | `delete_job` |
| `GET` | `/api/v1/scraper/jobs/{job_id}` | `get_job` |
| `POST` | `/api/v1/scraper/jobs/{job_id}/cancel` | `cancel_job` |
| `GET` | `/api/v1/scraper/jobs/{job_id}/results` | `get_results` |

## `vendor_targets` (7 routes)

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/v1/b2b/vendor-targets` | `list_vendor_targets` |
| `POST` | `/api/v1/b2b/vendor-targets` | `create_vendor_target` |
| `DELETE` | `/api/v1/b2b/vendor-targets/{target_id}` | `delete_vendor_target` |
| `GET` | `/api/v1/b2b/vendor-targets/{target_id}` | `get_vendor_target` |
| `PUT` | `/api/v1/b2b/vendor-targets/{target_id}` | `update_vendor_target` |
| `POST` | `/api/v1/b2b/vendor-targets/{target_id}/claim` | `claim_vendor_target` |
| `POST` | `/api/v1/b2b/vendor-targets/{target_id}/generate-report` | `generate_target_report` |

## `video` (6 routes)

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/v1/video/classes` | `list_classes` |
| `GET` | `/api/v1/video/rtsp/{camera_id}` | `stream_rtsp` |
| `GET` | `/api/v1/video/snapshot/webcam` | `snapshot_webcam` |
| `GET` | `/api/v1/video/webcam` | `stream_webcam` |
| `GET` | `/api/v1/video/webcam/recognition` | `stream_webcam_recognition_deprecated` |
| `GET` | `/api/v1/video/webcam/recognition/multitrack` | `stream_webcam_recognition_multitrack_deprecated` |

## `vision` (16 routes)

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/v1/vision/alerts` | `get_alerts` |
| `POST` | `/api/v1/vision/alerts/acknowledge-all` | `acknowledge_all_alerts` |
| `GET` | `/api/v1/vision/alerts/rules` | `list_alert_rules` |
| `POST` | `/api/v1/vision/alerts/rules` | `create_alert_rule` |
| `DELETE` | `/api/v1/vision/alerts/rules/{rule_name}` | `delete_alert_rule` |
| `POST` | `/api/v1/vision/alerts/rules/{rule_name}/disable` | `disable_alert_rule` |
| `POST` | `/api/v1/vision/alerts/rules/{rule_name}/enable` | `enable_alert_rule` |
| `GET` | `/api/v1/vision/alerts/stats` | `get_alert_stats` |
| `GET` | `/api/v1/vision/alerts/unacknowledged/count` | `get_unacknowledged_count` |
| `POST` | `/api/v1/vision/alerts/{alert_id}/acknowledge` | `acknowledge_alert` |
| `GET` | `/api/v1/vision/cameras` | `get_active_cameras` |
| `GET` | `/api/v1/vision/events` | `get_vision_events` |
| `DELETE` | `/api/v1/vision/events/cleanup` | `cleanup_old_events` |
| `GET` | `/api/v1/vision/events/counts` | `get_event_counts` |
| `GET` | `/api/v1/vision/events/range` | `get_events_in_range` |
| `GET` | `/api/v1/vision/nodes` | `get_known_nodes` |
