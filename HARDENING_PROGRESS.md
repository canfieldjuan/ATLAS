# Atlas Security Hardening Progress

Systematic audit and hardening of the entire `atlas_brain/` codebase.

## Fix Categories

| Code | Pattern | Fix |
|------|---------|-----|
| **H** | `str(e)` in user-facing responses (ToolResult, MCP, API, TTS) | Generic message; log with `exc_info=True` |
| **M1** | `datetime.utcnow()` (deprecated Python 3.12+) | `datetime.now(timezone.utc)` |
| **M2** | Unbounded SELECT (no LIMIT) | Add safety LIMIT |
| **M3** | Unclamped `limit` params | `min(limit, max_val)` |
| **M4** | Unbounded in-memory growth (dicts, lists, queues) | Max-size cap + eviction |
| **M5** | `str(e)` in logged messages leaking sensitive data | `type(e).__name__` |
| **L1** | Unicode in Python source (em-dash, arrows, smart quotes) | ASCII equivalents |
| **L2** | Silent `except: pass` | `logger.debug(...)` |
| **L3** | Missing timeouts (network, subprocess, executor) | Explicit timeout |
| **L4** | Miscellaneous (log rotation, parse safety, etc.) | Case-by-case |

---

## Completed Rounds

### Round 1-5: API Layer (all endpoints)
**Commit**: multiple, ending at `509b640`
**Scope**: Every file in `atlas_brain/api/`
- All API endpoints hardened

### Round 6: Services + Storage
**Commit**: `f2aa83e`
**Files (14)**:
- `services/email_provider.py` -- IMAP socket timeout (30s), silent catch logging, Unicode
- `services/llm/anthropic.py` -- proper `.close()` on sync+async clients in `unload()`
- `reasoning/event_bus.py` -- Queue maxsize 10,000, explicit SELECT columns, subscriber error logging
- `services/mcp_client.py` -- error details masked from ToolResult
- `services/llm/ollama.py` -- full response text demoted from INFO to DEBUG
- `services/speaker_id/service.py` -- enrollment sessions capped at 50
- `storage/repositories/speaker.py` -- LIMIT 500 on both get_all queries
- `reasoning/entity_locks.py` -- drain_queue CTE + LIMIT 200
- `services/campaign_sender.py` -- Resend error body truncated to 500 chars
- `comms/real_services.py` -- Resend executor timeout 30s
- `storage/database.py` -- `command_timeout=60` on standalone connections
- `reasoning/agent.py` -- per-event 120s timeout in process_drained_events
- `services/reminders.py` -- silent catch logged as warning
- `storage/repositories/email.py` -- limit clamped to max 500

### Round 7: Agents, Tools, Voice, Autonomous
**Commit**: `6fb614f`
**Files (14)**:
- `agents/graphs/atlas.py` -- TTS error masked, ActionResult str(e) masked (3x)
- `agents/graphs/home.py` -- TTS error masked, ActionResult str(e) masked (3x)
- `agents/graphs/presence.py` -- error dict str(e) masked (6x)
- `agents/graphs/receptionist.py` -- tool_results + graph error masked
- `agents/graphs/streaming.py` -- on_llm_error metadata masked
- `tools/registry.py` -- ToolResult str(e) masked
- `tools/calendar.py` -- ToolResult str(e) masked (2x), hours_ahead clamped [1,168], duration clamped [1,1440]
- `voice/pipeline.py` -- Unicode cleaned (5x), subprocess timeout=30
- `voice/launcher.py` -- Unicode cleaned (4x)
- `voice/free_mode.py` -- Unicode cleaned (4x)
- `autonomous/scheduler.py` -- DB error persistence truncated, silent catches logged (4x)
- `autonomous/tasks/campaign_send.py` -- audit log error truncated, return genericized
- `autonomous/hooks.py` -- execution dict capped at 500
- `capabilities/homeassistant.py` -- silent catches logged (2x)

### Round 8: MCP Servers, Remaining Agents/Capabilities/Services
**Commit**: `2a57024`
**Files (18)**:
- `mcp/crm_server.py` -- 10 str(exc) -> "Internal error"
- `mcp/email_server.py` -- 9 str(exc) -> "Internal error"
- `mcp/twilio_server.py` -- 11 str(exc) -> "Internal error"
- `mcp/calendar_server.py` -- 8 str(exc) -> "Internal error"
- `mcp/intelligence_server.py` -- 8 str(exc) -> "Internal error"
- `mcp/b2b_churn_server.py` -- 10 str(exc) -> "Internal error"
- `mcp/invoicing_server.py` -- 15 str(exc) -> "Internal error"
- `capabilities/actions.py` -- error message genericized
- `agents/graphs/email.py` -- context capped [-40:]
- `agents/graphs/email_query.py` -- context capped [-40:]
- `agents/graphs/calendar.py` -- context capped [-40:]
- `agents/graphs/call.py` -- context capped [-40:], silent catch logged
- `storage/repositories/scheduled_task.py` -- limit clamped to 200
- `storage/repositories/session.py` -- datetime.utcnow (7x), limit clamped to 500
- `reasoning/graph.py` -- action error uses type(exc).__name__
- `services/intent_router.py` -- fallback log rotation 10MB, executor timeout 10s, Unicode cleaned (6x)
- `capabilities/device_resolver.py` -- executor timeouts 10s (2x)
- `services/tracing.py` -- resp.text leak removed, str(e) masked

### Round 9: Models, State Cache, WebSocket
**Commit**: `62ea48a`
**Files (3)**:
- `storage/models.py` -- datetime.utcnow replaced across 30 field defaults (15 dataclasses) via `_utcnow()` helper
- `capabilities/state_cache.py` -- _MAX_ENTRIES=2000 eviction cap
- `capabilities/backends/homeassistant_ws.py` -- open_timeout=15 on websockets.connect()

### Round 10: Remaining Tools + Storage Repos
**Commit**: `ace0a11`
**Files (22)**:
- `tools/digest.py` -- ToolResult str(e) masked (2x)
- `tools/display.py` -- ToolResult str(e) masked (2x), ProcessLookupError logged
- `tools/email.py` -- ToolResult str(e) masked (2x), limit clamped to 200
- `tools/location.py` -- ToolResult str(e) masked
- `tools/notify.py` -- ToolResult str(e) masked
- `tools/presence.py` -- error=str(e) masked (7x)
- `tools/reminder.py` -- ToolResult str(e) masked (4x)
- `tools/scheduling.py` -- ToolResult str(e) masked (5x), Unicode cleaned (4x)
- `tools/time.py` -- ToolResult str(e) masked
- `tools/traffic.py` -- ToolResult str(e) masked
- `tools/weather.py` -- ToolResult str(e) masked
- `tools/security.py` -- datetime.now() -> datetime.now(timezone.utc) (2x)
- `storage/repositories/device.py` -- datetime.utcnow (2x), LIMIT 500 (3 queries)
- `storage/repositories/feedback.py` -- datetime.utcnow (4x)
- `storage/repositories/profile.py` -- datetime.utcnow (2x)
- `storage/repositories/unified_alerts.py` -- datetime.utcnow (4x)
- `storage/repositories/vision.py` -- datetime.utcnow (2x), LIMIT 5000 (2 queries)
- `storage/repositories/appointment.py` -- LIMIT 1000
- `storage/repositories/business_context.py` -- LIMIT 100, Unicode cleaned
- `storage/repositories/customer_service.py` -- LIMIT 500 (4 queries)
- `storage/repositories/identity.py` -- LIMIT 500 (3 queries)
- `storage/repositories/invoice.py` -- LIMIT 500 + LIMIT 200

---

## Remaining (Not Yet Audited)

### agents/ (6 files)
- `agents/entity_tracker.py`
- `agents/graphs/booking.py`
- `agents/graphs/reminder.py`
- `agents/graphs/state.py`
- `agents/graphs/workflow_state.py`
- `agents/interface.py`
- `agents/memory.py`
- `agents/protocols.py`
- `agents/tools.py`
- `agents/security.py`

### alerts/ (2 files)
- `alerts/delivery.py`
- `alerts/manager.py`

### auth/ (3 files)
- `auth/dependencies.py`
- `auth/jwt.py`
- `auth/passwords.py`

### autonomous/tasks/ (~40 files)
- `autonomous/event_queue.py`
- `autonomous/presence.py`
- `autonomous/runner.py`
- `autonomous/tasks/action_escalation.py`
- `autonomous/tasks/amazon_seller_campaign_generation.py`
- `autonomous/tasks/anomaly_detection.py`
- `autonomous/tasks/article_enrichment.py`
- `autonomous/tasks/b2b_campaign_generation.py`
- `autonomous/tasks/b2b_churn_intelligence.py`
- `autonomous/tasks/b2b_enrichment.py`
- `autonomous/tasks/b2b_scrape_intake.py`
- `autonomous/tasks/calendar_reminder.py`
- `autonomous/tasks/campaign_analytics_refresh.py`
- `autonomous/tasks/campaign_audit.py`
- `autonomous/tasks/campaign_sequence_progression.py`
- `autonomous/tasks/campaign_suppression.py`
- `autonomous/tasks/competitive_intelligence.py`
- `autonomous/tasks/complaint_analysis.py`
- `autonomous/tasks/complaint_content_generation.py`
- `autonomous/tasks/complaint_enrichment.py`
- `autonomous/tasks/daily_intelligence.py`
- `autonomous/tasks/deep_enrichment.py`
- `autonomous/tasks/departure_auto_fix.py`
- `autonomous/tasks/departure_check.py`
- `autonomous/tasks/device_health.py`
- `autonomous/tasks/email_auto_approve.py`
- `autonomous/tasks/email_backfill.py`
- `autonomous/tasks/email_classifier.py`
- `autonomous/tasks/email_draft.py`
- `autonomous/tasks/email_intake.py`
- `autonomous/tasks/email_stale_check.py`
- `autonomous/tasks/gmail_digest.py`
- `autonomous/tasks/invoice_overdue_check.py`
- `autonomous/tasks/invoice_payment_reminders.py`
- `autonomous/tasks/market_intake.py`
- `autonomous/tasks/model_swap.py`
- `autonomous/tasks/monthly_invoice_generation.py`
- `autonomous/tasks/morning_briefing.py`
- `autonomous/tasks/news_intake.py`
- `autonomous/tasks/news_intelligence.py`
- `autonomous/tasks/pattern_learning.py`
- `autonomous/tasks/_pipelines.py`
- `autonomous/tasks/preference_learning.py`
- `autonomous/tasks/proactive_actions.py`
- `autonomous/tasks/reasoning_reflection.py`
- `autonomous/tasks/reasoning_tick.py`
- `autonomous/tasks/security_summary.py`
- `autonomous/tasks/weather_traffic_alerts.py`

### capabilities/ (6 files)
- `capabilities/devices/lights.py`
- `capabilities/devices/media.py`
- `capabilities/devices/mock.py`
- `capabilities/devices/switches.py`
- `capabilities/backends/base.py`
- `capabilities/backends/mqtt.py`
- `capabilities/intent_parser.py`
- `capabilities/protocols.py`

### comms/ (10 files)
- `comms/action_planner.py`
- `comms/call_intelligence.py`
- `comms/context.py`
- `comms/invoice_detector.py`
- `comms/personaplex_processor.py`
- `comms/protocols.py`
- `comms/service.py`
- `comms/services.py`
- `comms/sms_intelligence.py`
- `comms/tool_bridge.py`

### discovery/ (3 files)
- `discovery/scanners/base.py`
- `discovery/scanners/mdns.py`
- `discovery/scanners/ssdp.py`
- `discovery/service.py`

### escalation/ + events/ (2 files)
- `escalation/evaluator.py`
- `events/broadcaster.py`

### jobs/ + main.py (4 files)
- `jobs/email_graph_sync.py`
- `jobs/model_swap.py`
- `jobs/nightly_memory_sync.py`
- `main.py`

### mcp/ (1 file)
- `mcp/auth.py`

### memory/ (4 files)
- `memory/feedback.py`
- `memory/query_classifier.py`
- `memory/rag_client.py`
- `memory/service.py`
- `memory/token_estimator.py`

### orchestration/ + pipelines/ + modes/ (5 files)
- `orchestration/temporal.py`
- `pipelines/llm.py`
- `pipelines/notify.py`
- `modes/config.py`
- `modes/manager.py`

### presence/ (2 files)
- `presence/config.py`
- `presence/proxy.py`

### reasoning/ (8 files)
- `reasoning/config.py`
- `reasoning/consumer.py`
- `reasoning/context_aggregator.py`
- `reasoning/events.py`
- `reasoning/lock_integration.py`
- `reasoning/patterns.py`
- `reasoning/producers.py`
- `reasoning/prompts.py`
- `reasoning/reflection.py`
- `reasoning/state.py`

### security/ (11 files)
- `security/monitor.py`
- `security/assets/asset_tracker.py`
- `security/assets/drone_tracker.py`
- `security/assets/sensor_network.py`
- `security/assets/vehicle_tracker.py`
- `security/network/arp_monitor.py`
- `security/network/port_scan_detector.py`
- `security/network/traffic_analyzer.py`
- `security/wireless/deauth_detector.py`
- `security/wireless/monitor.py`
- `security/wireless/rogue_ap_detector.py`

### services/ (~30 files)
- `services/base.py`
- `services/calendar_provider.py`
- `services/crm_provider.py`
- `services/customer_context.py`
- `services/embedding/sentence_transformer.py`
- `services/google_oauth.py`
- `services/intelligence_report.py`
- `services/intervention_pipeline.py`
- `services/llm/cloud.py`
- `services/llm/groq.py`
- `services/llm/hybrid.py`
- `services/llm/llama_cpp.py`
- `services/llm/model_manager.py`
- `services/llm/openrouter.py`
- `services/llm/together.py`
- `services/llm/transformers_flash.py`
- `services/llm/vllm.py`
- `services/llm_router.py`
- `services/personaplex/audio.py`
- `services/personaplex/config.py`
- `services/personaplex/service.py`
- `services/protocols.py`
- `services/registry.py`
- `services/safety_gate.py`
- `services/scraping/captcha.py`
- `services/scraping/client.py`
- `services/scraping/parsers/capterra.py`
- `services/scraping/parsers/g2.py`
- `services/scraping/parsers/github.py`
- `services/scraping/parsers/hackernews.py`
- `services/scraping/parsers/reddit.py`
- `services/scraping/parsers/rss.py`
- `services/scraping/parsers/trustradius.py`
- `services/scraping/profiles.py`
- `services/scraping/proxy.py`
- `services/scraping/rate_limiter.py`
- `services/scraping/relevance.py`
- `services/tool_executor.py`

### storage/ (7 files)
- `storage/config.py`
- `storage/exceptions.py`
- `storage/repositories/conversation.py`
- `storage/repositories/call_transcript.py`
- `storage/repositories/reminder.py`
- `storage/repositories/sms_message.py`
- `storage/repositories/vector.py`

### Other (misc)
- `config.py`
- `debug.py`
- `schemas/query.py`
- `skills/registry.py`
- `templates/email/estimate_confirmation.py`
- `templates/email/invoice.py`
- `templates/email/proposal.py`
- `utils/cuda_lock.py`
- `utils/session_id.py`
- `utils/time.py`
- `vision/models.py`
- `vision/subscriber.py`
- `voice/audio_capture.py`
- `voice/command_executor.py`
- `voice/entity_context.py`
- `voice/frame_processor.py`
- `voice/playback.py`
- `voice/segmenter.py`
- `voice/tts_kokoro.py`
- `voice/vad/silero.py`

---

## Priority for Remaining Work

**Tier 1 (higher risk -- external-facing or error-handling heavy):**
- `autonomous/tasks/` (~40 files) -- background tasks with error handling, network calls
- `comms/` (10 files) -- Twilio call/SMS handling, external APIs
- `services/` remaining (~30 files) -- LLM providers, scraping, CRM, calendar
- `storage/repositories/` remaining (7 files) -- datetime.utcnow, unbounded queries

**Tier 2 (moderate risk):**
- `agents/` remaining (10 files) -- workflow state, booking, reminder graphs
- `reasoning/` remaining (8 files) -- event consumer, context aggregator, patterns
- `jobs/` + `main.py` -- startup, scheduled jobs
- `memory/` (4 files) -- RAG client, feedback, service

**Tier 3 (lower risk -- config, protocols, internal):**
- `security/` (11 files) -- network monitors (internal-only)
- `discovery/` (3 files) -- mDNS/SSDP scanners
- `auth/` (3 files) -- JWT, passwords
- `capabilities/` remaining (6 files) -- device implementations, protocols
- `voice/` remaining (7 files) -- audio capture, TTS, VAD
- `config.py`, `schemas/`, `templates/`, `utils/`, `vision/`, `skills/`, etc.

---

## B2B Churn Pipeline -- Monetization Gaps

### MUST-HAVE (first paying customer)

| # | Area | Gap | Files | Status |
|---|------|-----|-------|--------|
| M1 | Backend | Challenger intel reports -- `generate_challenger_report()` already implemented; API already routes both modes; dashboard whitelist already includes both types | `autonomous/tasks/b2b_churn_intelligence.py`, `api/vendor_targets.py`, `api/b2b_dashboard.py` | DONE (pre-existing) |
| M2 | Churn UI | Reports page type filter missing `vendor_retention` and `challenger_intel` options | `atlas-churn-ui/src/pages/Reports.tsx` | DONE |
| M3 | Churn UI | VendorTargets "Generate Report" button only shows for `vendor_retention` -- challengers have no report action | `atlas-churn-ui/src/pages/VendorTargets.tsx` | DONE |
| M4 | Churn UI | Affiliates page built but orphaned from sidebar nav (redirects to /leads) | `atlas-churn-ui/src/components/Sidebar.tsx`, `src/App.tsx` | DONE |

### SHOULD-HAVE (first month)

| # | Area | Gap | Files | Status |
|---|------|-----|-------|--------|
| S1 | Churn UI | No pagination -- all pages fetch fixed batch (50-200) | `atlas-churn-ui/src/components/DataTable.tsx` | DONE -- client-side pagination (25/50/100 per page) with page controls |
| S2 | Churn UI | ReportDetail renders intelligence_data objects as raw `JSON.stringify` | `atlas-churn-ui/src/pages/ReportDetail.tsx` | DONE -- structured renderers (ranked bars, stat grids, quote blocks, pills) |
| S3 | Backend + UI | No data export (CSV) for signals, reviews, leads | `atlas_brain/api/b2b_dashboard.py` | DONE -- `/export/signals`, `/export/reviews`, `/export/high-intent` CSV endpoints |
| S4 | Churn UI | No export buttons | `Vendors.tsx`, `Reviews.tsx`, `Leads.tsx` | DONE -- "Export CSV" buttons on all three pages, pass current filters |

### NICE-TO-HAVE

| # | Area | Gap | Files | Status |
|---|------|-----|-------|--------|
| N1 | Churn UI | No 404 catch-all route | `atlas-churn-ui/src/App.tsx` | TODO |
| N2 | Churn UI | Leads page `Affiliates` tab duplicates standalone Affiliates page logic | `src/pages/Leads.tsx` vs `src/pages/Affiliates.tsx` | TODO |
| N3 | Backend | No webhook notifications for new high-urgency signals | N/A -- new feature | TODO |

---

## Statistics

| Metric | Count |
|--------|-------|
| Rounds completed | 10 |
| Files hardened | ~93 |
| Files remaining | ~160 |
| Error detail leaks fixed (str(e)) | ~130+ |
| datetime.utcnow() replaced | ~50+ |
| Unbounded queries capped | ~30+ |
| Silent catches logged | ~20+ |
| Timeouts added | ~15+ |
| Unicode cleaned | ~40+ occurrences |
