# Session Coordination

## Session Roles
- **Senior (this session)**: Architecture decisions, CRM wiring, agency workflow, cross-reference intelligence
- **Junior (other session)**: MCP server refinement, provider hardening, tool migration tasks assigned by senior

## Branch Discipline
- **Both sessions work on `main`** with pull-before-edit discipline
- Before editing ANY file: `git pull origin main`
- Commit frequently (small atomic commits) to minimize conflict windows
- If a conflict occurs: the session that caused it resolves it

## File Ownership (Conflict Avoidance)

### Senior Session OWNS (do not edit without asking):
- `atlas_brain/agents/` (all graphs, workflows, routing)
- `atlas_brain/autonomous/` (task handlers, runner, scheduler)
- `atlas_brain/comms/call_intelligence.py`
- `atlas_brain/voice/` (pipeline, launcher)
- `atlas_brain/memory/` (RAG, feedback, quality)
- `atlas_brain/storage/migrations/` (new migrations)

### Junior Session OWNS (do not edit without asking):
- `atlas_brain/mcp/` (all 4 MCP servers)
- `atlas_brain/services/calendar_provider.py`
- `atlas_brain/services/email_provider.py`
- `atlas_brain/services/crm_provider.py`
- `atlas_brain/templates/` (email templates)
- `atlas_comms/` (telephony providers)
- `tests/test_mcp_servers.py`

### SHARED FILES (coordinate before editing):
- `atlas_brain/config.py` — Senior adds new config sections, junior adds provider config
- `atlas_brain/main.py` — Senior manages lifespan, junior adds MCP startup if needed
- `atlas_brain/services/__init__.py` — Coordinate exports
- `atlas_brain/services/protocols.py` — Senior defines new protocols, junior implements
- `atlas_brain/tools/scheduling.py` — Senior owns CRM wiring, junior owns tool interface
- `requirements.txt` — Either can add deps, pull first
- `CLAUDE.md` — Senior updates architecture sections
- `docker-compose.yml` — Coordinate

### Protocol for shared files:
1. Announce intent: "I need to add X to config.py"
2. Pull latest
3. Make minimal, isolated edit
4. Commit + push immediately
5. Tell the other session: "pushed config.py change, pull before editing"

## Current Task Assignments — Round 1

### Completed — Round 1:
- **S1** Migration 036: `contact_id` FK on `call_transcripts` ✓ (fb17380)
- **S2** Wire `call_intelligence.py` → CRM auto-population ✓ (fb17380)
- **S3** `CustomerContextService` cross-reference layer ✓ (fb17380)
- **J1** CRM duplicate protection (`find_or_create_contact` + partial unique indexes) ✓ (0a29b39)
- **J2** MCP server smoke test + fixes ✓ (0a29b39)
- **J3** `BookAppointmentTool` → CRM contact linkage ✓ (06a5d3a)
- **J4** Email provider IMAP graceful fallback ✓ (06a5d3a)

---

## Round 2 — Agency Workflow

### Senior Session - Active Tasks:

**S4. Action Planner — LLM + CustomerContext → structured action plan**
- After CRM linkage in call_intelligence pipeline, build full CustomerContext
- Feed context + call data to LLM with a skill prompt
- LLM outputs structured JSON action plan: `[{action, params, rationale}]`
- Store plan on transcript record (enrich `proposed_actions`)
- New skill: `skills/call/action_planning.md`

**S5. Plan approval + execution endpoint**
- Enhanced ntfy notification showing full plan summary
- `POST /call-actions/{id}/approve-plan` → execute all planned actions
- Reuse existing action logic (calendar, email, SMS)
- Log each executed action to `contact_interactions`
- `POST /call-actions/{id}/reject-plan` → mark plan rejected

### Junior Session - Active Tasks:

**J5. Wire email inbox into CustomerContext**
- File: `atlas_brain/services/customer_context.py` (SHARED — coordinate with senior)
- Add `_get_inbox_emails()` method: search IMAP/Gmail for recent emails from/to the customer's email
- Uses `email_provider.list_messages(query="from:{email}")` for inbound
- Merge into `CustomerContext` as `inbox_emails: list[dict]`
- Fail-open: if IMAP unavailable, return empty list

**J6. MCP tool for CustomerContext**
- New tool in `atlas_brain/mcp/crm_server.py`: `get_customer_context`
- Wraps `CustomerContextService.get_context()` / `get_context_by_phone()`
- Exposes full customer view to MCP clients (NocoDB, external agents)

### Task Dependencies:
```
S4 (action planner) ──→ S5 (approval + execution)
J5 (inbox context)  — independent, enhances S4 output
J6 (MCP tool)       — independent, can start anytime
```

---

## Intelligence Platform Roadmap — Sprint Completion Log

| Sprint | Phase | Status | Key Deliverables |
|--------|-------|--------|-----------------|
| Phase 2 Sprint 1 | Phase 2 | DONE | displacement edges + company signals (migration 099) |
| Phase 2 Sprint 2 | Phase 2 | DONE | pain points + use cases + integrations (migration 100) |
| Phase 2 Sprint 3 | Phase 2 | DONE | buyer profiles (migration 101) |
| Phase 2 Sprint 4 | Phase 2 | DONE | confidence scoring close-out (migration 102) |
| Phase 3 Sprint 4 | Phase 3 | DONE | cross-vendor trend correlation (migration 109, concurrent events, Pearson r) |
| Phase 4 Sprint 1 | Phase 4 | DONE | campaign outcome tracking + signal effectiveness (migration 104) |
| Phase 4 Sprint 2 | Phase 4 | DONE | score calibration from outcomes (migration 106) |
| Phase 4 Sprint 3 | Phase 4 | DONE | CRM event ingestion pipeline (migration 108) |
| Phase 4 Sprint 4 | Phase 4 | DONE | Salesforce + Pipedrive native webhooks |
| Phase 5 Sprint 1 | Phase 5 | DONE | webhook outbound delivery (migration 107) |
| Phase 5 Sprint 2 | Phase 5 | DONE | PDF intelligence report export (fpdf2) |
| Phase 6 Sprint 1 | Phase 6 | DONE | data corrections infrastructure (migration 105) |
| Phase 6 Sprint 2 | Phase 6 | DONE | correction application logic (NOT EXISTS subqueries) |
| Phase 6 Sprint 3 | Phase 6 | DONE | vendor merge execution (17 tables) |
| Phase 6 Sprint 4 | Phase 6 | DONE | field override reads (single-entity endpoints) |
| Phase 6 Sprint 5 | Phase 6 | DONE | source quality controls (migration 110, suppress_source) |
| Phase 0 Sprint 1 | Phase 0 | DONE | dedup key uses canonical vendor_name |
| Phase 1 Sprint 1 | Phase 1 | DONE | CAPTCHA telemetry + block type classification (migration 111) |
| Phase 5 Sprint 3 | Phase 5 | DONE | Slack + Teams notification channels (migration 112, Block Kit, Adaptive Cards) |
| Phase 5 Sprint 4 | Phase 5 | DONE | CRM outbound push (migration 113, HubSpot/Salesforce/Pipedrive channels, push log) |
| Phase 0/2 Sprint | Phase 0+2 | DONE | Fuzzy vendor + company matching (migration 114, pg_trgm, difflib fallback, REST + MCP) |
| Phase 1 Sprint 2 | Phase 1 | DONE | Auto re-processing on parser version change (_queue_version_upgrades, REST + MCP) |
| Phase 3 Sprint 5 | Phase 3 | DONE | Product profile snapshots (migration 115, daily captures, REST + MCP history queries) |
| Phase 4 Sprint 5 | Phase 4 | DONE | CRM event enrichment (cross-event resolution, vendor normalization, fuzzy matching, REST + MCP stats) |
| Surface Sprint 1 | Phase 0 | DONE | Source-health telemetry (CAPTCHA, block types, proxy), source capabilities REST, MCP: get_source_telemetry |
| Surface Sprint 2 | Phase 5 | DONE | Operational overview (pipeline + health + telemetry + events), telemetry timeline, MCP: get_operational_overview |
| Surface Sprint 3 | Phase 5 | DONE | REST gaps closed: product-profile, displacement-history endpoints |
| Surface Sprint 4 | Phase 4 | DONE | Action feedback loop surfaces: outcome filter on sequences, outcome distribution, signal-to-outcome attribution, calibration trigger, date-range CRM events, batch created_ids, MCP tools |
| Surface Sprint 5 | Phase 5 | DONE | Thin delivery surfaces: PATCH webhook update, 7d delivery stats on REST list, delivery filtering (date/success/event_type), delivery summary, MCP update_webhook + get_webhook_delivery_summary |
| Surface Sprint 6 | Phase 6 | DONE | Analyst control surfaces: corrected_by + date range filters on corrections, correction stats dashboard, MCP get_data_correction + get_correction_stats |
