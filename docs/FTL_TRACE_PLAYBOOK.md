# Fine Tune Lab Trace Playbook

This document defines the Atlas trace views and filters that should be created
in Fine Tune Lab for the current tracing schema.

These views are based on verified Atlas span names and metadata fields in:

- `atlas_brain/services/tracing.py`
- `atlas_brain/agents/interface.py`
- `atlas_brain/reasoning/agent.py`
- `atlas_brain/autonomous/tasks/b2b_churn_intelligence.py`
- `atlas_brain/autonomous/tasks/b2b_tenant_report.py`
- `atlas_brain/api/b2b_crm_events.py`
- `atlas_brain/api/b2b_dashboard.py`
- `atlas_brain/services/b2b/webhook_dispatcher.py`
- `atlas_brain/services/b2b/pdf_renderer.py`

## Required env vars

Atlas reads these from `.env` and `.env.local`:

- `ATLAS_FTL_TRACING__ENABLED`
- `ATLAS_FTL_TRACING__BASE_URL`
- `ATLAS_FTL_TRACING__API_KEY`
- `ATLAS_FTL_TRACING__USER_ID`
- `ATLAS_FTL_TRACING__CAPTURE_BUSINESS_CONTEXT`
- `ATLAS_FTL_TRACING__CAPTURE_REASONING_SUMMARIES`
- `ATLAS_FTL_TRACING__CAPTURE_RAW_REASONING`
- `ATLAS_FTL_TRACING__MAX_REASONING_CHARS`

Recommended default posture:

- business context: on
- reasoning summaries: on
- raw reasoning: off

## Verified span names

- `agent.process`
- `agent.classify`
- `agent.memory`
- `agent.think`
- `agent.act`
- `agent.respond`
- `reasoning.process`
- `b2b.churn_intelligence.run`
- `b2b.tenant_report`
- `b2b.crm_event.ingest`
- `b2b.correction.create`
- `b2b.webhook.dispatch`
- `b2b.webhook.test`
- `b2b.report.export_pdf`

## Verified metadata paths

Business context:

- `metadata.business.account_id`
- `metadata.business.product`
- `metadata.business.workflow`
- `metadata.business.report_type`
- `metadata.business.event_type`
- `metadata.business.crm_provider`
- `metadata.business.vendor_name`
- `metadata.business.company_name`
- `metadata.business.signal_type`
- `metadata.business.entity_type`
- `metadata.business.entity_id`
- `metadata.business.correction_type`
- `metadata.business.source_name`
- `metadata.business.subscription_id`

Reasoning context:

- `metadata.reasoning.decision`
- `metadata.reasoning.evidence`
- `metadata.reasoning.triage`
- `metadata.reasoning.summary`
- `metadata.reasoning.raw_preview`

Core span fields:

- `span_name`
- `operation_type`
- `status`
- `duration_ms`
- `model_name`
- `model_provider`
- `input_tokens`
- `output_tokens`
- `total_tokens`
- `ttft_ms`
- `inference_time_ms`
- `queue_time_ms`
- `retrieval_latency_ms`
- `session_tag`

## Recommended views

### 1. Agent Decision Audit

Purpose: inspect user-facing agent decisions and pre-response reasoning summaries.

Filter:

- `span_name = agent.process`

Columns:

- `start_time`
- `status`
- `duration_ms`
- `model_name`
- `metadata.business.workflow`
- `metadata.reasoning.decision`
- `metadata.reasoning.summary`
- `metadata.reasoning.evidence`

### 2. Reasoning Deep Dive

Purpose: inspect cross-domain reasoning outcomes before actions are taken.

Filter:

- `span_name = reasoning.process`

Columns:

- `start_time`
- `status`
- `metadata.business.event_type`
- `metadata.business.entity_type`
- `metadata.business.entity_id`
- `metadata.reasoning.triage`
- `metadata.reasoning.summary`
- `metadata.reasoning.decision`
- `metadata.reasoning.evidence`

### 3. BI Pipeline Health

Purpose: monitor the main business-intelligence generation paths.

Filter:

- `span_name IN (b2b.churn_intelligence.run, b2b.tenant_report)`

Columns:

- `start_time`
- `span_name`
- `status`
- `duration_ms`
- `metadata.business.workflow`
- `metadata.business.account_id`
- `metadata.business.report_type`
- `metadata.reasoning.evidence`
- `metadata.reasoning.summary`

### 4. CRM Ingest Ops

Purpose: inspect CRM event ingestion quality by provider.

Filter:

- `span_name = b2b.crm_event.ingest`

Group by:

- `metadata.business.crm_provider`
- `status`

Columns:

- `start_time`
- `status`
- `duration_ms`
- `metadata.business.crm_provider`
- `metadata.business.event_type`
- `metadata.business.account_id`
- `metadata.reasoning.evidence`

### 5. Webhook Delivery Ops

Purpose: inspect outbound webhook and CRM push operations.

Filter:

- `span_name IN (b2b.webhook.dispatch, b2b.webhook.test)`

Columns:

- `start_time`
- `status`
- `duration_ms`
- `metadata.business.workflow`
- `metadata.business.event_type`
- `metadata.business.vendor_name`
- `metadata.business.subscription_id`
- `metadata.reasoning.evidence`

### 6. PDF Export Ops

Purpose: verify report export activity and data composition.

Filter:

- `span_name = b2b.report.export_pdf`

Columns:

- `start_time`
- `status`
- `duration_ms`
- `metadata.business.report_type`
- `metadata.business.vendor_name`
- `metadata.reasoning.evidence`
- `metadata.reasoning.summary`

### 7. Analyst Corrections

Purpose: audit business-intel operations and correction workflows.

Filter:

- `span_name = b2b.correction.create`

Columns:

- `start_time`
- `status`
- `metadata.business.account_id`
- `metadata.business.entity_type`
- `metadata.business.correction_type`
- `metadata.business.source_name`
- `metadata.reasoning.decision`
- `metadata.reasoning.evidence`

## Recommended alerts

### BI pipeline failures

Condition:

- `span_name IN (b2b.churn_intelligence.run, b2b.tenant_report)`
- `status = failed`

### CRM ingest failures

Condition:

- `span_name = b2b.crm_event.ingest`
- `status = failed`

### Webhook failures

Condition:

- `span_name IN (b2b.webhook.dispatch, b2b.webhook.test)`
- `status = failed`

### Reasoning failures

Condition:

- `span_name = reasoning.process`
- `status = failed`

## Implementation note

This playbook does not assume a Fine Tune Lab saved-dashboard API. If Fine Tune
Lab later exposes an import format, map the view pack in
`docs/ftl_trace_views.json` into that API rather than rewriting filters by hand.
