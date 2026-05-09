# Content Ops Frontend Contract

Date: 2026-05-06

This document defines the frontend domain model for the **AI Content Ops**
product — strictly grounded in the real backend surface that exists today
in `extracted_content_pipeline/`. Every type and route below has a code
citation; nothing is extrapolated or imagined.

---

## Hallucinations from the prior scoping pass

A prior planning pass (ChatGPT-driven) proposed a `/content-ops/*` API
with a Catalog → Preview → Plan → Execute lifecycle and a long list of
typed contracts (`OutputDefinition`, `ContentOpsRequest`,
`ControlSurfacePreview`, `GenerationPlan`, `GenerationPlanStep`,
`SignalExtractionResult`, etc.). **None of those exist in the
codebase.** Verified via `grep -rn` on each name; only one claim from
that proposal is real (`CampaignReasoningContext`).

Building a frontend against that proposal would produce a UI that 404s
against the running server. This doc replaces it.

| Proposed (hallucinated) | Reality in code |
| --- | --- |
| `GET /content-ops/control-surfaces` | does not exist |
| `POST /content-ops/preview` | does not exist (no preview endpoint at all) |
| `POST /content-ops/plan` | does not exist (no plan abstraction) |
| `POST /content-ops/execute` | closest analogue: `POST /campaigns/operations/drafts/generate` |
| `OutputDefinition` per-output catalog | not modeled — config is per-route, not per-output |
| `ContentOpsRequest` (target_mode, preset, outputs, max_cost_usd, …) | each route has its own payload shape; no unified request envelope |
| `GenerationPlan` / `GenerationPlanStep` | does not exist; pipeline doesn't externalize plan steps |
| `SignalExtractionResult` / `SignalExtractionService` | does not exist in this package |
| `CampaignReasoningContext` (11-field dataclass) | **REAL** — `extracted_content_pipeline/campaign_ports.py:53` |

---

## What the backend actually exposes

### Ports (host-adapter Protocols)

`extracted_content_pipeline/campaign_ports.py` defines the full host-port
surface — these are the seams the host implements, the package consumes.

**Data shapes (frozen dataclasses):**

| Type | Line | Fields |
| --- | --- | --- |
| `TenantScope` | 18 | `account_id`, `user_id`, `allowed_vendors`, `roles` |
| `LLMMessage` | 28 | `role`, `content` |
| `LLMResponse` | 34 | `content`, `model`, `usage`, `raw` |
| `CampaignDraft` | 42 | `target_id`, `target_mode`, `channel`, `subject`, `body`, `metadata` |
| `CampaignReasoningContext` | 52 | `anchor_examples`, `witness_highlights`, `reference_ids`, `top_theses`, `account_signals`, `timing_windows`, `proof_points`, `coverage_limits`, `canonical_reasoning`, `scope_summary`, `delta_summary` |
| `SendRequest` | 102 | `campaign_id`, `to_email`, `subject`, `html_body`, `text_body`, `from_email`, `reply_to`, `headers`, `tags`, `metadata` |
| `SendResult` | 116 | `provider`, `message_id`, `raw` |
| `WebhookEvent` | 123 | `provider`, `event_type`, `message_id`, `email`, `occurred_at`, `payload` |

**Protocols (host plugs these in):**

| Protocol | Line | Methods |
| --- | --- | --- |
| `LLMClient` | 133 | `complete(messages, *, max_tokens, temperature, metadata)` |
| `SkillStore` | 145 | `get_prompt(name)` |
| `IntelligenceRepository` | 150 | `read_campaign_opportunities`, `read_vendor_targets` |
| `CampaignReasoningContextProvider` | 171 | `read_campaign_reasoning_context(scope, target_id, target_mode, opportunity)` |
| `CampaignRepository` | 183 | `save_drafts`, `list_due_sends`, `mark_sent`, `mark_cancelled`, `mark_send_failed`, `record_webhook_event`, `refresh_analytics` |
| `CampaignSequenceRepository` | 234 | `list_due_sequences`, `list_previous_campaigns`, `queue_sequence_step`, `mark_sequence_step` |
| `SuppressionRepository` | 271 | `is_suppressed`, `add_suppression` |
| `CampaignSender` | 295 | `send(SendRequest) -> SendResult` |
| `WebhookVerifier` | 300 | `verify_and_parse(body, headers) -> WebhookEvent` |
| `AuditSink` | 310 | `record(event_type, *, campaign_id, sequence_id, metadata)` |
| `VisibilitySink` | 322 | `emit(event_type, payload)` |
| `Clock` | 327 | `now()` |

The frontend never calls Protocols directly; they describe what the
host wires up so the routes below can run. The frontend talks to the
routes.

### Result types

| Type | File | Fields | Used by |
| --- | --- | --- | --- |
| `CampaignGenerationConfig` | `campaign_generation.py:32` | `skill_name`, `channel`, `limit`, `max_tokens`, `temperature`, `include_source_opportunity`, `channels` | server-side runtime config |
| `CampaignGenerationResult` | `campaign_generation.py:43` | `requested`, `generated`, `skipped`, `saved_ids`, `errors` | response of `/drafts/generate` |
| `CampaignSendConfig` | `campaign_send.py:22` | `default_from_email`, `default_reply_to`, `unsubscribe_base_url`, `unsubscribe_token_secret`, `company_address`, `limit` | server-side runtime config |
| `CampaignSendSummary` | `campaign_send.py:34` | `sent`, `failed`, `suppressed`, `skipped` | response of `/send/queued` |

### API config classes (host-tunable defaults per route group)

| Class | File:line | Prefix | What it controls |
| --- | --- | --- | --- |
| `CampaignOperationsApiConfig` | `api/campaign_operations.py:66` | `/campaigns/operations` | generation/send/sequence/analytics defaults + reasoning mode (single-pass / multi-pass / explicit-provider) |
| `B2BCampaignApiConfig` | `api/b2b_campaigns.py:40` | `/b2b/campaigns` | draft listing/export/review |
| `SellerCampaignApiConfig` | `api/seller_campaigns.py:48` | `/seller/...` (configurable) | seller-side targets + opportunities + drafts |
| `CampaignWebhookApiConfig` | `api/campaign_webhooks.py:47` | `/webhooks` (configurable) | unsubscribe + provider engagement events |

### Route surface (the actual frontend integration target)

#### `CampaignOperationsApiConfig` — `/campaigns/operations/*`

| Route | Method | Body / query | Returns |
| --- | --- | --- | --- |
| `/status` | GET | — | `{status, database, providers, reasoning, features, limits}` (see "control surface" mapping below) |
| `/drafts/generate` | POST | `{limit?, target_mode?, channel?, channels?, filters?, account_id?}` | `CampaignGenerationResult.as_dict()` |
| `/send/queued` | POST | `{limit?}` | `CampaignSendSummary.as_dict()` |
| `/sequences/progress` | POST | `{limit?, max_steps?}` | sequence-progression result |
| `/analytics/refresh` | POST | — | analytics-refresh result (errors sanitized) |

#### `B2BCampaignApiConfig` — `/b2b/campaigns/*`

| Route | Method | Body / query | Returns |
| --- | --- | --- | --- |
| `/drafts` | GET | `?statuses, target_mode, channel, vendor_name, company_name, limit` | draft list `as_dict()` |
| `/drafts/export` | GET | same + `format=csv\|json` | CSV download or JSON |
| `/drafts/review` | POST | `{campaign_ids, status, from_statuses?, from_email?, reason?, reviewed_by?, metadata?, dry_run?}` | review result |

#### `SellerCampaignApiConfig` — `/seller/...`

| Route | Method |
| --- | --- |
| `/targets` | GET, POST |
| `/targets/{target_id}` | GET, PATCH, DELETE |
| `/intelligence/refresh` | POST |
| `/opportunities/prepare` | POST |
| `/operations/refresh-and-prepare` | POST |
| `/campaigns/drafts` | GET |
| `/campaigns/drafts/export` | GET |
| `/campaigns/drafts/review` | POST |

#### `CampaignWebhookApiConfig` — `/webhooks/*`

| Route | Method |
| --- | --- |
| `/unsubscribe` | GET, POST |
| `/campaign-email` | POST |

### `/status` is the closest thing to a "control surface catalog"

The proposal's `Catalog` concept maps to the real `GET /campaigns/operations/status`
response shape (`api/campaign_operations.py:435-515`):

```json
{
  "status": "ready" | "degraded",
  "database": { "configured": true, "available": true|false, "reason"?: "..." },
  "providers": {
    "database": bool, "sender": bool, "llm": bool,
    "skills": bool, "reasoning": bool, "visibility": bool
  },
  "reasoning": {
    "mode": "explicit_provider" | "multi_pass" | "single_pass" | "none",
    "single_pass_configured": bool, "single_pass_ready": bool,
    "multi_pass_configured": bool, "multi_pass_ready": bool
  },
  "features": {
    "draft_generation": bool,
    "send_queued": bool,
    "sequence_progression": bool,
    "analytics_refresh": bool
  },
  "limits": {
    "generation": {"default_limit", "max_limit", "target_mode", "channel", "channels"},
    "send":       {"default_limit", "max_limit"},
    "sequence":   {"default_limit", "max_limit", "default_max_steps", "max_steps"}
  }
}
```

Important differences from the hallucinated `OutputDefinition` catalog:

- It is **per-feature**, not per-output. Four features exist:
  `draft_generation`, `send_queued`, `sequence_progression`,
  `analytics_refresh`. There is no per-output cost estimate, no
  "implemented vs configured vs executable" three-state, no
  required-input declaration per output.
- Cost is not exposed at all in this surface today. Token cost lives
  in `extracted_llm_infrastructure/services/cost/` but is not
  surfaced through the operations API.
- Reasoning state is exposed at the *service* level
  (`reasoning.mode`, `reasoning.single_pass_ready`,
  `reasoning.multi_pass_ready`), not per-output.

---

## Frontend domain model — grounded in the real surface

### Top-level abstraction

A **CampaignOperationsRun** (not "ContentOpsRun" — naming follows the
real config/route prefix `campaigns/operations`):

```ts
interface CampaignOperationsRun {
  // Snapshot of /campaigns/operations/status at run start
  controlSurface: ControlSurface;

  // The single feature this run invoked
  feature: "draft_generation" | "send_queued"
         | "sequence_progression" | "analytics_refresh";

  // Per-feature request (only one of these is populated)
  draftGenerationRequest?: DraftGenerationRequest;
  sendQueuedRequest?: SendQueuedRequest;
  sequenceProgressionRequest?: SequenceProgressionRequest;
  analyticsRefreshRequest?: AnalyticsRefreshRequest;

  // Per-feature result (mirrors backend `as_dict()`)
  draftGenerationResult?: DraftGenerationResult;
  sendQueuedResult?: SendQueuedResult;
  sequenceProgressionResult?: SequenceProgressionResult;
  analyticsRefreshResult?: AnalyticsRefreshResult;

  // Optional draft-review / draft-listing artifacts
  drafts?: CampaignDraftListing;
  reviewResult?: DraftReviewResult;
}
```

Each run is **single-feature**. The proposal's "outputs: [...]
multi-select" doesn't exist — the backend has separate routes per
feature, not a unified execute endpoint.

### Catalog / control surface

Mirrors `/campaigns/operations/status` exactly:

```ts
interface ControlSurface {
  status: "ready" | "degraded";
  database: { configured: boolean; available: boolean; reason?: string };
  providers: {
    database: boolean;
    sender: boolean;
    llm: boolean;
    skills: boolean;
    reasoning: boolean;
    visibility: boolean;
  };
  reasoning: {
    mode: "explicit_provider" | "multi_pass" | "single_pass" | "none";
    singlePassConfigured: boolean;
    singlePassReady: boolean;
    multiPassConfigured: boolean;
    multiPassReady: boolean;
  };
  features: {
    draftGeneration: boolean;
    sendQueued: boolean;
    sequenceProgression: boolean;
    analyticsRefresh: boolean;
  };
  limits: {
    generation: { defaultLimit: number; maxLimit: number; targetMode: string; channel: string; channels: string[] };
    send:       { defaultLimit: number; maxLimit: number };
    sequence:   { defaultLimit: number; maxLimit: number; defaultMaxSteps: number; maxSteps: number };
  };
}
```

Frontend behavior: `features.<x>=false` disables the corresponding run
button. `providers.<x>=false` shows a "host not configured" state.
`reasoning.mode` drives the reasoning-state badge.

### Draft generation

```ts
interface DraftGenerationRequest {
  limit?: number;          // capped by limits.generation.maxLimit
  targetMode?: string;     // defaults to limits.generation.targetMode
  channel?: string;        // defaults to limits.generation.channel
  channels?: string[];     // overrides single-channel mode
  filters?: Record<string, unknown>;
  accountId?: string;
}

interface DraftGenerationResult {
  requested: number;
  generated: number;
  skipped: number;
  savedIds: string[];
  errors: Array<Record<string, unknown>>;
}
```

Field correspondence to backend (`campaign_operations.py:632-755`):

- `limit` → `_payload_limit`-bounded by `default_generation_limit` /
  `max_generation_limit`.
- `target_mode` defaults to `generation_target_mode`.
- `channel` defaults to `generation_channel`.
- Result is `CampaignGenerationResult.as_dict()`
  (`campaign_generation.py:51-58`).

### Send queued

```ts
interface SendQueuedRequest {
  limit?: number;          // capped by limits.send.maxLimit
}

interface SendQueuedResult {
  sent: number;
  failed: number;
  suppressed: number;
  skipped: number;
}
```

Result is `CampaignSendSummary.as_dict()` (`campaign_send.py:34-47`).

### Draft listing (per `/b2b/campaigns/drafts`)

```ts
interface CampaignDraftListingRequest {
  statuses?: string[];      // default ["draft"] per B2BCampaignApiConfig
  targetMode?: string;
  channel?: string;
  vendorName?: string;
  companyName?: string;
  limit?: number;           // capped by maxLimit
}

interface CampaignDraftRow {
  id: string;
  vendorName: string;
  companyName: string;
  channel: string;
  subject: string;
  body: string;
  status: "draft" | "approved" | "sent" | "expired";
  createdAt: string;
  // plus opportunity_score, urgency_score, pain_categories, etc.
  // (see migration 066_b2b_campaigns.sql for the full set)
}
```

Frontend rule: every column rendered must trace to an actual
`b2b_campaigns` schema column. The schema is canonical at
`extracted_content_pipeline/storage/migrations/066_b2b_campaigns.sql`.

### Draft review

```ts
interface DraftReviewRequest {
  campaignIds: string[];
  status: "approved" | "draft" | "expired" | string;
  fromStatuses?: string[];   // default ["draft"]
  fromEmail?: string;
  reason?: string;
  reviewedBy?: string;
  metadata?: Record<string, unknown>;
  dryRun?: boolean;
}
```

Maps to `POST /b2b/campaigns/drafts/review`
(`api/b2b_campaigns.py:221-248`).

### Reasoning context (the only fully verified shape)

```ts
interface CampaignReasoningContextView {
  anchorExamples: Record<string, Array<Record<string, unknown>>>;
  witnessHighlights: Array<Record<string, unknown>>;
  referenceIds: Record<string, string[]>;
  topTheses: Array<Record<string, unknown>>;
  accountSignals: Array<Record<string, unknown>>;
  timingWindows: Array<Record<string, unknown>>;
  proofPoints: Array<Record<string, unknown>>;
  coverageLimits: string[];
  scopeSummary: Record<string, unknown>;
  deltaSummary: Record<string, unknown>;
  // canonical_reasoning is merged into the as_dict output;
  // surface it as a sibling raw blob in the UI.
  raw: Record<string, unknown>;
}
```

Note the prior proposal had `ReasoningCapability` with a
`requirement: "absent" | "optional_host_context"` enum. **That enum
does not exist.** What the backend actually models is a **mode** at
the operations-API level:

```ts
type ReasoningMode = "explicit_provider" | "multi_pass" | "single_pass" | "none";
```

with `singlePassReady` / `multiPassReady` booleans alongside.

The frontend should surface "reasoning mode + readiness" rather than
"reasoning requirement per output."

---

## Architectural layering for the frontend

### API adapter layer

One adapter module per backend route group, snake_case → camelCase
field translation only — no business logic:

```
src/api/
  campaignOperations.ts   // /campaigns/operations/*
  b2bCampaigns.ts         // /b2b/campaigns/*
  sellerCampaigns.ts      // /seller/...
  campaignWebhooks.ts     // /webhooks/*  (mostly out of UI scope)
```

### Domain layer

Owns the typed models above. No HTTP, no React.

```
src/domain/
  controlSurface.ts
  campaignOperationsRun.ts
  campaignDraft.ts
  campaignReasoningContext.ts
```

### View-model layer

Builds UI-ready state from domain + adapter:

- Feature-availability chips (driven by `controlSurface.features`)
- Reasoning-mode badge (driven by `controlSurface.reasoning`)
- Draft-generation form (form fields capped by
  `controlSurface.limits.generation`)
- Draft list table (driven by `CampaignDraftRow[]`)
- Result panels (one per feature kind)

### UI layer

Dumb components; no fetch, no business rules.

---

## What the proposal got right (architecturally) — and what it would take to get there

The proposal's **shape** (Catalog → Request → Preview → Plan → Execute
→ Step Results → Artifacts → Review) is a reasonable v2 architecture.
It just describes a backend that **doesn't exist yet**. If the goal is
that lifecycle, the backend work needed first:

1. **Multi-output catalog endpoint.** Today there's no per-output
   declaration. A new `/campaigns/catalog` (or expansion of
   `/status`) would need to enumerate outputs with id, label, cost
   estimate, required inputs, implementation status. ~1 new module.
2. **Preview endpoint.** Today validation happens server-side on the
   request payload. A separate `/preview` would re-use the
   validators but skip the side-effecting body. ~1 new module.
3. **Plan abstraction.** The pipeline currently runs synchronously
   inside `/drafts/generate`. Externalizing a plan (steps + per-step
   config) means refactoring `generate_campaign_drafts_from_postgres`
   to expose its plan-vs-execute boundary. Real refactor, not a
   wrapper. **Multi-PR effort.**
4. **Step-level result model.** Today the result is a single
   `CampaignGenerationResult`. Step-level results require the plan
   abstraction first.
5. **Signal extraction.** Not in this package today. Would either
   need to be ported over from `atlas_brain/services/scraping/` or
   designed fresh.

If you want the v2 architecture, treat the proposal as a backend
design doc and queue those five items — not a frontend contract that
can be built today.

---

## MVP screens grounded in the real backend

No fancy design — strictly what the four real route groups support:

1. **Operations Status / Control Surface**
   - Reads `/campaigns/operations/status`
   - Shows: feature-readiness chips, reasoning-mode badge, provider
     wiring, configured limits.

2. **Generate Drafts**
   - Form: limit, target_mode, channel(s), filters, account_id (form
     bounded by `limits.generation`)
   - Submit → `POST /campaigns/operations/drafts/generate`
   - Result panel: `requested / generated / skipped / saved_ids /
     errors`

3. **Drafts Listing & Review**
   - Reads `GET /b2b/campaigns/drafts`
   - Filters: statuses, target_mode, channel, vendor_name,
     company_name, limit
   - Bulk action: select rows → `POST /b2b/campaigns/drafts/review`
     with status + dry_run option
   - Export button: `/drafts/export?format=csv` → file download

4. **Send / Sequence / Analytics**
   - Three buttons, each posts to its respective `/campaigns/operations/*`
     route with optional `limit` / `max_steps`.
   - Result panels show the corresponding result dataclass.

5. **Reasoning Context Drawer**
   - Per-draft, opens a side panel showing the
     `CampaignReasoningContext` if the host wired
     `CampaignReasoningContextProvider`.
   - For drafts with no provider, shows "no reasoning context
     available."

Not in MVP (because backend doesn't support):

- A "preview before run" flow (no `/preview` endpoint exists)
- A "plan steps" view (no plan abstraction exists)
- A multi-output run wizard (no multi-output catalog exists)
- A per-output cost estimator (cost is not surfaced through this API)

---

## Out of scope for this contract

- Final dashboard metrics
- Full asset editor UX
- Visual workflow builder
- Model-selection UX
- Collaboration / role-permission flows
- CMS export

These can land later when there's product clarity. They do not block
v0 of the frontend, which can ship against just the four MVP screens
above.

---

## Code references summary

Every claim in this doc cites a file and line:

| Claim | Citation |
| --- | --- |
| `CampaignReasoningContext` 11 fields | `extracted_content_pipeline/campaign_ports.py:53-66` |
| Host-port Protocols | `extracted_content_pipeline/campaign_ports.py:133-329` |
| `CampaignGenerationResult` shape | `extracted_content_pipeline/campaign_generation.py:43-58` |
| `CampaignSendSummary` shape | `extracted_content_pipeline/campaign_send.py:34-47` |
| `/campaigns/operations/status` response | `extracted_content_pipeline/api/campaign_operations.py:435-515, 613-630` |
| `/drafts/generate` request shape | `extracted_content_pipeline/api/campaign_operations.py:632-755` |
| `/b2b/campaigns/*` routes | `extracted_content_pipeline/api/b2b_campaigns.py:146-248` |
| `/seller/...` routes | `extracted_content_pipeline/api/seller_campaigns.py:404-635` |
| Multi-pass reasoning config fields | `extracted_content_pipeline/api/campaign_operations.py:107-111` |
| `b2b_campaigns` schema columns | `extracted_content_pipeline/storage/migrations/066_b2b_campaigns.sql` |

If a future revision adds claims, the same rule applies: every type
or field must cite a file:line. No silent extrapolation.
