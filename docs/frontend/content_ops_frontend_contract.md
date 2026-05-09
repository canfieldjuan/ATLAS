# Content Ops Frontend Contract

Date: 2026-05-09
Code reference HEAD: `a4020c1` (post PR-Campaign-Config-V2 #398)

This document defines the frontend domain model for the **AI Content Ops**
product, grounded entirely in the real backend surface that exists today
in `extracted_content_pipeline/`. Every type and route below cites a
file:line.

---

## Retraction of the prior draft

A prior draft of this doc (commit `61f1003` on
`claude/content-ops-frontend-contract`) declared the proposed
`/content-ops/*` API "hallucinated" because a local `grep` returned
zero hits. **That verdict was wrong.** The grep ran against an
`origin/main` snapshot 11 PRs behind current head — PRs #389 through
#398 had already shipped the entire surface. Re-running the same grep
against fresh main flips every "fabricated" finding to "verified."

This is the failure mode AGENTS.md §4a names ("Don't trust claims;
reproduce them") in its inverse: trusting a stale local tree to
*reject* real claims. Worth naming so it doesn't recur.

---

## What the backend exposes

### Routes

`extracted_content_pipeline/api/control_surfaces.py:212-272`

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/content-ops/control-surfaces` | Output catalog + presets + ingestion-profile menu + per-output execution flags |
| `POST` | `/content-ops/preview` | Preflight validation; returns `ControlSurfacePreview` |
| `POST` | `/content-ops/plan` | Build runnable plan; returns `GenerationPlan` |
| `POST` | `/content-ops/execute` | Execute the plan via host-injected services; returns `ContentOpsExecutionResult` |

The router prefix is configurable via `ContentOpsControlSurfaceApiConfig`
(`api/control_surfaces.py:143`). All four routes share one pydantic
body model (`ContentOpsRequestModel`, `api/control_surfaces.py:157-185`)
with bounded sizes (max 50 input keys, depth 6, string ≤ 10000 chars,
`outputs` length ≤ 20, `limit` 1–1000).

### Catalog response shape (`/content-ops/control-surfaces`)

Composed at `api/control_surfaces.py:101-131`. The static portion is
cached at module import (PR-Describe-Control-Surfaces-Cache, #397);
only the per-output `execution_configured` / `can_execute` and
top-level `execution.{configured,configured_outputs}` /
`reasoning.configured` flags are recomputed per request.

```jsonc
{
  "outputs": [
    {
      "id": "email_campaign",
      "label": "Email Campaign",
      "description": "Cold email and follow-up campaign drafts.",
      "implemented": true,
      "estimated_unit_cost_usd": 0.18,
      "default_parse_retry_attempts": 1,
      "estimated_retry_adjusted_unit_cost_usd": 0.36,
      "required_inputs": ["target_account", "offer"],
      "default_max_items": 3,
      "reasoning_requirement": "optional_host_context",
      // dynamic per request:
      "execution_configured": true,
      "can_execute": true
    }
    // …5 more outputs
  ],
  "presets": [
    {"id": "email_only", "label": "Email Only", "description": "Lowest-cost outreach draft run.", "outputs": ["email_campaign"]}
    // …4 more presets
  ],
  "execution": {
    "configured": true,
    "configured_outputs": ["blog_post", "email_campaign", "report"]
  },
  "reasoning": {
    "configured": true
  },
  "ingestion_profiles": ["domain_specific", "manual", "existing_evidence"]
}
```

### Real types — frozen dataclasses

| Type | File:line | Fields |
| --- | --- | --- |
| `OutputDefinition` | `control_surfaces.py:16-30` | `id`, `label`, `description`, `implemented`, `estimated_unit_cost_usd`, `required_inputs`, `default_max_items`, `reasoning_requirement`, `default_parse_retry_attempts` |
| `ControlSurfacePreset` | `control_surfaces.py:33-40` | `id`, `label`, `outputs`, `description` |
| `ContentOpsRequest` | `control_surfaces.py:43-55` | `target_mode`, `preset`, `outputs`, `limit`, `max_cost_usd`, `inputs`, `ingestion_profile`, `require_quality_gates`, `allow_unimplemented_outputs` |
| `ControlSurfacePreview` | `control_surfaces.py:58-90` | `can_run`, `outputs`, `estimated_cost_usd`, `missing_inputs`, `blocked_outputs`, `warnings`, `normalized_request` |
| `GenerationPlanStep` | `generation_plan.py:28-45` | `output`, `runner`, `status`, `config`, `reason` |
| `GenerationPlan` | `generation_plan.py:48-65` | `can_execute`, `target_mode`, `limit`, `steps`, `preview` |
| `ContentOpsExecutionServices` | `content_ops_execution.py:16-63` | `campaign`, `blog_post`, `report`, `landing_page`, `sales_brief`, `signal_extraction` (host-port bundle) |
| `ContentOpsStepExecution` | `content_ops_execution.py:66-83` | `output`, `runner`, `status`, `result`, `error` |
| `ContentOpsExecutionResult` | `content_ops_execution.py:86-101` | `status`, `plan`, `steps`, `errors` |
| `SignalExtractionConfig` | `signal_extraction.py:19-24` | `limit`, `max_text_chars` |
| `SignalExtractionResult` | `signal_extraction.py:27-45` | `opportunities`, `warnings`, `target_mode` (+ `generated` property) |

### Catalog values

`OUTPUT_CATALOG` (`control_surfaces.py:98-153`) — six outputs:
`email_campaign`, `blog_post`, `report`, `landing_page`, `sales_brief`,
`signal_extraction`. All `implemented=true` today; only
`signal_extraction` has `estimated_unit_cost_usd=0.0` (deterministic,
no LLM call).

`PRESETS` (`control_surfaces.py:156-193`) — five presets:
`email_only`, `intelligence_report`, `content_marketing`,
`lead_gen_campaign`, `full_campaign`.

### Reasoning requirement vocabulary

`OutputDefinition.reasoning_requirement` is a string, not an enum,
but the values used in `OUTPUT_CATALOG` are:

- `"absent"` — output does not consume host reasoning context (default)
- `"optional_host_context"` — output uses reasoning context if the
  host wires `CampaignReasoningContextProvider`, otherwise runs
  without it

Outputs currently flagged `optional_host_context`: `email_campaign`,
`blog_post`, `report`, `landing_page`, `sales_brief`. Outputs flagged
`absent`: `signal_extraction`.

### Execution status vocabulary

`ContentOpsExecutionResult.status` (`content_ops_execution.py:147-155`):

- `"completed"` — every step succeeded
- `"failed"` — every step failed (distinguished from `partial`
  per PR-Audit-MINOR-Batch-3)
- `"partial"` — some steps failed, some succeeded
- `"blocked"` — `plan.can_execute=false`; nothing ran

`GenerationPlanStep.status`: `"runnable"` or `"blocked"`. Blocked
steps carry a `reason` string.

### HTTP status code mapping (`api/control_surfaces.py:266-272`)

The `/execute` route maps `ContentOpsExecutionResult.status` to HTTP:

| Execution status | HTTP code |
| --- | --- |
| `completed` | 200 |
| `partial` | 207 |
| `failed` | 502 |
| `blocked` | 400 |

`ValueError` from `request_from_mapping` → 400. Missing execution
services → 503.

---

## Frontend domain model

### Top-level abstraction — `ContentOpsRun`

A single run is the lifecycle of: catalog snapshot → request → preview
→ plan → optional execution → typed step results.

```ts
interface ContentOpsRun {
  // Snapshot of /content-ops/control-surfaces at run start
  catalog: ContentOpsCatalog;

  // The user's typed request (validated client-side against
  // ContentOpsRequestModel constraints before submit)
  request: ContentOpsRequest;

  // Preview is required before plan/execute
  preview: ControlSurfacePreview;

  // Plan + (optionally) execution result
  plan?: GenerationPlan;
  execution?: ContentOpsExecutionResult;
}
```

### Catalog

```ts
interface ContentOpsCatalog {
  outputs: OutputDefinitionView[];
  presets: ControlSurfacePresetView[];
  execution: { configured: boolean; configuredOutputs: string[] };
  reasoning: { configured: boolean };
  ingestionProfiles: string[];           // ["domain_specific", "manual", "existing_evidence"]
}

interface OutputDefinitionView {
  id: string;
  label: string;
  description: string;
  implemented: boolean;
  estimatedUnitCostUsd: number;
  defaultParseRetryAttempts: number;
  estimatedRetryAdjustedUnitCostUsd: number;   // worst-case preview budget
  requiredInputs: string[];
  defaultMaxItems: number;
  reasoningRequirement: "absent" | "optional_host_context";
  // Per-request, computed from host-injected services:
  executionConfigured: boolean;
  canExecute: boolean;                          // implemented AND executionConfigured
}

interface ControlSurfacePresetView {
  id: string;
  label: string;
  description: string;
  outputs: string[];
}
```

UI rules:
- `canExecute=false` ⇒ disable the "Execute" button on that output.
- `canExecute=false && implemented=true` ⇒ show "Host service not
  configured" badge (the package implements it; the host hasn't
  injected the runner).
- `implemented=false` ⇒ show "Coming soon" badge; user can still
  preview if they tick `allowUnimplementedOutputs`.
- Show catalog-level `reasoning.configured` near the run controls.
  For outputs whose `reasoningRequirement` is not `"absent"`, show
  whether host reasoning context is ready or unavailable. This is
  informational only; optional host reasoning does not block preview
  or planning.

### Request

```ts
interface ContentOpsRequest {
  targetMode: string;                  // default "vendor_retention"; min 1, max 80 chars
  preset: string | null;               // optional; max 80 chars
  outputs: string[];                   // max 20 entries; if non-empty overrides preset
  limit: number;                       // 1..1000; default 1
  maxCostUsd: number | null;           // > 0 if provided
  inputs: Record<string, unknown>;     // max 50 keys, depth 6, string values ≤ 10000 chars
  ingestionProfile: string;            // default "domain_specific"
  requireQualityGates: boolean;        // default true
  allowUnimplementedOutputs: boolean;  // default false
}
```

The pydantic constraints come from
`ContentOpsRequestModel` (`api/control_surfaces.py:157-185`) and the
input-shape validator (`_MAX_INPUT_KEYS=50`, `_MAX_INPUT_DEPTH=6`,
`_MAX_INPUT_STRING_CHARS=10000`).

### Preview

```ts
interface ControlSurfacePreview {
  canRun: boolean;
  outputs: string[];
  estimatedCostUsd: number;            // rounded to 4 decimals server-side
  missingInputs: string[];
  blockedOutputs: string[];
  warnings: string[];
  normalizedRequest: ContentOpsRequest | null;
}
```

UI rules:
- `canRun=false` ⇒ disable "Plan" / "Execute" buttons.
- `missingInputs` drives form-field validation hints (each name is
  one of the `OutputDefinitionView.requiredInputs` values across
  selected outputs).
- `blockedOutputs` ⇒ strike through the corresponding output cards
  with the matching `warnings[]` message inline.
- `warnings` always render as non-blocking advisories.
- `normalizedRequest` is the canonical request shape after
  preset/outputs resolution; show it in the "Execution contract"
  panel so the user can see exactly what the backend received.

### Plan

```ts
interface GenerationPlan {
  canExecute: boolean;
  targetMode: string;
  limit: number;
  steps: GenerationPlanStep[];
  preview: ControlSurfacePreview;
}

interface GenerationPlanStep {
  output: string;                      // e.g. "email_campaign"
  runner: string;                      // e.g. "CampaignGenerationService.generate"
  status: "runnable" | "blocked";
  config: Record<string, unknown>;     // runner-specific config snapshot
  reason: string;                      // populated when status="blocked"
}
```

The `config` per-step shapes are runner-specific. Examples (from
`generation_plan.py:166-260`):

- **`email_campaign`** → `CampaignGenerationConfig`-shaped: `skill_name`,
  `channels`, `limit`, `max_tokens`, `temperature`,
  `quality_revalidation_enabled`, `quality_prompt_proof_term_limit`,
  `parse_retry_attempts`, `parse_retry_response_excerpt_chars`.
- **`report`** → `default_report_type`, `limit`, `max_tokens`,
  `temperature`, `quality_gates_enabled`, retry knobs.
- **`signal_extraction`** → `limit`, `max_text_chars`.
- All others have a similar shape; the UI should render `config` as
  a read-only key-value panel (the backend execution contract), not
  decorative settings.

UI rules:
- A `blocked` step's `reason` is the diagnostic; surface it inline
  on the step card.
- `canExecute=true` ⇒ show "Execute" button; otherwise show why
  (`preview.warnings` + per-step `reason`).
- The button is enabled only when `plan.canExecute=true`,
  `catalog.execution.configured=true`, and every planned output is
  listed in `catalog.execution.configuredOutputs`.

### Execution result

```ts
interface ContentOpsExecutionResult {
  status: "completed" | "partial" | "failed" | "blocked";
  plan: GenerationPlan;
  steps: ContentOpsStepExecution[];
  errors: Array<Record<string, unknown>>;
}

interface ContentOpsStepExecution {
  output: string;
  runner: string;
  status: "completed" | "failed" | "skipped";   // step-level
  result: Record<string, unknown>;              // runner-specific result blob
  error: string;                                // populated when status="failed"
  reasoning?: ContentOpsStepReasoningAudit;     // compact readiness audit only
}

interface ContentOpsStepReasoningAudit {
  requirement: "absent" | "optional_host_context" | string;
  service_supports_reasoning: boolean;
  provider_configured: boolean;
  contexts_used?: number;
}
```

UI rules:
- HTTP status to user-facing banner: 200 → green; 207 → yellow with
  per-step results; 502 → red banner + per-step errors; 400 →
  blocked-plan panel.
- The step's `result` is runner-specific; render via per-output
  view adapters (see below).
- The MVP execute view may render `result` as read-only JSON while
  per-output adapters are still landing. It must still surface
  step-level status, runner, and `error` inline.
- `email_campaign` should summarize `generated`,
  `reasoning_contexts_used` when present, `saved_ids`, and `errors.length`
  before the raw JSON block so users do not need to inspect the result
  payload for the common draft-generation case.
- `blog_post`, `report`, `landing_page`, and `sales_brief` should
  summarize their shared generated-asset result shape: `requested`,
  `generated`, `skipped`, `reasoning_contexts_used` when present,
  `saved_ids`, and `errors.length`.
- `signal_extraction` should summarize `generated`, `target_mode`,
  `warnings.length`, and a short list of extracted opportunities
  before the raw JSON block.

### Signal extraction (special case)

`signal_extraction` is in `OUTPUT_CATALOG` but its result is a
pipeline artifact, not a content asset. The UI should treat it
distinctly from generated content.

```ts
interface SignalExtractionResultView {
  generated: number;                    // = opportunities.length
  targetMode: string;
  opportunities: ExtractedOpportunity[];
  warnings: SignalExtractionWarning[];
}

interface ExtractedOpportunity {
  // Shape produced by source_row_to_campaign_opportunity ->
  // normalize_campaign_opportunity_rows. Best-effort known fields:
  targetId?: string;
  sourceId?: string;
  sourceType?: string;
  companyName?: string;
  vendor?: string;
  contactEmail?: string;
  painPoints?: unknown;
  evidence?: unknown;
  // Anything else surfaces under "Raw metadata"
  raw: Record<string, unknown>;
}

interface SignalExtractionWarning {
  code: string;
  message: string;
  rowIndex: number | null;
  field: string | null;
}
```

The opportunity shape is intentionally flexible — the backend
normalizer (`source_row_to_campaign_opportunity` in
`campaign_source_adapters.py`) accepts heterogeneous source rows. The
UI should render known fields first-class and put unknown fields
under a collapsible "Raw metadata" section.

### Reasoning context (host-injected, optional per output)

When an output has `reasoning_requirement="optional_host_context"`
AND the host has wired `CampaignReasoningContextProvider`
(`extracted_content_pipeline/campaign_ports.py:171`), the runner
threads the resulting `CampaignReasoningContext`
(`campaign_ports.py:53-99`) into the prompt.

The catalog-level `reasoning.configured` flag tells the UI whether
the host mounted a route-level reasoning provider. It is separate
from each output's `reasoning_requirement`: an output can support
reasoning even when the current host has not wired a provider.

Important runtime boundary: `ContentOpsStepExecution.result` contains
the per-service summary returned by `as_dict()` (counts, saved IDs,
warnings, and errors). It does **not** reliably expose the consumed
reasoning payload. `ContentOpsStepExecution.reasoning` is a compact
readiness audit only: it tells the UI whether the output can use host
reasoning, whether the service supports the seam, and whether a
provider was attached. When the service result includes
`reasoning_contexts_used`, the audit also exposes `contexts_used` as
the count of generated assets that actually consumed reasoning context.
A Reasoning Context Drawer still requires a future field that carries
the consumed context itself.

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
  raw: Record<string, unknown>;        // canonical_reasoning + anything unknown
}
```

UI rules:
- Show a "Reasoning context" badge next to each output card when
  `reasoningRequirement="optional_host_context"`.
- Treat catalog badges as configuration readiness only. Use
  `step.reasoning` for execution-level readiness ("provider attached" /
  "provider absent") but do not render a context drawer from it; it is
  not the prompt payload.

---

## Frontend layering

### API adapter layer

One module per route group; snake_case ↔ camelCase translation only.

```
src/api/
  contentOpsControlSurfaces.ts   // GET  /content-ops/control-surfaces
  contentOpsPreview.ts           // POST /content-ops/preview
  contentOpsPlan.ts              // POST /content-ops/plan
  contentOpsExecute.ts           // POST /content-ops/execute
```

### Domain layer

Owns the typed models above. No HTTP, no React.

```
src/domain/
  contentOpsCatalog.ts
  contentOpsRequest.ts
  controlSurfacePreview.ts
  generationPlan.ts
  contentOpsExecutionResult.ts
  signalExtraction.ts
  campaignReasoningContext.ts
```

### View-model layer

- Output-card list (driven by `catalog.outputs`)
- Preset selector (driven by `catalog.presets`)
- Required-input form fields (driven by `output.requiredInputs`)
- Preview result panel (driven by `ControlSurfacePreview`)
- Plan step list (driven by `GenerationPlan.steps`)
- Execution result panel (driven by `ContentOpsExecutionResult`)
- Per-output result render adapters
  (`email_campaign | blog_post | report | landing_page | sales_brief
  | signal_extraction`) — each translates `step.result` into a
  presentation
- Generated-asset result summary for `blog_post`, `report`,
  `landing_page`, and `sales_brief`, all of which currently expose
  `requested`, `generated`, `skipped`, `saved_ids`, and `errors`
- Signal extraction table (driven by `SignalExtractionResultView`)

### UI layer

Dumb components only. No fetch, no business rules.

---

## MVP screens

1. **New Run / Control Surface**
   - Loads `GET /content-ops/control-surfaces`.
   - Renders preset picker, output picker, required-input form,
     options (`limit`, `maxCostUsd`, `requireQualityGates`,
     `allowUnimplementedOutputs`, `ingestionProfile`).
   - On submit: `POST /content-ops/preview`, render
     `ControlSurfacePreview`.

2. **Plan Preview**
   - Triggered when preview `canRun=true`; calls `POST /content-ops/plan`.
   - Renders `GenerationPlan.steps` with config panels.
   - "Execute" button enabled when `plan.canExecute=true` and host
     execution services are configured (`catalog.execution.configured`).

3. **Execution / Run Result**
   - Calls `POST /content-ops/execute`.
   - Renders status banner per HTTP code.
   - Per-step results via render adapters.
   - Errors panel from `result.errors`.

4. **Signal Extraction Review**
   - When `signal_extraction` was in the run, render its result via
     the special-case view.

---

## Out of scope for this contract

- Final dashboard metrics
- Full asset editor UX
- Visual workflow builder
- Model-selection UX
- Reasoning Context Drawer until `/content-ops/execute` exposes an
  explicit consumed-reasoning context or reasoning-audit result field.
- Collaboration / role-permission flows
- CMS export
- Approval workflow (no backend approval state on the control-surface
  routes today; lives elsewhere in the host)

These can land later; they don't block v0 of the frontend.

---

## Code references summary

Every claim in this doc cites a real file:line at HEAD `a4020c1`:

| Topic | Citation |
| --- | --- |
| 4 routes | `extracted_content_pipeline/api/control_surfaces.py:212-272` |
| Pydantic body model | `extracted_content_pipeline/api/control_surfaces.py:157-185` |
| Body input bounds (`_MAX_INPUT_KEYS=50` etc.) | `extracted_content_pipeline/api/control_surfaces.py:50-52` |
| Catalog response composition | `extracted_content_pipeline/api/control_surfaces.py:101-131` |
| Static catalog cache (PR #397) | `extracted_content_pipeline/api/control_surfaces.py:55-98` |
| `OutputDefinition` | `extracted_content_pipeline/control_surfaces.py:16-30` |
| `ControlSurfacePreset` | `extracted_content_pipeline/control_surfaces.py:33-40` |
| `ContentOpsRequest` | `extracted_content_pipeline/control_surfaces.py:43-55` |
| `ControlSurfacePreview` | `extracted_content_pipeline/control_surfaces.py:58-90` |
| `OUTPUT_CATALOG` (6 outputs) | `extracted_content_pipeline/control_surfaces.py:98-153` |
| `PRESETS` (5 presets) | `extracted_content_pipeline/control_surfaces.py:156-193` |
| `preview_control_surface` | `extracted_content_pipeline/control_surfaces.py:306-381` |
| `GenerationPlan` / `GenerationPlanStep` | `extracted_content_pipeline/generation_plan.py:28-65` |
| Per-output config builders | `extracted_content_pipeline/generation_plan.py:68-148` |
| `_step_for_output` per-runner shapes | `extracted_content_pipeline/generation_plan.py:165-266` |
| `ContentOpsExecutionServices` | `extracted_content_pipeline/content_ops_execution.py:16-63` |
| `ContentOpsExecutionResult` status logic | `extracted_content_pipeline/content_ops_execution.py:147-155` |
| `ContentOpsStepExecution` | `extracted_content_pipeline/content_ops_execution.py:66-83` |
| `SignalExtractionConfig` / `SignalExtractionResult` / `SignalExtractionService` | `extracted_content_pipeline/signal_extraction.py:19-81` |
| `CampaignReasoningContext` (11 fields) | `extracted_content_pipeline/campaign_ports.py:53-99` |

If a future revision adds claims, the same rule applies: every type
or field must cite a file:line. No silent extrapolation, and no
verification against a stale tree.
