/**
 * Content Ops domain types (camelCase).
 *
 * Mirrors the wire interfaces in `src/api/contentOps.ts` but uses
 * idiomatic TS field names. Screens import from
 * `src/domain/contentOps` (the barrel) and never reach into the
 * API adapter directly.
 *
 * Contract reference: `docs/frontend/content_ops_frontend_contract.md`
 * (PR #401, backend HEAD `a4020c1`).
 */

// ---------------------------------------------------------------------------
// Catalog (GET /content-ops/control-surfaces)
// ---------------------------------------------------------------------------

export type ReasoningRequirement =
  | 'absent'
  | 'optional_host_context'
  | string

export interface OutputDefinitionView {
  id: string
  label: string
  description: string
  implemented: boolean
  estimatedUnitCostUsd: number
  defaultParseRetryAttempts: number
  estimatedRetryAdjustedUnitCostUsd: number
  requiredInputs: string[]
  defaultMaxItems: number
  reasoningRequirement: ReasoningRequirement
  // Per-request, computed from host-injected services:
  executionConfigured: boolean
  canExecute: boolean
}

export interface ControlSurfacePresetView {
  id: string
  label: string
  description: string
  outputs: string[]
}

export interface ContentOpsCatalog {
  outputs: OutputDefinitionView[]
  presets: ControlSurfacePresetView[]
  execution: {
    configured: boolean
    configuredOutputs: string[]
  }
  reasoning: {
    configured: boolean
    source?: 'db' | 'file' | 'none' | string
    modes?: Array<string | number | boolean>
    packs?: Array<string | number | boolean>
    capabilities?:
      | Array<string | number | boolean>
      | Record<string, ReasoningCapabilityStatus>
  }
  ingestionProfiles: string[]
}

export interface ReasoningCapabilityStatus {
  configured?: boolean
  ready?: boolean
  active?: boolean
  missing?: string[]
}

// ---------------------------------------------------------------------------
// Request (POST /content-ops/{preview, plan, execute} body)
// ---------------------------------------------------------------------------

export interface ContentOpsRequest {
  targetMode: string
  preset: string | null
  outputs: string[]
  limit: number
  maxCostUsd: number | null
  inputs: Record<string, unknown>
  ingestionProfile: string
  requireQualityGates: boolean
  allowUnimplementedOutputs: boolean
}

// ---------------------------------------------------------------------------
// Ingestion inspect (POST /content-ops/ingestion/inspect)
// ---------------------------------------------------------------------------

export interface ContentOpsIngestionInspectRequest {
  rows: Array<Record<string, unknown>>
  sourceRows: boolean
  source: string | null
  targetMode: string | null
  maxSourceTextChars: number
  sampleLimit: number
}

export interface ContentOpsIngestionWarning {
  code: string
  message: string
  rowIndex?: number
  field?: string
}

export interface ContentOpsIngestionDiagnostics {
  ok: boolean
  mode: 'opportunities' | 'source_rows'
  source: string
  opportunityCount: number
  warningCount: number
  warningCounts: Record<string, number>
  missingFieldCounts: Record<string, number>
  sourceTypeCounts: Record<string, number>
  samples: Array<Record<string, unknown>>
  warnings: ContentOpsIngestionWarning[]
}

// ---------------------------------------------------------------------------
// Preview (POST /content-ops/preview)
// ---------------------------------------------------------------------------

export interface ControlSurfacePreview {
  canRun: boolean
  outputs: string[]
  estimatedCostUsd: number
  missingInputs: string[]
  blockedOutputs: string[]
  warnings: string[]
  // The preview's normalized request omits `inputs` per the
  // backend's `as_dict()` contract (see contract doc); it is
  // typed as a partial of `ContentOpsRequest` here.
  normalizedRequest: Omit<ContentOpsRequest, 'inputs'> | null
}

// ---------------------------------------------------------------------------
// Plan (POST /content-ops/plan)
// ---------------------------------------------------------------------------

export type GenerationPlanStepStatus = 'runnable' | 'blocked'

export interface GenerationPlanStep {
  output: string
  runner: string
  status: GenerationPlanStepStatus
  config: Record<string, unknown>
  reason: string
}

export interface GenerationPlan {
  canExecute: boolean
  targetMode: string
  limit: number
  steps: GenerationPlanStep[]
  preview: ControlSurfacePreview
}

// ---------------------------------------------------------------------------
// Execution result (POST /content-ops/execute)
// ---------------------------------------------------------------------------

export type ContentOpsExecutionStatus =
  | 'completed'
  | 'partial'
  | 'failed'
  | 'blocked'

export type ContentOpsStepStatus = 'completed' | 'failed' | 'skipped'

export interface ContentOpsStepReasoningAudit {
  requirement: ReasoningRequirement
  serviceSupportsReasoning: boolean
  providerConfigured: boolean
  contextsUsed?: number
  consumedContexts?: CampaignReasoningContextView[]
}

export interface CampaignReasoningContextView {
  summary?: string
  anchorExamples?: Record<string, Array<Record<string, unknown>>>
  witnessHighlights?: Array<Record<string, unknown>>
  referenceIds?: Record<string, string[]>
  topTheses?: Array<Record<string, unknown>>
  accountSignals?: Array<Record<string, unknown>>
  timingWindows?: Array<Record<string, unknown>>
  proofPoints?: Array<Record<string, unknown>>
  coverageLimits?: string[]
  scopeSummary?: Record<string, unknown>
  deltaSummary?: Record<string, unknown>
  extra: Record<string, unknown>
}

export interface ContentOpsStepExecution {
  output: string
  runner: string
  status: ContentOpsStepStatus
  result: Record<string, unknown>
  error: string
  reasoning?: ContentOpsStepReasoningAudit
}

export interface ContentOpsExecutionResult {
  status: ContentOpsExecutionStatus
  plan: GenerationPlan
  steps: ContentOpsStepExecution[]
  errors: Array<Record<string, unknown>>
}

// ---------------------------------------------------------------------------
// Aggregate
// ---------------------------------------------------------------------------

/**
 * A complete Content Ops run lifecycle.
 *
 * Starts with `catalog` + `request`; `preview`, `plan`, and
 * `execution` are populated as the run progresses through the
 * three steps. A blocked run may stop after `preview`; a runnable
 * one proceeds through `plan` and (optionally) `execution`.
 */
export interface ContentOpsRun {
  catalog: ContentOpsCatalog
  request: ContentOpsRequest
  preview?: ControlSurfacePreview
  plan?: GenerationPlan
  execution?: ContentOpsExecutionResult
}
