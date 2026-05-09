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
  ingestionProfiles: string[]
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

export interface ContentOpsStepExecution {
  output: string
  runner: string
  status: ContentOpsStepStatus
  result: Record<string, unknown>
  error: string
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
