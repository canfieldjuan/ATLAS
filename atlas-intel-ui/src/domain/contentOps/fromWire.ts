/**
 * Wire-to-domain mappers for Content Ops.
 *
 * Translates the snake_case API adapter response shapes into
 * camelCase domain types. Pure data transforms; no React, no
 * HTTP, no side effects.
 *
 * Screens import the result of these mappers via
 * `src/domain/contentOps` (the barrel), never the wire types
 * directly.
 */

import type {
  ContentOpsCatalogResponse,
  ContentOpsExecutionResult as WireExecutionResult,
  ContentOpsOutputDefinition as WireOutputDefinition,
  ContentOpsPreset as WirePreset,
  ContentOpsPreviewResponse,
  ContentOpsRequestBody,
  ContentOpsStepExecution as WireStepExecution,
  GenerationPlanResponse,
  GenerationPlanStep as WirePlanStep,
} from '../../api/contentOps'
import type {
  ContentOpsCatalog,
  ContentOpsExecutionResult,
  CampaignReasoningContextView,
  ContentOpsRequest,
  ContentOpsStepExecution,
  ControlSurfacePresetView,
  ControlSurfacePreview,
  GenerationPlan,
  GenerationPlanStep,
  OutputDefinitionView,
} from './types'

// ---------------------------------------------------------------------------
// Catalog
// ---------------------------------------------------------------------------

export function fromWireOutputDefinition(
  wire: WireOutputDefinition,
): OutputDefinitionView {
  return {
    id: wire.id,
    label: wire.label,
    description: wire.description,
    implemented: wire.implemented,
    estimatedUnitCostUsd: wire.estimated_unit_cost_usd,
    defaultParseRetryAttempts: wire.default_parse_retry_attempts,
    estimatedRetryAdjustedUnitCostUsd: wire.estimated_retry_adjusted_unit_cost_usd,
    requiredInputs: [...wire.required_inputs],
    defaultMaxItems: wire.default_max_items,
    reasoningRequirement: wire.reasoning_requirement,
    executionConfigured: wire.execution_configured,
    canExecute: wire.can_execute,
  }
}

export function fromWirePreset(wire: WirePreset): ControlSurfacePresetView {
  return {
    id: wire.id,
    label: wire.label,
    description: wire.description,
    outputs: [...wire.outputs],
  }
}

export function fromWireCatalog(
  wire: ContentOpsCatalogResponse,
): ContentOpsCatalog {
  return {
    outputs: wire.outputs.map(fromWireOutputDefinition),
    presets: wire.presets.map(fromWirePreset),
    execution: {
      configured: wire.execution.configured,
      configuredOutputs: [...wire.execution.configured_outputs],
    },
    reasoning: {
      configured: wire.reasoning.configured,
      source: wire.reasoning.source,
    },
    ingestionProfiles: [...wire.ingestion_profiles],
  }
}

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

/**
 * Translate a wire request body (all fields optional) into a
 * domain request with explicit defaults applied. Mirrors the
 * pydantic defaults at `api/control_surfaces.py:79-92`.
 */
export function fromWireRequest(
  wire: ContentOpsRequestBody,
): ContentOpsRequest {
  return {
    targetMode: wire.target_mode ?? 'vendor_retention',
    preset: wire.preset ?? null,
    outputs: wire.outputs ? [...wire.outputs] : [],
    limit: wire.limit ?? 1,
    maxCostUsd: wire.max_cost_usd ?? null,
    inputs: { ...(wire.inputs ?? {}) },
    ingestionProfile: wire.ingestion_profile ?? 'domain_specific',
    requireQualityGates: wire.require_quality_gates ?? true,
    allowUnimplementedOutputs: wire.allow_unimplemented_outputs ?? false,
  }
}

/**
 * Inverse of `fromWireRequest`: domain to wire body. Used when
 * a screen submits a request to the backend.
 */
export function toWireRequest(
  domain: ContentOpsRequest,
): ContentOpsRequestBody {
  return {
    target_mode: domain.targetMode,
    preset: domain.preset,
    outputs: [...domain.outputs],
    limit: domain.limit,
    max_cost_usd: domain.maxCostUsd,
    inputs: { ...domain.inputs },
    ingestion_profile: domain.ingestionProfile,
    require_quality_gates: domain.requireQualityGates,
    allow_unimplemented_outputs: domain.allowUnimplementedOutputs,
  }
}

// ---------------------------------------------------------------------------
// Preview
// ---------------------------------------------------------------------------

export function fromWirePreview(
  wire: ContentOpsPreviewResponse,
): ControlSurfacePreview {
  return {
    canRun: wire.can_run,
    outputs: [...wire.outputs],
    estimatedCostUsd: wire.estimated_cost_usd,
    missingInputs: [...wire.missing_inputs],
    blockedOutputs: [...wire.blocked_outputs],
    warnings: [...wire.warnings],
    normalizedRequest: wire.normalized_request
      ? {
          targetMode: wire.normalized_request.target_mode ?? 'vendor_retention',
          preset: wire.normalized_request.preset ?? null,
          outputs: wire.normalized_request.outputs
            ? [...wire.normalized_request.outputs]
            : [],
          limit: wire.normalized_request.limit ?? 1,
          maxCostUsd: wire.normalized_request.max_cost_usd ?? null,
          ingestionProfile:
            wire.normalized_request.ingestion_profile ?? 'domain_specific',
          requireQualityGates:
            wire.normalized_request.require_quality_gates ?? true,
          allowUnimplementedOutputs:
            wire.normalized_request.allow_unimplemented_outputs ?? false,
        }
      : null,
  }
}

// ---------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------

export function fromWirePlanStep(wire: WirePlanStep): GenerationPlanStep {
  return {
    output: wire.output,
    runner: wire.runner,
    status: wire.status,
    config: { ...wire.config },
    reason: wire.reason,
  }
}

export function fromWirePlan(wire: GenerationPlanResponse): GenerationPlan {
  return {
    canExecute: wire.can_execute,
    targetMode: wire.target_mode,
    limit: wire.limit,
    steps: wire.steps.map(fromWirePlanStep),
    preview: fromWirePreview(wire.preview),
  }
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

export function fromWireStepExecution(
  wire: WireStepExecution,
): ContentOpsStepExecution {
  const step: ContentOpsStepExecution = {
    output: wire.output,
    runner: wire.runner,
    status: wire.status,
    result: { ...wire.result },
    error: wire.error,
  }
  if (wire.reasoning) {
    step.reasoning = {
      requirement: wire.reasoning.requirement,
      serviceSupportsReasoning: wire.reasoning.service_supports_reasoning,
      providerConfigured: wire.reasoning.provider_configured,
    }
    if (typeof wire.reasoning.contexts_used === 'number') {
      step.reasoning.contextsUsed = wire.reasoning.contexts_used
    }
    if (Array.isArray(wire.reasoning.consumed_contexts)) {
      step.reasoning.consumedContexts =
        wire.reasoning.consumed_contexts
          .filter(isRecord)
          .map(fromWireReasoningContext)
    }
  }
  return step
}

function fromWireReasoningContext(
  wire: Record<string, unknown>,
): CampaignReasoningContextView {
  const known = new Set([
    'summary',
    'anchor_examples',
    'witness_highlights',
    'reference_ids',
    'top_theses',
    'account_signals',
    'timing_windows',
    'proof_points',
    'coverage_limits',
    'scope_summary',
    'delta_summary',
  ])
  const extra = Object.fromEntries(
    Object.entries(wire).filter(([key]) => !known.has(key)),
  )
  return {
    summary: typeof wire.summary === 'string' ? wire.summary : undefined,
    anchorExamples: isRecordOfRecordArrays(wire.anchor_examples)
      ? wire.anchor_examples
      : undefined,
    witnessHighlights: recordArray(wire.witness_highlights),
    referenceIds: toReferenceIds(wire.reference_ids),
    topTheses: recordArray(wire.top_theses),
    accountSignals: recordArray(wire.account_signals),
    timingWindows: recordArray(wire.timing_windows),
    proofPoints: recordArray(wire.proof_points),
    coverageLimits: stringArray(wire.coverage_limits),
    scopeSummary: isRecord(wire.scope_summary) ? { ...wire.scope_summary } : undefined,
    deltaSummary: isRecord(wire.delta_summary) ? { ...wire.delta_summary } : undefined,
    extra,
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function recordArray(value: unknown): Array<Record<string, unknown>> | undefined {
  return Array.isArray(value) ? value.filter(isRecord).map((row) => ({ ...row })) : undefined
}

function stringArray(value: unknown): string[] | undefined {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === 'string')
    : undefined
}

function isRecordOfRecordArrays(
  value: unknown,
): value is Record<string, Array<Record<string, unknown>>> {
  if (!isRecord(value)) return false
  return Object.values(value).every(
    (rows) => Array.isArray(rows) && rows.every(isRecord),
  )
}

function toReferenceIds(value: unknown): Record<string, string[]> | undefined {
  if (!isRecord(value)) return undefined
  const entries = Object.entries(value).map(([key, values]) => [
    key,
    stringArray(values) ?? [],
  ])
  return Object.fromEntries(entries)
}

export function fromWireExecution(
  wire: WireExecutionResult,
): ContentOpsExecutionResult {
  return {
    status: wire.status,
    plan: fromWirePlan(wire.plan),
    steps: wire.steps.map(fromWireStepExecution),
    errors: wire.errors.map((e) => ({ ...e })),
  }
}
