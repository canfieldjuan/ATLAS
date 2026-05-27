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
  ContentOpsIngestionDiagnosticsResponse,
  ContentOpsIngestionImportRequest as WireIngestionImportRequest,
  ContentOpsIngestionImportResponse as WireIngestionImportResponse,
  ContentOpsIngestionInspectRequest as WireIngestionInspectRequest,
  ContentOpsExecutionResult as WireExecutionResult,
  ContentOpsInputProviderDiagnostics as WireInputProviderDiagnostics,
  ContentOpsInputContract as WireInputContract,
  ContentOpsOutputDefinition as WireOutputDefinition,
  ContentOpsPreset as WirePreset,
  ContentOpsPreviewResponse,
  ContentOpsRequestBody,
  ContentOpsStepExecution as WireStepExecution,
  ContentOpsUsageBudgetEvaluationResponse as WireUsageBudgetEvaluation,
  ContentOpsUsageSummaryBreakdownResponse as WireUsageSummaryBreakdown,
  ContentOpsUsageSummaryResponse,
  GenerationPlanResponse,
  GenerationPlanStep as WirePlanStep,
} from '../../api/contentOps'
import type {
  ContentOpsCatalog,
  ContentOpsIngestionLimits,
  ContentOpsIngestionDiagnostics,
  ContentOpsIngestionImportRequest,
  ContentOpsIngestionImportResponse,
  ContentOpsIngestionInspectRequest,
  ContentOpsExecutionResult,
  ContentOpsInputContractView,
  ContentOpsInputProviderDiagnostics,
  ContentOpsInputProviderWarning,
  ContentOpsUsageSummary,
  ContentOpsUsageSummaryBreakdown,
  ContentOpsUsageBudgetEvaluation,
  CampaignReasoningContextView,
  ContentOpsRequest,
  ContentOpsStepExecution,
  ControlSurfacePresetView,
  ControlSurfacePreview,
  GenerationPlan,
  GenerationPlanStep,
  OutputDefinitionView,
  ReasoningCapabilityStatus,
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
    defaultQualityRepairAttempts: wire.default_quality_repair_attempts,
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

export function fromWireInputContract(
  wire: WireInputContract,
): ContentOpsInputContractView {
  return {
    key: wire.key,
    label: wire.label,
    type: wire.type,
    asset: wire.asset,
    group: wire.group,
    placeholder: wire.placeholder,
    min: wire.min,
    max: wire.max,
    default: wire.default,
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
      modes: copyScalarList(wire.reasoning.modes),
      packs: copyScalarList(wire.reasoning.packs),
      capabilities: copyReasoningCapabilities(wire.reasoning.capabilities),
    },
    ingestionProfiles: [...wire.ingestion_profiles],
    ingestionLimits: fromWireIngestionLimits(wire.ingestion_limits),
    inputContracts: Object.fromEntries(
      Object.entries(wire.input_contracts ?? {}).map(([key, contract]) => [
        key,
        fromWireInputContract(contract),
      ]),
    ),
  }
}

function fromWireIngestionLimits(
  wire: ContentOpsCatalogResponse['ingestion_limits'],
): ContentOpsIngestionLimits {
  return {
    inlineRows: {
      maxRows: wire.inline_rows.max_rows,
      deprecated: wire.inline_rows.deprecated,
    },
    fileUpload: {
      maxFileBytes: wire.file_upload.max_file_bytes,
      maxRows: wire.file_upload.max_rows,
      supportedFormats: [...wire.file_upload.supported_formats],
    },
    maxSourceTextChars: wire.max_source_text_chars,
    maxSampleLimit: wire.max_sample_limit,
  }
}

function copyScalarList(
  value: Array<string | number | boolean> | undefined,
): Array<string | number | boolean> | undefined {
  return value ? [...value] : undefined
}

function copyReasoningCapabilities(
  value:
    | Array<string | number | boolean>
    | Record<string, ReasoningCapabilityStatus>
    | undefined,
):
  | Array<string | number | boolean>
  | Record<string, ReasoningCapabilityStatus>
  | undefined {
  if (!value) return undefined
  if (Array.isArray(value)) return [...value]
  return Object.fromEntries(
    Object.entries(value).map(([key, status]) => [
      key,
      {
        ...status,
        missing: status.missing ? [...status.missing] : undefined,
      },
    ]),
  )
}

// ---------------------------------------------------------------------------
// Usage summary
// ---------------------------------------------------------------------------

export function fromWireUsageSummary(
  wire: ContentOpsUsageSummaryResponse,
): ContentOpsUsageSummary {
  return {
    periodDays: wire.period_days,
    filters: {
      accountId: wire.filters.account_id,
      assetType: wire.filters.asset_type,
      runId: wire.filters.run_id,
      requestId: wire.filters.request_id,
    },
    summary: {
      totalCostUsd: wire.summary.total_cost_usd,
      totalCalls: wire.summary.total_calls,
      failedCalls: wire.summary.failed_calls,
      inputTokens: wire.summary.input_tokens,
      billableInputTokens: wire.summary.billable_input_tokens,
      outputTokens: wire.summary.output_tokens,
      totalTokens: wire.summary.total_tokens,
      cachedTokens: wire.summary.cached_tokens,
      cacheWriteTokens: wire.summary.cache_write_tokens,
      cacheHitCalls: wire.summary.cache_hit_calls,
      avgDurationMs: wire.summary.avg_duration_ms,
      latestCallAt: wire.summary.latest_call_at,
    },
    byModel: wire.by_model.map(fromWireUsageSummaryBreakdown),
    byAssetType: wire.by_asset_type.map(fromWireUsageSummaryBreakdown),
  }
}

function fromWireUsageSummaryBreakdown(
  wire: WireUsageSummaryBreakdown,
): ContentOpsUsageSummaryBreakdown {
  return {
    provider: wire.provider,
    model: wire.model,
    assetType: wire.asset_type,
    costUsd: wire.cost_usd,
    calls: wire.calls,
    inputTokens: wire.input_tokens,
    outputTokens: wire.output_tokens,
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
    accountUsageBudgetUsd: wire.account_usage_budget_usd ?? null,
    accountUsageBudgetDays: wire.account_usage_budget_days ?? 7,
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
    account_usage_budget_usd: domain.accountUsageBudgetUsd,
    account_usage_budget_days: domain.accountUsageBudgetDays,
    inputs: { ...domain.inputs },
    ingestion_profile: domain.ingestionProfile,
    require_quality_gates: domain.requireQualityGates,
    allow_unimplemented_outputs: domain.allowUnimplementedOutputs,
  }
}

// ---------------------------------------------------------------------------
// Ingestion inspect
// ---------------------------------------------------------------------------

export function toWireIngestionInspectRequest(
  domain: ContentOpsIngestionInspectRequest,
): WireIngestionInspectRequest {
  return {
    rows: domain.rows.map((row) => ({ ...row })),
    source_rows: domain.sourceRows,
    source: domain.source,
    target_mode: domain.targetMode,
    max_source_text_chars: domain.maxSourceTextChars,
    sample_limit: domain.sampleLimit,
    default_fields: { ...domain.defaultFields },
  }
}

export function toWireIngestionImportRequest(
  domain: ContentOpsIngestionImportRequest,
): WireIngestionImportRequest {
  return {
    ...toWireIngestionInspectRequest(domain),
    replace_existing: domain.replaceExisting,
    dry_run: domain.dryRun,
  }
}

export function fromWireIngestionDiagnostics(
  wire: ContentOpsIngestionDiagnosticsResponse,
): ContentOpsIngestionDiagnostics {
  return {
    ok: wire.ok,
    mode: wire.mode,
    source: wire.source,
    opportunityCount: wire.opportunity_count,
    warningCount: wire.warning_count,
    warningCounts: { ...wire.warning_counts },
    missingFieldCounts: { ...wire.missing_field_counts },
    sourceTypeCounts: { ...wire.source_type_counts },
    samples: wire.samples.map((row) => ({ ...row })),
    warnings: wire.warnings.map((warning) => ({
      code: warning.code,
      message: warning.message,
      rowIndex: warning.row_index,
      field: warning.field,
    })),
  }
}

export function fromWireIngestionImportResponse(
  wire: WireIngestionImportResponse,
): ContentOpsIngestionImportResponse {
  return {
    diagnostics: fromWireIngestionDiagnostics(wire.diagnostics),
    importResult: {
      inserted: wire.import.inserted,
      skipped: wire.import.skipped,
      dryRun: wire.import.dry_run,
      replaceExisting: wire.import.replace_existing,
      targetIds: [...wire.import.target_ids],
      source: wire.import.source,
      warnings: wire.import.warnings.map((warning) => ({
        code: warning.code,
        message: warning.message,
        rowIndex: warning.row_index,
        field: warning.field,
      })),
    },
  }
}

// ---------------------------------------------------------------------------
// Preview
// ---------------------------------------------------------------------------

export function fromWirePreview(
  wire: ContentOpsPreviewResponse,
): ControlSurfacePreview {
  const preview: ControlSurfacePreview = {
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
          accountUsageBudgetUsd:
            wire.normalized_request.account_usage_budget_usd ?? null,
          accountUsageBudgetDays:
            wire.normalized_request.account_usage_budget_days ?? 7,
          ingestionProfile:
            wire.normalized_request.ingestion_profile ?? 'domain_specific',
          requireQualityGates:
            wire.normalized_request.require_quality_gates ?? true,
          allowUnimplementedOutputs:
            wire.normalized_request.allow_unimplemented_outputs ?? false,
        }
      : null,
  }
  const inputProvider = fromWireInputProviderDiagnostics(wire.input_provider)
  if (inputProvider) {
    preview.inputProvider = inputProvider
  }
  if (wire.usage_budget) {
    preview.usageBudget = fromWireUsageBudgetEvaluation(wire.usage_budget)
  }
  return preview
}

export function fromWireUsageBudgetEvaluation(
  wire: WireUsageBudgetEvaluation,
): ContentOpsUsageBudgetEvaluation {
  return {
    budgetUsd: wire.budget_usd,
    periodDays: wire.period_days,
    currentCostUsd: wire.current_cost_usd,
    estimatedCostUsd: wire.estimated_cost_usd,
    projectedCostUsd: wire.projected_cost_usd,
    exceeded: wire.exceeded,
  }
}

export function fromWireInputProviderDiagnostics(
  wire: WireInputProviderDiagnostics | undefined,
): ContentOpsInputProviderDiagnostics | undefined {
  if (!wire) return undefined
  return {
    provider: wire.provider,
    metadata: { ...wire.metadata },
    warnings: (wire.warnings ?? []).map(fromWireInputProviderWarning),
  }
}

function fromWireInputProviderWarning(
  warning: Record<string, unknown>,
): ContentOpsInputProviderWarning {
  const { code, message, ...details } = warning
  return {
    code: typeof code === 'string' && code ? code : 'input_provider_warning',
    message:
      typeof message === 'string' && message
        ? message
        : 'Input provider warning',
    details,
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
  const plan: GenerationPlan = {
    canExecute: wire.can_execute,
    targetMode: wire.target_mode,
    limit: wire.limit,
    steps: wire.steps.map(fromWirePlanStep),
    preview: fromWirePreview(wire.preview),
  }
  const inputProvider = fromWireInputProviderDiagnostics(wire.input_provider)
  if (inputProvider) {
    plan.inputProvider = inputProvider
  }
  return plan
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
  const result: ContentOpsExecutionResult = {
    status: wire.status,
    plan: fromWirePlan(wire.plan),
    steps: wire.steps.map(fromWireStepExecution),
    errors: wire.errors.map((e) => ({ ...e })),
  }
  const inputProvider = fromWireInputProviderDiagnostics(wire.input_provider)
  if (inputProvider) {
    result.inputProvider = inputProvider
  }
  return result
}
