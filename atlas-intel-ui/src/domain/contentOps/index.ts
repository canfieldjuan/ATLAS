/**
 * Content Ops domain barrel.
 *
 * Screens and view-models import from this path; the underlying
 * `types.ts` / `fromWire.ts` split is implementation detail.
 */

export type {
  CampaignReasoningContextView,
  ContentOpsCachePolicy,
  ContentOpsCatalog,
  ContentOpsIngestionDiagnostics,
  ContentOpsIngestionImportRequest,
  ContentOpsIngestionImportResponse,
  ContentOpsIngestionImportResult,
  ContentOpsIngestionInspectRequest,
  ContentOpsIngestionWarning,
  ContentOpsInputProviderDiagnostics,
  ContentOpsInputProviderWarning,
  ContentOpsInputContractView,
  ContentOpsUsageSummary,
  ContentOpsUsageSummaryBreakdown,
  ContentOpsUsageSummaryFilters,
  ContentOpsUsageSummaryTotals,
  ContentOpsUsageBudgetEvaluation,
  ContentOpsExecutionResult,
  ContentOpsExecutionStatus,
  ContentOpsRequest,
  ContentOpsRun,
  ContentOpsStepExecution,
  ContentOpsStepReasoningAudit,
  ContentOpsStepStatus,
  ControlSurfacePresetView,
  ControlSurfacePreview,
  GenerationPlan,
  GenerationPlanStep,
  GenerationPlanStepStatus,
  OutputDefinitionView,
  ReasoningRequirement,
} from './types'

export {
  contentOpsIngestionFilePreflightError,
  contentOpsInlineRowsPreflightError,
  formatContentOpsBytes,
  formatContentOpsCount,
} from './ingestionLimits'

export {
  fromWireCatalog,
  fromWireExecution,
  fromWireIngestionDiagnostics,
  fromWireIngestionImportResponse,
  fromWireInputContract,
  fromWireInputProviderDiagnostics,
  fromWireOutputDefinition,
  fromWirePlan,
  fromWirePlanStep,
  fromWirePreset,
  fromWirePreview,
  fromWireRequest,
  fromWireStepExecution,
  fromWireUsageBudgetEvaluation,
  fromWireUsageSummary,
  toWireIngestionImportRequest,
  toWireIngestionInspectRequest,
  toWireRequest,
} from './fromWire'

export { inputContractDisplay } from './inputDisplay'
export type { ContentOpsInputDisplayFallback } from './inputDisplay'
