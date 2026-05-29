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

export {
  FAQ_CONFIGURATION_OUTPUTS,
  FAQ_DEFLECTION_REPORT_CONFIGURATION_OUTPUT,
  FAQ_MARKDOWN_OUTPUT,
  faqConfigurationInputsSelected,
} from './faqConfigurationInputs'

export {
  FAQ_DEFLECTION_REPORT_OUTPUT,
  FAQ_RESOLUTION_EVIDENCE_STATUS,
  faqDeflectionReportAnswerSteps,
  faqDeflectionReportView,
  isProvenFAQDeflectionReportItem,
} from './faqDeflectionReport'
export type {
  FAQDeflectionReportAnswerTone,
  FAQDeflectionReportItemView,
  FAQDeflectionReportSummaryView,
  FAQDeflectionReportView,
  FAQTermMappingView,
} from './faqDeflectionReport'
