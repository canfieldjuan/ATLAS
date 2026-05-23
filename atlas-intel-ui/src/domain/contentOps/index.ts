/**
 * Content Ops domain barrel.
 *
 * Screens and view-models import from this path; the underlying
 * `types.ts` / `fromWire.ts` split is implementation detail.
 */

export type {
  CampaignReasoningContextView,
  ContentOpsCatalog,
  ContentOpsIngestionDiagnostics,
  ContentOpsIngestionImportRequest,
  ContentOpsIngestionImportResponse,
  ContentOpsIngestionImportResult,
  ContentOpsIngestionInspectRequest,
  ContentOpsIngestionWarning,
  ContentOpsInputContractView,
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
  formatContentOpsBytes,
} from './ingestionLimits'

export {
  fromWireCatalog,
  fromWireExecution,
  fromWireIngestionDiagnostics,
  fromWireIngestionImportResponse,
  fromWireInputContract,
  fromWireOutputDefinition,
  fromWirePlan,
  fromWirePlanStep,
  fromWirePreset,
  fromWirePreview,
  fromWireRequest,
  fromWireStepExecution,
  toWireIngestionImportRequest,
  toWireIngestionInspectRequest,
  toWireRequest,
} from './fromWire'

export { inputContractDisplay } from './inputDisplay'
export type { ContentOpsInputDisplayFallback } from './inputDisplay'
