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
  BRAND_VOICE_PROFILE_PRESETS,
  applyBrandVoiceProfileEditorPatch,
  blankBrandVoiceProfileEditorState,
  brandVoicePresetEditorPatch,
  brandVoiceProfileEditorRequest,
  brandVoiceProfileEditorStateFromProfile,
  canSaveBrandVoiceProfileEditor,
  deriveBrandVoiceProfileEditorPatch,
} from './brandVoiceProfileEditor'
export type {
  BrandVoiceProfileEditorState,
  BrandVoiceProfilePreset,
} from './brandVoiceProfileEditor'

export {
  DEFAULT_SOCIAL_POST_CHANNELS,
  SOCIAL_POST_CHANNEL_OPTIONS,
  SOCIAL_POST_OUTPUT,
  normalizeSocialPostChannels,
  requestWithSocialPostChannels,
} from './socialPostChannels'
export type { SocialPostChannelId } from './socialPostChannels'

export {
  FAQ_CONFIGURATION_OUTPUTS,
  FAQ_DEFLECTION_REPORT_CONFIGURATION_OUTPUT,
  FAQ_DOCUMENTATION_TERMS_INPUT,
  FAQ_INTENT_RULES_INPUT,
  FAQ_MARKDOWN_OUTPUT,
  FAQ_VOCABULARY_GAP_RULES_INPUT,
  faqConfigurationControlsVisible,
  faqConfigurationInputsSelected,
  faqIntentRulesDraftValue,
  faqIntentRulesFromDraft,
} from './faqConfigurationInputs'
export type { FAQConfigurationInputContracts } from './faqConfigurationInputs'

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
