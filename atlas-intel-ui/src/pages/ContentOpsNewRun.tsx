import { useMemo, useRef, useState, type ChangeEvent } from 'react'
import {
  ChevronRight,
  FileUp,
  Loader2,
  Play,
  RefreshCw,
  Search,
  Upload,
} from 'lucide-react'
import { clsx } from 'clsx'
import {
  executeContentOpsRun,
  fetchContentOpsControlSurfaces,
  importContentOpsIngestion,
  importContentOpsIngestionFile,
  inspectContentOpsIngestion,
  inspectContentOpsIngestionFile,
  planContentOpsRun,
  previewContentOpsRun,
} from '../api/contentOps'
import {
  fromWireCatalog,
  fromWireExecution,
  fromWireIngestionDiagnostics,
  fromWireIngestionImportResponse,
  fromWirePlan,
  fromWirePreview,
  fromWireRequest,
  inputContractDisplay,
  toWireIngestionInspectRequest,
  toWireIngestionImportRequest,
  toWireRequest,
  type ContentOpsCatalog,
  type ContentOpsIngestionDiagnostics,
  type ContentOpsIngestionImportResponse,
  type ContentOpsExecutionResult,
  type ContentOpsInputContractView,
  type ContentOpsRequest,
  type CampaignReasoningContextView,
  type ContentOpsStepReasoningAudit,
  type ControlSurfacePreview,
  type GenerationPlan,
  type GenerationPlanStep,
} from '../domain/contentOps'
import useApiData from '../hooks/useApiData'
import { PageError } from '../components/ErrorBoundary'

type SubmitState =
  | { kind: 'idle' }
  | { kind: 'submitting' }
  | { kind: 'invalid_inputs_json'; message: string }
  | { kind: 'invalid_max_cost'; message: string }
  | { kind: 'error'; message: string }
  | { kind: 'success'; preview: ControlSurfacePreview }

// Screen 2 (Plan Preview): parallel state for the plan response.
// Lives alongside SubmitState so the user can see preview + plan
// stacked. Form mutations clear both via markStale().
type PlanState =
  | { kind: 'idle' }
  | { kind: 'submitting' }
  | { kind: 'error'; message: string }
  | { kind: 'success'; plan: GenerationPlan }

type ExecutionState =
  | { kind: 'idle' }
  | { kind: 'submitting' }
  | { kind: 'error'; message: string }
  | { kind: 'success'; result: ContentOpsExecutionResult }

type IngestionInspectState =
  | { kind: 'idle' }
  | { kind: 'submitting' }
  | { kind: 'invalid_input'; message: string }
  | { kind: 'error'; message: string }
  | { kind: 'success'; diagnostics: ContentOpsIngestionDiagnostics }

type IngestionImportState =
  | { kind: 'idle' }
  | { kind: 'submitting' }
  | { kind: 'invalid_input'; message: string }
  | { kind: 'not_ready'; diagnostics: ContentOpsIngestionDiagnostics }
  | { kind: 'error'; message: string }
  | { kind: 'success'; response: ContentOpsIngestionImportResponse }

type IngestionFileLoadState =
  | { kind: 'idle' }
  | { kind: 'loading' }
  | { kind: 'error'; message: string }
  | { kind: 'success'; filename: string; size: number }

const DEFAULT_INPUTS_JSON = '{\n  \n}'
const DEFAULT_INGESTION_ROWS_JSON = '[\n  \n]'
const DEFAULT_INGESTION_DEFAULT_FIELDS_JSON = '{\n  \n}'
const DEFAULT_INGESTION_SOURCE = 'manual'
const INGESTION_SAMPLE_LIMIT = 5
const INGESTION_MAX_SOURCE_TEXT_CHARS = 1200
const LANDING_PAGE_QUALITY_REPAIR_INPUT =
  'landing_page_quality_repair_attempts'
const LANDING_PAGE_INPUT_ASSET = 'landing_page'
const LANDING_PAGE_SEO_GEO_AEO_INPUT_GROUP = 'seo_geo_aeo'
const FAQ_MARKDOWN_OUTPUT = 'faq_markdown'
const FAQ_DOCUMENTATION_TERMS_INPUT = 'faq_documentation_terms'
const FAQ_VOCABULARY_GAP_RULES_INPUT = 'faq_vocabulary_gap_rules'
const FAQ_DOCUMENTATION_TERMS_DISPLAY_FALLBACK = {
  label: 'Documentation terms',
  placeholder: 'Single sign-on setup\nData export guide',
}
const FAQ_VOCABULARY_GAP_RULES_DISPLAY_FALLBACK = {
  label: 'Vocabulary-gap rules',
  placeholder: 'SSO, single sign-on\nexport, data export',
}
const LANDING_PAGE_SEO_GEO_AEO_INPUT_ORDER = [
  'target_keyword',
  'secondary_keywords',
  'search_intent',
  'primary_entity',
  'audience_entity',
  'competitors',
  'objections',
  'faq_questions',
  'source_period',
  'internal_links',
  'cta_label',
  'cta_url',
]
const INVALID_LANDING_PAGE_QUALITY_REPAIR_VALUE = '__invalid__'
const LEGACY_LANDING_PAGE_REPAIR_ATTEMPT_CONTRACT: IntegerInputContract = {
  key: LANDING_PAGE_QUALITY_REPAIR_INPUT,
  label: 'Landing page quality repair attempts',
  type: 'integer',
  min: 0,
  max: 10,
  default: 1,
}

export default function ContentOpsNewRun() {
  const {
    data: wireCatalog,
    loading,
    error,
    refresh,
    refreshing,
  } = useApiData(() => fetchContentOpsControlSurfaces(), [])

  const catalog = useMemo<ContentOpsCatalog | null>(
    () => (wireCatalog ? fromWireCatalog(wireCatalog) : null),
    [wireCatalog],
  )

  const [request, setRequest] = useState<ContentOpsRequest>(() =>
    fromWireRequest({}),
  )
  // Codex P2 fix v3: keep the max-cost input as a string draft so the
  // user can type `0.` mid-entry without React re-rendering as `0` and
  // swallowing the decimal point. Parsed to a number at submit time.
  const [maxCostUsdInput, setMaxCostUsdInput] = useState<string>('')
  const [inputsJson, setInputsJson] = useState<string>(DEFAULT_INPUTS_JSON)
  const [ingestionRowsJson, setIngestionRowsJson] = useState<string>(
    DEFAULT_INGESTION_ROWS_JSON,
  )
  const [ingestionDefaultFieldsJson, setIngestionDefaultFieldsJson] =
    useState<string>(DEFAULT_INGESTION_DEFAULT_FIELDS_JSON)
  const [ingestionSourceRows, setIngestionSourceRows] = useState(false)
  const [ingestionSource, setIngestionSource] = useState(DEFAULT_INGESTION_SOURCE)
  const [ingestionDryRun, setIngestionDryRun] = useState(true)
  const [ingestionReplaceExisting, setIngestionReplaceExisting] = useState(false)
  const [ingestionFileLoadState, setIngestionFileLoadState] =
    useState<IngestionFileLoadState>({ kind: 'idle' })
  const [selectedIngestionFile, setSelectedIngestionFile] = useState<File | null>(
    null,
  )
  const [submitState, setSubmitState] = useState<SubmitState>({ kind: 'idle' })
  const [planState, setPlanState] = useState<PlanState>({ kind: 'idle' })
  const [executionState, setExecutionState] = useState<ExecutionState>({
    kind: 'idle',
  })
  const [ingestionInspectState, setIngestionInspectState] =
    useState<IngestionInspectState>({ kind: 'idle' })
  const [ingestionImportState, setIngestionImportState] =
    useState<IngestionImportState>({ kind: 'idle' })
  // Codex P2 fix: request-id ref so a stale in-flight preview/plan
  // response can't overwrite a result the user has since invalidated
  // by editing the form. Both preview and plan share the same id
  // namespace -- any form mutation invalidates both.
  const submitRequestIdRef = useRef(0)
  const ingestionInspectRequestIdRef = useRef(0)
  const ingestionImportRequestIdRef = useRef(0)
  const ingestionFileLoadRequestIdRef = useRef(0)
  const ingestionFileInputRef = useRef<HTMLInputElement | null>(null)
  const parsedInputsForControls = useMemo(
    () => parseInputsJsonObject(inputsJson),
    [inputsJson],
  )

  if (error) {
    return <PageError error={error} onRetry={refresh} />
  }
  if (loading || !catalog) {
    return (
      <div className="flex items-center justify-center py-24 text-slate-400">
        <Loader2 className="mr-2 h-5 w-5 animate-spin" />
        Loading control surfaces…
      </div>
    )
  }
  const reasoningConfigured = catalog.reasoning.configured
  const reasoningSource = catalog.reasoning.source
  const reasoningCapabilityHint = reasoningConfigured
    ? reasoningStatusHint(catalog.reasoning)
    : ''
  const landingPageOutputSelected = request.outputs.includes('landing_page')
  const faqMarkdownOutputSelected = request.outputs.includes(FAQ_MARKDOWN_OUTPUT)
  const faqDocumentationTermsContract = faqMarkdownOutputSelected
    ? catalog.inputContracts[FAQ_DOCUMENTATION_TERMS_INPUT]
    : undefined
  const faqVocabularyGapRulesContract = faqMarkdownOutputSelected
    ? catalog.inputContracts[FAQ_VOCABULARY_GAP_RULES_INPUT]
    : undefined
  const faqDocumentationTermsDisplay = inputContractDisplay(
    faqDocumentationTermsContract,
    FAQ_DOCUMENTATION_TERMS_DISPLAY_FALLBACK,
  )
  const faqVocabularyGapRulesDisplay = inputContractDisplay(
    faqVocabularyGapRulesContract,
    FAQ_VOCABULARY_GAP_RULES_DISPLAY_FALLBACK,
  )
  const landingPageRepairAttemptContract = landingPageOutputSelected
    ? integerInputContract(catalog, LANDING_PAGE_QUALITY_REPAIR_INPUT)
    : null
  const landingPageRepairAttemptOptions = landingPageRepairAttemptContract
    ? integerInputOptions(landingPageRepairAttemptContract)
    : []
  const landingPageRepairAttemptValue = landingPageOutputSelected
    ? landingPageRepairAttemptSelectValue(
        parsedInputsForControls,
        landingPageRepairAttemptContract,
      )
    : ''
  const landingPageRepairAttemptMessage = landingPageOutputSelected
    ? landingPageRepairAttemptHelpText(
        parsedInputsForControls,
        landingPageRepairAttemptContract,
      )
    : ''
  const landingPageInputContracts = landingPageOutputSelected
    ? landingPageSeoGeoAeoInputContracts(catalog)
    : []
  const landingPageInputsDisabled = !parsedInputsForControls.ok
  const faqInputsDisabled = !parsedInputsForControls.ok
  const landingPageRepairAttemptDisabled =
    !parsedInputsForControls.ok || !landingPageRepairAttemptContract

  // Codex P2 fix: any form mutation invalidates a stale preview verdict
  // and plan panel so the user never sees a "Can run" badge or plan
  // panel that doesn't match the current form state. Bumping the
  // request id also drops any in-flight preview / plan response so it
  // can't overwrite the panels for a newer form state.
  const markStale = () => {
    submitRequestIdRef.current += 1
    setSubmitState((prev) => (prev.kind === 'idle' ? prev : { kind: 'idle' }))
    setPlanState((prev) => (prev.kind === 'idle' ? prev : { kind: 'idle' }))
    setExecutionState((prev) => (prev.kind === 'idle' ? prev : { kind: 'idle' }))
  }

  const markIngestionStale = () => {
    ingestionInspectRequestIdRef.current += 1
    ingestionImportRequestIdRef.current += 1
    ingestionFileLoadRequestIdRef.current += 1
    setIngestionInspectState((prev) =>
      prev.kind === 'idle' ? prev : { kind: 'idle' },
    )
    setIngestionImportState((prev) =>
      prev.kind === 'idle' ? prev : { kind: 'idle' },
    )
  }

  const handleLandingPageRepairAttemptsChange = (value: string) => {
    const updated = updateLandingPageRepairAttemptsInputJson(
      inputsJson,
      value,
      landingPageRepairAttemptContract,
    )
    if (!updated.ok) return
    setInputsJson(updated.value)
    markStale()
  }

  const handleLandingPageInputChange = (
    contract: ContentOpsInputContractView,
    value: string,
  ) => {
    const updated = updateLandingPageInputJson(inputsJson, contract, value)
    if (!updated.ok) return
    setInputsJson(updated.value)
    markStale()
  }

  const handleFaqDocumentationTermsChange = (value: string) => {
    const updated = updateFaqDocumentationTermsInputJson(inputsJson, value)
    if (!updated.ok) return
    setInputsJson(updated.value)
    markStale()
  }

  const handleFaqVocabularyRulesChange = (value: string) => {
    const updated = updateFaqVocabularyRulesInputJson(inputsJson, value)
    if (!updated.ok) return
    setInputsJson(updated.value)
    markStale()
  }

  const togglePreset = (presetId: string) => {
    const preset = catalog.presets.find((p) => p.id === presetId)
    if (!preset) return
    setRequest((prev) => ({
      ...prev,
      preset: prev.preset === presetId ? null : presetId,
      outputs: prev.preset === presetId ? prev.outputs : [...preset.outputs],
    }))
    markStale()
  }

  const toggleOutput = (outputId: string) => {
    setRequest((prev) => {
      const has = prev.outputs.includes(outputId)
      return {
        ...prev,
        outputs: has
          ? prev.outputs.filter((o) => o !== outputId)
          : [...prev.outputs, outputId],
      }
    })
    markStale()
  }

  type ParsedRequest =
    | { ok: true; domainRequest: ContentOpsRequest }
    | {
        ok: false
        kind: 'invalid_inputs_json' | 'invalid_max_cost'
        message: string
      }

  // Shared submit-time parse for both preview and plan. Validates the
  // inputs JSON, landing-page repair override, and the max-cost string
  // draft against backend constraints before either round-trip.
  const buildDomainRequest = (): ParsedRequest => {
    let parsedInputs: Record<string, unknown>
    try {
      const trimmed = inputsJson.trim()
      parsedInputs = trimmed ? JSON.parse(trimmed) : {}
      if (!isRecord(parsedInputs)) {
        throw new Error('inputs must be a JSON object')
      }
    } catch (err) {
      return {
        ok: false,
        kind: 'invalid_inputs_json',
        message: err instanceof Error ? err.message : String(err),
      }
    }

    if (request.outputs.includes('landing_page')) {
      const repairAttemptContract =
        landingPageRepairAttemptContract ??
        LEGACY_LANDING_PAGE_REPAIR_ATTEMPT_CONTRACT
      const repairAttemptValue = parsedInputs[LANDING_PAGE_QUALITY_REPAIR_INPUT]
      if (
        repairAttemptValue !== null &&
        typeof repairAttemptValue !== 'undefined' &&
        !normalizeLandingPageRepairAttemptValue(
          repairAttemptValue,
          repairAttemptContract,
        ).ok
      ) {
        return {
          ok: false,
          kind: 'invalid_inputs_json',
          message: landingPageRepairAttemptErrorMessage(
            repairAttemptContract,
          ),
        }
      }
    }

    const trimmedMaxCost = maxCostUsdInput.trim()
    let normalizedMaxCost: number | null = null
    if (trimmedMaxCost !== '') {
      const parsed = Number(trimmedMaxCost)
      if (!Number.isFinite(parsed) || parsed <= 0) {
        return {
          ok: false,
          kind: 'invalid_max_cost',
          message:
            'Max cost must be a positive number (e.g. 0.50). Leave blank for no cap.',
        }
      }
      normalizedMaxCost = parsed
    }

    return {
      ok: true,
      domainRequest: {
        ...request,
        inputs: parsedInputs,
        maxCostUsd: normalizedMaxCost,
      },
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    // Bump the id so any prior in-flight request is ignored on resolve.
    const requestId = ++submitRequestIdRef.current
    setSubmitState({ kind: 'submitting' })
    // Codex P2 fix: also reset planState. A previewing-while-planning
    // sequence would otherwise leave planState stuck at 'submitting'
    // (its in-flight response is dropped by the requestId guard but
    // never overwritten), and the Build-plan CTA would stay disabled
    // indefinitely until the user mutates the form.
    setPlanState((prev) => (prev.kind === 'idle' ? prev : { kind: 'idle' }))
    setExecutionState((prev) => (prev.kind === 'idle' ? prev : { kind: 'idle' }))

    const parsed = buildDomainRequest()
    if (!parsed.ok) {
      if (requestId !== submitRequestIdRef.current) return
      setSubmitState({ kind: parsed.kind, message: parsed.message })
      return
    }

    try {
      const wirePreview = await previewContentOpsRun(toWireRequest(parsed.domainRequest))
      if (requestId !== submitRequestIdRef.current) return
      setSubmitState({ kind: 'success', preview: fromWirePreview(wirePreview) })
    } catch (err) {
      if (requestId !== submitRequestIdRef.current) return
      setSubmitState({
        kind: 'error',
        message: err instanceof Error ? err.message : String(err),
      })
    }
  }

  const handlePlan = async () => {
    // Bump id so a stale plan response can't overwrite a newer one.
    const requestId = ++submitRequestIdRef.current
    setPlanState({ kind: 'submitting' })
    setExecutionState((prev) => (prev.kind === 'idle' ? prev : { kind: 'idle' }))

    const parsed = buildDomainRequest()
    if (!parsed.ok) {
      if (requestId !== submitRequestIdRef.current) return
      setPlanState({ kind: 'error', message: parsed.message })
      return
    }

    try {
      const wirePlan = await planContentOpsRun(toWireRequest(parsed.domainRequest))
      if (requestId !== submitRequestIdRef.current) return
      setPlanState({ kind: 'success', plan: fromWirePlan(wirePlan) })
    } catch (err) {
      if (requestId !== submitRequestIdRef.current) return
      setPlanState({
        kind: 'error',
        message: err instanceof Error ? err.message : String(err),
      })
    }
  }

  const handleExecute = async () => {
    const requestId = ++submitRequestIdRef.current
    setExecutionState({ kind: 'submitting' })

    const parsed = buildDomainRequest()
    if (!parsed.ok) {
      if (requestId !== submitRequestIdRef.current) return
      setExecutionState({ kind: 'error', message: parsed.message })
      return
    }

    try {
      const outcome = await executeContentOpsRun(toWireRequest(parsed.domainRequest))
      if (requestId !== submitRequestIdRef.current) return
      if ('result' in outcome) {
        setExecutionState({
          kind: 'success',
          result: fromWireExecution(outcome.result),
        })
        return
      }
      setExecutionState({
        kind: 'error',
        message: executionDetailMessage(outcome.detail),
      })
    } catch (err) {
      if (requestId !== submitRequestIdRef.current) return
      setExecutionState({
        kind: 'error',
        message: err instanceof Error ? err.message : String(err),
      })
    }
  }

  const handleInspectIngestion = async () => {
    const requestId = ++ingestionInspectRequestIdRef.current
    const parsedRows = selectedIngestionFile
      ? null
      : parseIngestionRowsJson(ingestionRowsJson)
    if (parsedRows && !parsedRows.ok) {
      setIngestionInspectState({
        kind: 'invalid_input',
        message: parsedRows.message,
      })
      return
    }
    const parsedDefaultFields = parseIngestionDefaultFieldsJson(
      ingestionDefaultFieldsJson,
    )
    if (!parsedDefaultFields.ok) {
      setIngestionInspectState({
        kind: 'invalid_input',
        message: parsedDefaultFields.message,
      })
      return
    }
    const inlineRows = parsedRows?.ok ? parsedRows.rows : []

    setIngestionInspectState({ kind: 'submitting' })
    try {
      const wire = selectedIngestionFile
        ? await inspectContentOpsIngestionFile({
            file: selectedIngestionFile,
            source_rows: ingestionSourceRows,
            source: ingestionSource.trim() || null,
            target_mode: request.targetMode,
            file_format: 'auto',
            max_source_text_chars: INGESTION_MAX_SOURCE_TEXT_CHARS,
            sample_limit: INGESTION_SAMPLE_LIMIT,
            default_fields: parsedDefaultFields.fields,
          })
        : await inspectContentOpsIngestion(
            toWireIngestionInspectRequest({
              rows: inlineRows,
              sourceRows: ingestionSourceRows,
              source: ingestionSource.trim() || null,
              targetMode: request.targetMode,
              maxSourceTextChars: INGESTION_MAX_SOURCE_TEXT_CHARS,
              sampleLimit: INGESTION_SAMPLE_LIMIT,
              defaultFields: parsedDefaultFields.fields,
            }),
          )
      if (requestId !== ingestionInspectRequestIdRef.current) return
      setIngestionInspectState({
        kind: 'success',
        diagnostics: fromWireIngestionDiagnostics(wire),
      })
    } catch (err) {
      if (requestId !== ingestionInspectRequestIdRef.current) return
      setIngestionInspectState({
        kind: 'error',
        message: err instanceof Error ? err.message : String(err),
      })
    }
  }

  const handleImportIngestion = async () => {
    const requestId = ++ingestionImportRequestIdRef.current
    const inspectRequestId = ++ingestionInspectRequestIdRef.current
    const parsedRows = selectedIngestionFile
      ? null
      : parseIngestionRowsJson(ingestionRowsJson)
    if (parsedRows && !parsedRows.ok) {
      setIngestionImportState({
        kind: 'invalid_input',
        message: parsedRows.message,
      })
      return
    }
    const parsedDefaultFields = parseIngestionDefaultFieldsJson(
      ingestionDefaultFieldsJson,
    )
    if (!parsedDefaultFields.ok) {
      setIngestionImportState({
        kind: 'invalid_input',
        message: parsedDefaultFields.message,
      })
      return
    }
    const inlineRows = parsedRows?.ok ? parsedRows.rows : []

    setIngestionImportState({ kind: 'submitting' })
    try {
      const outcome = selectedIngestionFile
        ? await importContentOpsIngestionFile({
            file: selectedIngestionFile,
            source_rows: ingestionSourceRows,
            source: ingestionSource.trim() || null,
            target_mode: request.targetMode,
            file_format: 'auto',
            max_source_text_chars: INGESTION_MAX_SOURCE_TEXT_CHARS,
            sample_limit: INGESTION_SAMPLE_LIMIT,
            default_fields: parsedDefaultFields.fields,
            replace_existing: ingestionReplaceExisting,
            dry_run: ingestionDryRun,
          })
        : await importContentOpsIngestion(
            toWireIngestionImportRequest({
              rows: inlineRows,
              sourceRows: ingestionSourceRows,
              source: ingestionSource.trim() || null,
              targetMode: request.targetMode,
              maxSourceTextChars: INGESTION_MAX_SOURCE_TEXT_CHARS,
              sampleLimit: INGESTION_SAMPLE_LIMIT,
              defaultFields: parsedDefaultFields.fields,
              replaceExisting: ingestionReplaceExisting,
              dryRun: ingestionDryRun,
            }),
          )
      if (requestId !== ingestionImportRequestIdRef.current) return
      if (outcome.kind === 'success') {
        setIngestionImportState({
          kind: 'success',
          response: fromWireIngestionImportResponse(outcome.response),
        })
        if (inspectRequestId === ingestionInspectRequestIdRef.current) {
          setIngestionInspectState({
            kind: 'success',
            diagnostics: fromWireIngestionDiagnostics(outcome.response.diagnostics),
          })
        }
        return
      }
      if (outcome.kind === 'not_ready') {
        const diagnostics = fromWireIngestionDiagnostics(outcome.diagnostics)
        setIngestionImportState({ kind: 'not_ready', diagnostics })
        if (inspectRequestId === ingestionInspectRequestIdRef.current) {
          setIngestionInspectState({
            kind: 'success',
            diagnostics,
          })
        }
        return
      }
      setIngestionImportState({
        kind: 'error',
        message: importOutcomeMessage(outcome),
      })
    } catch (err) {
      if (requestId !== ingestionImportRequestIdRef.current) return
      setIngestionImportState({
        kind: 'error',
        message: err instanceof Error ? err.message : String(err),
      })
    }
  }

  const handleLoadIngestionFile = async (
    event: ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.currentTarget.files?.[0]
    event.currentTarget.value = ''
    if (!file) return

    ingestionFileLoadRequestIdRef.current += 1
    setIngestionFileLoadState({ kind: 'loading' })
    setSelectedIngestionFile(file)
    setIngestionSource(file.name)
    markIngestionStale()
    setIngestionFileLoadState({
      kind: 'success',
      filename: file.name,
      size: file.size,
    })
  }

  return (
    <div className="mx-auto max-w-5xl px-6 py-10">
      <div className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-slate-100">Content Ops · New Run</h1>
          <p className="mt-1 text-sm text-slate-400">
            Pick a preset or outputs, supply inputs, preview the run before
            you commit.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span
            className={clsx(
              'rounded-full border px-3 py-1 text-xs font-medium',
              reasoningConfigured
                ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300'
                : 'border-slate-700 bg-slate-900 text-slate-400',
            )}
          >
            Reasoning {reasoningConfigured ? 'ready' : 'not configured'}
            {reasoningCapabilityHint && (
              <span className="ml-1 text-slate-400">
                {reasoningCapabilityHint}
              </span>
            )}
          </span>
          <button
            type="button"
            onClick={refresh}
            disabled={refreshing}
            className="flex items-center gap-2 rounded-md border border-slate-700 px-3 py-1.5 text-sm text-slate-300 hover:bg-slate-800 disabled:opacity-50"
          >
            <RefreshCw className={clsx('h-4 w-4', refreshing && 'animate-spin')} />
            Refresh catalog
          </button>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-8">
        {/* Preset picker */}
        <section>
          <h2 className="mb-2 text-sm font-medium uppercase tracking-wide text-slate-400">
            Preset
          </h2>
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {catalog.presets.map((preset) => {
              const selected = request.preset === preset.id
              return (
                <button
                  key={preset.id}
                  type="button"
                  onClick={() => togglePreset(preset.id)}
                  className={clsx(
                    'rounded-lg border px-4 py-3 text-left transition',
                    selected
                      ? 'border-cyan-500 bg-cyan-500/10 text-slate-100'
                      : 'border-slate-700 bg-slate-900 text-slate-300 hover:border-slate-500',
                  )}
                >
                  <div className="text-sm font-medium">{preset.label}</div>
                  <div className="mt-0.5 text-xs text-slate-400">{preset.description}</div>
                  <div className="mt-2 text-xs text-slate-500">
                    {preset.outputs.join(' · ')}
                  </div>
                </button>
              )
            })}
          </div>
        </section>

        {/* Output picker */}
        <section>
          <h2 className="mb-2 text-sm font-medium uppercase tracking-wide text-slate-400">
            Outputs
          </h2>
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {catalog.outputs.map((output) => {
              const checked = request.outputs.includes(output.id)
              return (
                <label
                  key={output.id}
                  className={clsx(
                    'flex cursor-pointer items-start gap-3 rounded-lg border px-3 py-2 transition',
                    checked
                      ? 'border-cyan-500 bg-cyan-500/10'
                      : 'border-slate-700 bg-slate-900 hover:border-slate-500',
                    !output.implemented && 'opacity-60',
                  )}
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => toggleOutput(output.id)}
                    className="mt-1 h-4 w-4 rounded border-slate-600 bg-slate-800 accent-cyan-500"
                  />
                  <div className="flex-1">
                    <div className="flex items-center gap-2 text-sm font-medium text-slate-100">
                      {output.label}
                      {!output.canExecute && output.implemented && (
                        <span className="rounded bg-amber-500/10 px-1.5 py-0.5 text-[10px] text-amber-400">
                          host service not configured
                        </span>
                      )}
                      {!output.implemented && (
                        <span className="rounded bg-slate-700 px-1.5 py-0.5 text-[10px] text-slate-400">
                          coming soon
                        </span>
                      )}
                      {output.reasoningRequirement !== 'absent' && (
                        <span
                          className={clsx(
                            'rounded px-1.5 py-0.5 text-[10px]',
                            reasoningConfigured
                              ? 'bg-emerald-500/10 text-emerald-300'
                              : 'bg-slate-700 text-slate-400',
                          )}
                        >
                          {reasoningConfigured
                            ? `reasoning ready${reasoningSource ? ` (${reasoningSource})` : ''}`
                            : 'reasoning unavailable'}
                        </span>
                      )}
                    </div>
                    <div className="mt-0.5 text-xs text-slate-400">{output.description}</div>
                    <div className="mt-1 text-[11px] text-slate-500">
                      requires:{' '}
                      {output.requiredInputs.length === 0
                        ? '(none)'
                        : output.requiredInputs.join(', ')}{' '}
                      · est ${output.estimatedRetryAdjustedUnitCostUsd.toFixed(2)}/item
                    </div>
                  </div>
                </label>
              )
            })}
          </div>
        </section>

        {/* Inputs (v0: raw JSON) */}
        <section>
          <h2 className="mb-2 text-sm font-medium uppercase tracking-wide text-slate-400">
            Inputs (JSON)
          </h2>
          <p className="mb-2 text-xs text-slate-500">
            Required keys depend on the selected outputs (see chip list
            above). Type a JSON object or use the per-output fields below.
          </p>
          <textarea
            value={inputsJson}
            onChange={(e) => {
              setInputsJson(e.target.value)
              markStale()
            }}
            rows={6}
            spellCheck={false}
            className="w-full rounded-md border border-slate-700 bg-slate-900 px-3 py-2 font-mono text-xs text-slate-200 focus:border-cyan-500 focus:outline-none"
            placeholder='{"target_account": "Acme", "offer": "Audit"}'
          />
          {landingPageOutputSelected && landingPageInputContracts.length > 0 && (
            <div className="mt-3 rounded-md border border-slate-800 bg-slate-900/70 p-3">
              <div className="mb-3">
                <h3 className="text-sm font-medium text-slate-200">
                  Landing page SEO/GEO/AEO inputs
                </h3>
                <p className="mt-1 text-xs text-slate-500">
                  These fields write to the inputs JSON and flow into the
                  landing-page campaign context.
                </p>
              </div>
              <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
                {landingPageInputContracts.map((contract) => {
                  const value = landingPageInputDraftValue(
                    parsedInputsForControls,
                    contract,
                  )
                  const controlClass =
                    'mt-1 w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-sm text-slate-200 placeholder:text-slate-600 focus:border-cyan-500 focus:outline-none disabled:cursor-not-allowed disabled:opacity-60'

                  return (
                    <label key={contract.key} className="block text-sm">
                      <span className="text-slate-300">{contract.label}</span>
                      {contract.type === 'string_list' ? (
                        <textarea
                          value={value}
                          onChange={(e) =>
                            handleLandingPageInputChange(contract, e.target.value)
                          }
                          rows={3}
                          disabled={landingPageInputsDisabled}
                          className={controlClass}
                          placeholder={contract.placeholder}
                        />
                      ) : (
                        <input
                          type="text"
                          value={value}
                          onChange={(e) =>
                            handleLandingPageInputChange(contract, e.target.value)
                          }
                          disabled={landingPageInputsDisabled}
                          className={controlClass}
                          placeholder={contract.placeholder}
                        />
                      )}
                    </label>
                  )
                })}
              </div>
              {!parsedInputsForControls.ok && (
                <p className="mt-2 text-xs text-amber-200">
                  Fix inputs JSON before changing landing-page fields:{' '}
                  {parsedInputsForControls.message}
                </p>
              )}
            </div>
          )}
          {landingPageOutputSelected && (
            <div className="mt-3 rounded-md border border-slate-800 bg-slate-900/70 p-3">
              <label className="block text-sm">
                <span className="text-slate-300">
                  Landing page quality repair attempts
                </span>
                <select
                  value={landingPageRepairAttemptValue}
                  onChange={(e) =>
                    handleLandingPageRepairAttemptsChange(e.target.value)
                  }
                  disabled={landingPageRepairAttemptDisabled}
                  className="mt-1 w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-slate-200 focus:border-cyan-500 focus:outline-none disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {landingPageRepairAttemptValue ===
                    INVALID_LANDING_PAGE_QUALITY_REPAIR_VALUE && (
                    <option value={INVALID_LANDING_PAGE_QUALITY_REPAIR_VALUE}>
                      Invalid value in inputs JSON
                    </option>
                  )}
                  <option value="">
                    {landingPageRepairAttemptContract
                      ? `Backend default (${landingPageRepairAttemptContract.default})`
                      : 'Backend default'}
                  </option>
                  {landingPageRepairAttemptOptions.map((option) => (
                    <option key={option} value={option}>
                      {option === '0'
                        ? '0 - disable repair'
                        : `${option} repair attempt${option === '1' ? '' : 's'}`}
                    </option>
                  ))}
                </select>
              </label>
              <p
                className={clsx(
                  'mt-2 text-xs',
                  landingPageRepairAttemptValue ===
                    INVALID_LANDING_PAGE_QUALITY_REPAIR_VALUE ||
                    !parsedInputsForControls.ok
                    ? 'text-amber-200'
                    : 'text-slate-500',
                )}
              >
                {landingPageRepairAttemptMessage}
              </p>
            </div>
          )}
          {faqMarkdownOutputSelected && (
            <div className="mt-3 rounded-md border border-slate-800 bg-slate-900/70 p-3">
              <div className="mb-3">
                <h3 className="text-sm font-medium text-slate-200">
                  FAQ vocabulary-gap inputs
                </h3>
              </div>
              <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
                <label className="block text-sm">
                  <span className="text-slate-300">
                    {faqDocumentationTermsDisplay.label}
                  </span>
                  <textarea
                    value={faqDocumentationTermsDraftValue(parsedInputsForControls)}
                    onChange={(e) =>
                      handleFaqDocumentationTermsChange(e.target.value)
                    }
                    rows={4}
                    disabled={faqInputsDisabled}
                    className="mt-1 w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-sm text-slate-200 placeholder:text-slate-600 focus:border-cyan-500 focus:outline-none disabled:cursor-not-allowed disabled:opacity-60"
                    placeholder={faqDocumentationTermsDisplay.placeholder}
                  />
                </label>
                <label className="block text-sm">
                  <span className="text-slate-300">
                    {faqVocabularyGapRulesDisplay.label}
                  </span>
                  <textarea
                    value={faqVocabularyRulesDraftValue(parsedInputsForControls)}
                    onChange={(e) =>
                      handleFaqVocabularyRulesChange(e.target.value)
                    }
                    rows={4}
                    disabled={faqInputsDisabled}
                    className="mt-1 w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-sm text-slate-200 placeholder:text-slate-600 focus:border-cyan-500 focus:outline-none disabled:cursor-not-allowed disabled:opacity-60"
                    placeholder={faqVocabularyGapRulesDisplay.placeholder}
                  />
                </label>
              </div>
              {!parsedInputsForControls.ok && (
                <p className="mt-2 text-xs text-amber-200">
                  Fix inputs JSON before changing FAQ fields:{' '}
                  {parsedInputsForControls.message}
                </p>
              )}
            </div>
          )}
        </section>

        {/* Ingestion inspection */}
        <section className="rounded-lg border border-slate-800 bg-slate-900/50 p-4">
          <div className="mb-3 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h2 className="text-sm font-medium uppercase tracking-wide text-slate-400">
                Ingestion inspector
              </h2>
              <p className="mt-1 text-xs text-slate-500">
                Paste or load customer export rows before running generation.
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                onClick={() => ingestionFileInputRef.current?.click()}
                disabled={ingestionFileLoadState.kind === 'loading'}
                className="flex items-center justify-center gap-2 rounded-md border border-slate-600 bg-slate-800/70 px-3 py-1.5 text-xs font-medium text-slate-200 hover:bg-slate-800 focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400 disabled:opacity-50"
              >
                <FileUp className="h-3.5 w-3.5" />
                Load JSON/JSONL/CSV
              </button>
              <input
                ref={ingestionFileInputRef}
                type="file"
                accept=".json,.jsonl,.ndjson,.csv,application/json,application/x-ndjson,text/csv"
                aria-label="Load ingestion JSON, JSONL, or CSV file"
                onChange={handleLoadIngestionFile}
                className="hidden"
              />
              <button
                type="button"
                onClick={handleInspectIngestion}
                disabled={
                  ingestionInspectState.kind === 'submitting' ||
                  ingestionFileLoadState.kind === 'loading'
                }
                className="flex items-center justify-center gap-2 rounded-md border border-cyan-500/40 bg-cyan-500/10 px-3 py-1.5 text-xs font-medium text-cyan-200 hover:bg-cyan-500/20 disabled:opacity-50"
              >
                {ingestionInspectState.kind === 'submitting' ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <Search className="h-3.5 w-3.5" />
                )}
                Inspect rows
              </button>
              <button
                type="button"
                onClick={handleImportIngestion}
                disabled={
                  ingestionImportState.kind === 'submitting' ||
                  ingestionFileLoadState.kind === 'loading'
                }
                className="flex items-center justify-center gap-2 rounded-md border border-emerald-500/40 bg-emerald-500/10 px-3 py-1.5 text-xs font-medium text-emerald-200 hover:bg-emerald-500/20 disabled:opacity-50"
              >
                {ingestionImportState.kind === 'submitting' ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <Upload className="h-3.5 w-3.5" />
                )}
                {ingestionDryRun ? 'Dry-run import' : 'Import rows'}
              </button>
            </div>
          </div>
          <div className="mb-3 grid grid-cols-1 gap-3 sm:grid-cols-3">
            <label className="block text-sm sm:col-span-2">
              <span className="text-slate-300">Source label</span>
              <input
                type="text"
                value={ingestionSource}
                onChange={(e) => {
                  setIngestionSource(e.target.value)
                  markIngestionStale()
                }}
                className="mt-1 w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-slate-200 focus:border-cyan-500 focus:outline-none"
              />
            </label>
            <label className="flex items-end gap-2 text-sm">
              <input
                type="checkbox"
                checked={ingestionSourceRows}
                onChange={(e) => {
                  setIngestionSourceRows(e.target.checked)
                  markIngestionStale()
                }}
                className="mb-1 h-4 w-4 rounded border-slate-600 bg-slate-800 accent-cyan-500"
              />
              <span className="pb-0.5 text-slate-300">Rows are source exports</span>
            </label>
          </div>
          <div className="mb-3 flex flex-wrap gap-x-6 gap-y-2 text-sm">
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={ingestionDryRun}
                onChange={(e) => {
                  setIngestionDryRun(e.target.checked)
                  setIngestionImportState({ kind: 'idle' })
                }}
                className="h-4 w-4 rounded border-slate-600 bg-slate-800 accent-cyan-500"
              />
              <span className="text-slate-300">Dry run only</span>
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={ingestionReplaceExisting}
                onChange={(e) => {
                  setIngestionReplaceExisting(e.target.checked)
                  setIngestionImportState({ kind: 'idle' })
                }}
                className="h-4 w-4 rounded border-slate-600 bg-slate-800 accent-cyan-500"
              />
              <span className="text-slate-300">Replace existing targets</span>
            </label>
          </div>
          <label className="mb-3 block text-sm">
            <span className="text-slate-300">Fallback fields JSON</span>
            <textarea
              value={ingestionDefaultFieldsJson}
              onChange={(e) => {
                setIngestionDefaultFieldsJson(e.target.value)
                markIngestionStale()
              }}
              rows={3}
              spellCheck={false}
              className="mt-1 w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2 font-mono text-xs text-slate-200 focus:border-cyan-500 focus:outline-none"
              placeholder='{"company_name": "Acme", "contact_email": "ops@example.com"}'
            />
          </label>
          <textarea
            value={ingestionRowsJson}
            onChange={(e) => {
              setSelectedIngestionFile(null)
              setIngestionFileLoadState({ kind: 'idle' })
              setIngestionRowsJson(e.target.value)
              markIngestionStale()
            }}
            rows={5}
            spellCheck={false}
            className="w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2 font-mono text-xs text-slate-200 focus:border-cyan-500 focus:outline-none"
            placeholder='[{"company_name": "Acme", "vendor": "HubSpot", "email": "ops@example.com"}]'
          />
          <IngestionFileLoadResult state={ingestionFileLoadState} />
          <IngestionInspectResult state={ingestionInspectState} />
          <IngestionImportResult state={ingestionImportState} />
        </section>

        {/* Options */}
        <section>
          <h2 className="mb-2 text-sm font-medium uppercase tracking-wide text-slate-400">
            Options
          </h2>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <label className="block text-sm">
              <span className="text-slate-300">Limit</span>
              <input
                type="number"
                min={1}
                max={1000}
                value={request.limit}
                onChange={(e) => {
                  setRequest((p) => ({
                    ...p,
                    limit: Math.max(1, Math.min(1000, Number(e.target.value) || 1)),
                  }))
                  markStale()
                }}
                className="mt-1 w-full rounded-md border border-slate-700 bg-slate-900 px-2 py-1 text-slate-200 focus:border-cyan-500 focus:outline-none"
              />
            </label>
            <label className="block text-sm">
              <span className="text-slate-300">Max cost (USD)</span>
              <input
                type="text"
                inputMode="decimal"
                value={maxCostUsdInput}
                onChange={(e) => {
                  // Codex P2 fix v3: render from the raw string draft
                  // so the user can type "0.", "0.5", "0.50" without
                  // mid-keystroke coercion to a number that re-renders
                  // as "0" and swallows the decimal point. Parsed to a
                  // number only at submit time.
                  setMaxCostUsdInput(e.target.value)
                  markStale()
                }}
                placeholder="(no cap)"
                className="mt-1 w-full rounded-md border border-slate-700 bg-slate-900 px-2 py-1 text-slate-200 focus:border-cyan-500 focus:outline-none"
              />
            </label>
            <label className="block text-sm">
              <span className="text-slate-300">Ingestion profile</span>
              <select
                value={request.ingestionProfile}
                onChange={(e) => {
                  setRequest((p) => ({ ...p, ingestionProfile: e.target.value }))
                  markStale()
                }}
                className="mt-1 w-full rounded-md border border-slate-700 bg-slate-900 px-2 py-1 text-slate-200 focus:border-cyan-500 focus:outline-none"
              >
                {catalog.ingestionProfiles.map((profile) => (
                  <option key={profile} value={profile}>
                    {profile}
                  </option>
                ))}
              </select>
            </label>
            <div className="space-y-2 text-sm">
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={request.requireQualityGates}
                  onChange={(e) => {
                    setRequest((p) => ({ ...p, requireQualityGates: e.target.checked }))
                    markStale()
                  }}
                  className="h-4 w-4 rounded border-slate-600 bg-slate-800 accent-cyan-500"
                />
                <span className="text-slate-300">Require quality gates</span>
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={request.allowUnimplementedOutputs}
                  onChange={(e) => {
                    setRequest((p) => ({
                      ...p,
                      allowUnimplementedOutputs: e.target.checked,
                    }))
                    markStale()
                  }}
                  className="h-4 w-4 rounded border-slate-600 bg-slate-800 accent-cyan-500"
                />
                <span className="text-slate-300">Allow unimplemented outputs</span>
              </label>
            </div>
          </div>
        </section>

        {/* Submit */}
        <section className="flex items-center justify-between border-t border-slate-800 pt-4">
          <div className="text-xs text-slate-500">
            {request.outputs.length === 0
              ? 'Pick at least one output (or a preset).'
              : `${request.outputs.length} output(s) selected.`}
          </div>
          <button
            type="submit"
            disabled={
              submitState.kind === 'submitting' || request.outputs.length === 0
            }
            className="flex items-center gap-2 rounded-md bg-cyan-500 px-4 py-2 text-sm font-medium text-slate-900 hover:bg-cyan-400 disabled:opacity-50"
          >
            {submitState.kind === 'submitting' ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Play className="h-4 w-4" />
            )}
            Preview
          </button>
        </section>
      </form>

      {/* Preview verdict */}
      {submitState.kind === 'invalid_inputs_json' && (
        <div className="mt-6 rounded-md border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
          Invalid inputs JSON: {submitState.message}
        </div>
      )}
      {submitState.kind === 'invalid_max_cost' && (
        <div className="mt-6 rounded-md border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
          Invalid max cost: {submitState.message}
        </div>
      )}
      {submitState.kind === 'error' && (
        <div className="mt-6 rounded-md border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
          Preview failed: {submitState.message}
        </div>
      )}
      {submitState.kind === 'success' && (
        <PreviewVerdict
          preview={submitState.preview}
          planState={planState}
          onBuildPlan={handlePlan}
          executionConfigured={catalog.execution.configured}
        />
      )}
      {planState.kind === 'success' && (
        <PlanPanel
          plan={planState.plan}
          executionConfigured={catalog.execution.configured}
          configuredOutputs={catalog.execution.configuredOutputs}
          executionState={executionState}
          onExecute={handleExecute}
        />
      )}
      {planState.kind === 'error' && (
        <div className="mt-6 rounded-md border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
          Plan failed: {planState.message}
        </div>
      )}
      {executionState.kind === 'error' && (
        <div className="mt-6 rounded-md border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
          Execute failed: {executionState.message}
        </div>
      )}
      {executionState.kind === 'success' && (
        <ExecutionPanel result={executionState.result} />
      )}
    </div>
  )
}

function PreviewVerdict({
  preview,
  planState,
  onBuildPlan,
  executionConfigured,
}: {
  preview: ControlSurfacePreview
  planState: PlanState
  onBuildPlan: () => void
  executionConfigured: boolean
}) {
  const planning = planState.kind === 'submitting'
  return (
    <section className="mt-8 rounded-lg border border-slate-800 bg-slate-900/60 p-5">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <span
            className={clsx(
              'rounded-full px-3 py-0.5 text-xs font-medium',
              preview.canRun
                ? 'bg-emerald-500/20 text-emerald-300'
                : 'bg-amber-500/20 text-amber-200',
            )}
          >
            {preview.canRun ? 'Can run' : 'Blocked'}
          </span>
          <span className="text-sm text-slate-400">
            Estimated cost: ${preview.estimatedCostUsd.toFixed(2)}
          </span>
        </div>
        {preview.canRun && (
          <button
            type="button"
            onClick={onBuildPlan}
            disabled={planning}
            title={
              executionConfigured
                ? 'Build the plan that the backend would execute.'
                : 'Build a read-only plan (host execution services not configured).'
            }
            className="flex items-center gap-2 rounded-md border border-cyan-500/40 bg-cyan-500/10 px-3 py-1.5 text-xs font-medium text-cyan-200 hover:bg-cyan-500/20 disabled:opacity-50"
          >
            {planning ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <ChevronRight className="h-3.5 w-3.5" />
            )}
            Build plan
          </button>
        )}
      </div>

      {preview.outputs.length > 0 && (
        <Section label="Outputs">
          <div className="text-sm text-slate-200">
            {preview.outputs.join(', ')}
          </div>
        </Section>
      )}

      {preview.missingInputs.length > 0 && (
        <Section label="Missing inputs">
          <ul className="ml-4 list-disc text-sm text-amber-200">
            {preview.missingInputs.map((name) => (
              <li key={name}>{name}</li>
            ))}
          </ul>
        </Section>
      )}

      {preview.blockedOutputs.length > 0 && (
        <Section label="Blocked outputs">
          <ul className="ml-4 list-disc text-sm text-amber-200">
            {preview.blockedOutputs.map((name) => (
              <li key={name}>{name}</li>
            ))}
          </ul>
        </Section>
      )}

      {preview.warnings.length > 0 && (
        <Section label="Warnings">
          <ul className="ml-4 list-disc text-sm text-slate-300">
            {preview.warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </Section>
      )}

      {preview.normalizedRequest && (
        <Section label="Normalized request">
          <pre className="overflow-x-auto rounded-md bg-slate-950/80 p-3 font-mono text-xs text-slate-300">
            {JSON.stringify(preview.normalizedRequest, null, 2)}
          </pre>
        </Section>
      )}
    </section>
  )
}

function Section({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="mt-3 first:mt-0">
      <div className="mb-1 text-[11px] font-medium uppercase tracking-wide text-slate-500">
        {label}
      </div>
      {children}
    </div>
  )
}

function IngestionInspectResult({ state }: { state: IngestionInspectState }) {
  if (state.kind === 'idle' || state.kind === 'submitting') {
    return null
  }
  if (state.kind === 'invalid_input') {
    return (
      <div className="mt-3 rounded-md border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-xs text-rose-200">
        Invalid ingestion payload: {state.message}
      </div>
    )
  }
  if (state.kind === 'error') {
    return (
      <div className="mt-3 rounded-md border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-xs text-rose-200">
        Ingestion inspect failed: {state.message}
      </div>
    )
  }

  const diagnostics = state.diagnostics
  return (
    <div className="mt-3 rounded-md border border-slate-800 bg-slate-950/50 p-3">
      <div className="mb-2 flex flex-wrap items-center gap-3 text-xs">
        <span
          className={clsx(
            'rounded-full px-2.5 py-0.5 font-medium',
            diagnostics.ok
              ? 'bg-emerald-500/20 text-emerald-300'
              : 'bg-amber-500/20 text-amber-200',
          )}
        >
          {diagnostics.ok ? 'Ready' : 'Needs attention'}
        </span>
        <span className="text-slate-400">Mode: {diagnostics.mode}</span>
        <span className="text-slate-400">
          Opportunities: {diagnostics.opportunityCount}
        </span>
        <span className="text-slate-400">
          Warnings: {diagnostics.warningCount}
        </span>
      </div>
      {diagnostics.warnings.length > 0 && (
        <Section label="Warnings">
          <ul className="ml-4 list-disc text-xs text-amber-100">
            {diagnostics.warnings.slice(0, INGESTION_SAMPLE_LIMIT).map((warning, i) => (
              <li key={`${warning.code}-${warning.rowIndex ?? i}-${warning.field ?? ''}`}>
                {warning.code}: {warning.message}
              </li>
            ))}
          </ul>
        </Section>
      )}
      {diagnostics.samples.length > 0 && (
        <Section label="Sample rows">
          <pre className="max-h-52 overflow-auto rounded-md bg-slate-950/80 p-3 font-mono text-[11px] text-slate-300">
            {JSON.stringify(diagnostics.samples, null, 2)}
          </pre>
        </Section>
      )}
    </div>
  )
}

function IngestionFileLoadResult({ state }: { state: IngestionFileLoadState }) {
  if (state.kind === 'idle') {
    return null
  }
  if (state.kind === 'loading') {
    return (
      <div className="mt-3 flex items-center gap-2 rounded-md border border-slate-700 bg-slate-950/50 px-3 py-2 text-xs text-slate-300">
        <Loader2 className="h-3.5 w-3.5 animate-spin" />
        Loading ingestion file...
      </div>
    )
  }
  if (state.kind === 'error') {
    return (
      <div className="mt-3 rounded-md border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-xs text-rose-200">
        File load failed: {state.message}
      </div>
    )
  }
  return (
    <div className="mt-3 rounded-md border border-slate-700 bg-slate-950/50 px-3 py-2 text-xs text-slate-300">
      Selected{' '}
      <span className="font-mono text-slate-100">{state.filename}</span>{' '}
      ({formatBytes(state.size)}) for server-side inspection.
    </div>
  )
}

function formatBytes(value: number): string {
  if (!Number.isFinite(value) || value < 0) return 'unknown size'
  if (value < 1024) return `${value} B`
  const kib = value / 1024
  if (kib < 1024) return `${kib.toFixed(1)} KB`
  return `${(kib / 1024).toFixed(1)} MB`
}

function IngestionImportResult({ state }: { state: IngestionImportState }) {
  if (state.kind === 'idle' || state.kind === 'submitting') {
    return null
  }
  if (state.kind === 'invalid_input') {
    return (
      <div className="mt-3 rounded-md border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-xs text-rose-200">
        Invalid ingestion payload: {state.message}
      </div>
    )
  }
  if (state.kind === 'error') {
    return (
      <div className="mt-3 rounded-md border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-xs text-rose-200">
        Import failed: {state.message}
      </div>
    )
  }
  if (state.kind === 'not_ready') {
    return (
      <div className="mt-3 rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-xs text-amber-100">
        <div className="mb-2 font-medium">
          Import blocked: rows need attention before they can be written.
        </div>
        <div className="mb-2 flex flex-wrap gap-3 text-slate-300">
          <span>Opportunities: {state.diagnostics.opportunityCount}</span>
          <span>Warnings: {state.diagnostics.warningCount}</span>
        </div>
        {state.diagnostics.warnings.length > 0 && (
          <Section label="Blocking diagnostics">
            <ul className="ml-4 list-disc text-xs text-amber-100">
              {state.diagnostics.warnings
                .slice(0, INGESTION_SAMPLE_LIMIT)
                .map((warning, i) => (
                  <li key={`${warning.code}-${warning.rowIndex ?? i}-${warning.field ?? ''}`}>
                    {warning.code}: {warning.message}
                  </li>
                ))}
            </ul>
          </Section>
        )}
      </div>
    )
  }

  const result = state.response.importResult
  return (
    <div className="mt-3 rounded-md border border-emerald-500/30 bg-emerald-500/10 p-3 text-xs">
      <div className="mb-2 flex flex-wrap items-center gap-3">
        <span className="rounded-full bg-emerald-500/20 px-2.5 py-0.5 font-medium text-emerald-200">
          {result.dryRun ? 'Dry run complete' : 'Import complete'}
        </span>
        <span className="text-slate-300">Inserted: {result.inserted}</span>
        <span className="text-slate-300">Skipped: {result.skipped}</span>
        {result.replaceExisting && (
          <span className="text-amber-200">Replace existing enabled</span>
        )}
      </div>
      {result.targetIds.length > 0 && (
        <Section label="Target ids">
          <div className="break-all font-mono text-[11px] text-slate-300">
            {result.targetIds.slice(0, INGESTION_SAMPLE_LIMIT).join(', ')}
          </div>
        </Section>
      )}
      {result.warnings.length > 0 && (
        <Section label="Import warnings">
          <ul className="ml-4 list-disc text-xs text-amber-100">
            {result.warnings.slice(0, INGESTION_SAMPLE_LIMIT).map((warning, i) => (
              <li key={`${warning.code}-${warning.rowIndex ?? i}-${warning.field ?? ''}`}>
                {warning.code}: {warning.message}
              </li>
            ))}
          </ul>
        </Section>
      )}
    </div>
  )
}

function importOutcomeMessage(
  outcome: Exclude<
    Awaited<ReturnType<typeof importContentOpsIngestion>>,
    { kind: 'success' } | { kind: 'not_ready' }
  >,
): string {
  if (outcome.kind === 'validation_error') {
    return JSON.stringify(outcome.detail)
  }
  return outcome.detail
}

type ParsedIngestionRows =
  | { ok: true; rows: Array<Record<string, unknown>> }
  | { ok: false; message: string }

type ParsedIngestionDefaultFields =
  | { ok: true; fields: Record<string, unknown> }
  | { ok: false; message: string }

type ParsedInputsJsonObject =
  | { ok: true; value: Record<string, unknown> }
  | { ok: false; message: string }

type UpdatedInputsJson =
  | { ok: true; value: string }
  | { ok: false; message: string }

function parseInputsJsonObject(value: string): ParsedInputsJsonObject {
  let parsed: unknown
  try {
    parsed = JSON.parse(value.trim() || '{}')
  } catch (err) {
    return {
      ok: false,
      message: err instanceof Error ? err.message : String(err),
    }
  }
  if (!isRecord(parsed)) {
    return { ok: false, message: 'Inputs JSON must be an object.' }
  }
  return { ok: true, value: { ...parsed } }
}

function landingPageSeoGeoAeoInputContracts(
  catalog: ContentOpsCatalog,
): ContentOpsInputContractView[] {
  return Object.values(catalog.inputContracts)
    .filter(
      (contract) =>
        contract.asset === LANDING_PAGE_INPUT_ASSET &&
        contract.group === LANDING_PAGE_SEO_GEO_AEO_INPUT_GROUP,
    )
    .sort(
      (left, right) =>
        landingPageInputSortIndex(left.key) -
        landingPageInputSortIndex(right.key),
    )
}

function landingPageInputSortIndex(key: string): number {
  const index = LANDING_PAGE_SEO_GEO_AEO_INPUT_ORDER.indexOf(key)
  return index === -1 ? LANDING_PAGE_SEO_GEO_AEO_INPUT_ORDER.length : index
}

function landingPageInputDraftValue(
  parsed: ParsedInputsJsonObject,
  contract: ContentOpsInputContractView,
): string {
  if (!parsed.ok) return ''

  const raw = parsed.value[contract.key]
  if (raw === null || typeof raw === 'undefined') return ''
  if (contract.type === 'string_list') return stringListDraftValue(raw)
  return scalarDraftValue(raw)
}

function updateLandingPageInputJson(
  current: string,
  contract: ContentOpsInputContractView,
  draftValue: string,
): UpdatedInputsJson {
  const parsed = parseInputsJsonObject(current)
  if (!parsed.ok) return parsed

  const next = { ...parsed.value }
  if (contract.type === 'string_list') {
    const values = stringListFromDraft(draftValue)
    if (values.length === 0) {
      delete next[contract.key]
    } else {
      next[contract.key] = values
    }
  } else {
    const value = draftValue.trim()
    if (value === '') {
      delete next[contract.key]
    } else {
      next[contract.key] = value
    }
  }

  return { ok: true, value: `${JSON.stringify(next, null, 2)}\n` }
}

function faqDocumentationTermsDraftValue(parsed: ParsedInputsJsonObject): string {
  if (!parsed.ok) return ''
  const raw = parsed.value[FAQ_DOCUMENTATION_TERMS_INPUT]
  if (raw === null || typeof raw === 'undefined') return ''
  return stringListDraftValue(raw)
}

function updateFaqDocumentationTermsInputJson(
  current: string,
  draftValue: string,
): UpdatedInputsJson {
  const parsed = parseInputsJsonObject(current)
  if (!parsed.ok) return parsed

  const next = { ...parsed.value }
  const values = stringListFromDraft(draftValue)
  if (values.length === 0) {
    delete next[FAQ_DOCUMENTATION_TERMS_INPUT]
  } else {
    next[FAQ_DOCUMENTATION_TERMS_INPUT] = values
  }

  return { ok: true, value: `${JSON.stringify(next, null, 2)}\n` }
}

function faqVocabularyRulesDraftValue(parsed: ParsedInputsJsonObject): string {
  if (!parsed.ok) return ''
  const raw = parsed.value[FAQ_VOCABULARY_GAP_RULES_INPUT]
  if (raw === null || typeof raw === 'undefined') return ''
  if (!Array.isArray(raw)) return scalarDraftValue(raw)
  return raw
    .filter((rule) => typeof rule !== 'undefined' && rule !== null)
    .map((rule) =>
      Array.isArray(rule)
        ? rule
            .filter((term) => typeof term !== 'undefined' && term !== null)
            .map((term) => String(term))
            .join(', ')
        : String(rule),
    )
    .join('\n')
}

function updateFaqVocabularyRulesInputJson(
  current: string,
  draftValue: string,
): UpdatedInputsJson {
  const parsed = parseInputsJsonObject(current)
  if (!parsed.ok) return parsed

  const next = { ...parsed.value }
  const rules = vocabularyRulesFromDraft(draftValue)
  if (rules.length === 0) {
    delete next[FAQ_VOCABULARY_GAP_RULES_INPUT]
  } else {
    next[FAQ_VOCABULARY_GAP_RULES_INPUT] = rules
  }

  return { ok: true, value: `${JSON.stringify(next, null, 2)}\n` }
}

function vocabularyRulesFromDraft(value: string): string[][] {
  const rules: string[][] = []
  for (const line of value.split(/\r?\n/)) {
    const terms = stringListFromDraft(line)
    if (terms.length > 0) {
      rules.push(terms)
    }
  }
  return rules
}

function stringListDraftValue(value: unknown): string {
  if (Array.isArray(value)) {
    return value
      .filter((item) => typeof item !== 'undefined' && item !== null)
      .map((item) => String(item))
      .join('\n')
  }
  return scalarDraftValue(value)
}

function stringListFromDraft(value: string): string[] {
  const seen = new Set<string>()
  const items: string[] = []
  for (const item of value.split(/[\n,]+/)) {
    const trimmed = item.trim()
    if (trimmed === '' || seen.has(trimmed)) continue
    seen.add(trimmed)
    items.push(trimmed)
  }
  return items
}

function scalarDraftValue(value: unknown): string {
  if (typeof value === 'string') return value
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  return JSON.stringify(value, null, 2)
}

function landingPageRepairAttemptSelectValue(
  parsed: ParsedInputsJsonObject,
  contract: IntegerInputContract | null,
): string {
  if (!parsed.ok || !contract) return ''

  const raw = parsed.value[LANDING_PAGE_QUALITY_REPAIR_INPUT]
  if (raw === null || typeof raw === 'undefined') return ''

  const normalized = normalizeLandingPageRepairAttemptValue(raw, contract)
  return normalized.ok
    ? String(normalized.value)
    : INVALID_LANDING_PAGE_QUALITY_REPAIR_VALUE
}

function landingPageRepairAttemptHelpText(
  parsed: ParsedInputsJsonObject,
  contract: IntegerInputContract | null,
): string {
  if (!parsed.ok) {
    return `Fix inputs JSON before changing repair attempts: ${parsed.message}`
  }
  if (!contract) {
    return `${LANDING_PAGE_QUALITY_REPAIR_INPUT} contract is missing from the control-surface catalog.`
  }

  const raw = parsed.value[LANDING_PAGE_QUALITY_REPAIR_INPUT]
  if (raw === null || typeof raw === 'undefined') {
    return `Uses the backend default: ${contract.default} repair attempt${contract.default === 1 ? '' : 's'}.`
  }

  const normalized = normalizeLandingPageRepairAttemptValue(raw, contract)
  if (!normalized.ok) {
    return landingPageRepairAttemptErrorMessage(contract)
  }

  if (normalized.value === 0) {
    return 'Quality repair is disabled for landing-page generation.'
  }
  return `Landing-page generation can make ${normalized.value} quality repair attempt${normalized.value === 1 ? '' : 's'}.`
}

function updateLandingPageRepairAttemptsInputJson(
  current: string,
  selected: string,
  contract: IntegerInputContract | null,
): UpdatedInputsJson {
  const parsed = parseInputsJsonObject(current)
  if (!parsed.ok) return parsed
  if (!contract) {
    return {
      ok: false,
      message: `${LANDING_PAGE_QUALITY_REPAIR_INPUT} contract is missing from the control-surface catalog.`,
    }
  }

  const next = { ...parsed.value }
  if (selected === '') {
    delete next[LANDING_PAGE_QUALITY_REPAIR_INPUT]
  } else {
    const normalized = normalizeLandingPageRepairAttemptValue(selected, contract)
    if (!normalized.ok) {
      return {
        ok: false,
        message: landingPageRepairAttemptErrorMessage(contract),
      }
    }
    next[LANDING_PAGE_QUALITY_REPAIR_INPUT] = normalized.value
  }

  return { ok: true, value: `${JSON.stringify(next, null, 2)}\n` }
}

function normalizeLandingPageRepairAttemptValue(
  value: unknown,
  contract: IntegerInputContract,
): { ok: true; value: number } | { ok: false } {
  if (
    typeof value === 'boolean' ||
    (typeof value === 'number' && !Number.isInteger(value))
  ) {
    return { ok: false }
  }

  let normalized: number
  if (typeof value === 'number') {
    normalized = value
  } else if (typeof value === 'string') {
    const trimmed = value.trim()
    if (!/^\d+$/.test(trimmed)) return { ok: false }
    normalized = Number(trimmed)
  } else {
    return { ok: false }
  }

  if (
    !Number.isSafeInteger(normalized) ||
    normalized < contract.min ||
    normalized > contract.max
  ) {
    return { ok: false }
  }
  return { ok: true, value: normalized }
}

type IntegerInputContract = ContentOpsInputContractView & {
  min: number
  max: number
  default: number
}

function integerInputContract(
  catalog: ContentOpsCatalog,
  key: string,
): IntegerInputContract | null {
  const contract = catalog.inputContracts[key]
  if (
    !contract ||
    contract.type !== 'integer' ||
    typeof contract.min !== 'number' ||
    typeof contract.max !== 'number' ||
    typeof contract.default !== 'number' ||
    !Number.isSafeInteger(contract.min) ||
    !Number.isSafeInteger(contract.max) ||
    !Number.isSafeInteger(contract.default) ||
    contract.min > contract.max ||
    contract.default < contract.min ||
    contract.default > contract.max
  ) {
    if (key === LANDING_PAGE_QUALITY_REPAIR_INPUT) {
      const landingPage = catalog.outputs.find((output) => output.id === 'landing_page')
      return {
        ...LEGACY_LANDING_PAGE_REPAIR_ATTEMPT_CONTRACT,
        default: landingPage?.defaultQualityRepairAttempts ??
          LEGACY_LANDING_PAGE_REPAIR_ATTEMPT_CONTRACT.default,
      }
    }
    return null
  }
  return {
    ...contract,
    min: contract.min,
    max: contract.max,
    default: contract.default,
  }
}

function integerInputOptions(contract: IntegerInputContract): string[] {
  return Array.from(
    { length: contract.max - contract.min + 1 },
    (_, index) => String(contract.min + index),
  )
}

function landingPageRepairAttemptErrorMessage(
  contract: IntegerInputContract,
): string {
  return `${LANDING_PAGE_QUALITY_REPAIR_INPUT} must be an integer from ${contract.min} to ${contract.max}.`
}

function parseIngestionRowsJson(value: string): ParsedIngestionRows {
  let parsed: unknown
  try {
    parsed = JSON.parse(value.trim() || '[]')
  } catch (err) {
    return {
      ok: false,
      message: err instanceof Error ? err.message : String(err),
    }
  }

  const rows = extractIngestionRows(parsed)
  if (!rows.ok) return rows
  if (rows.rows.length === 0) {
    return { ok: false, message: 'Provide at least one row to inspect.' }
  }
  return rows
}

function parseIngestionDefaultFieldsJson(
  value: string,
): ParsedIngestionDefaultFields {
  let parsed: unknown
  try {
    parsed = JSON.parse(value.trim() || '{}')
  } catch (err) {
    return {
      ok: false,
      message: err instanceof Error ? err.message : String(err),
    }
  }
  if (!isRecord(parsed)) {
    return { ok: false, message: 'Fallback fields must be a JSON object.' }
  }
  return { ok: true, fields: { ...parsed } }
}

function extractIngestionRows(value: unknown): ParsedIngestionRows {
  if (Array.isArray(value)) {
    return normalizeIngestionRows(value)
  }
  if (isRecord(value)) {
    for (const key of ['rows', 'opportunities', 'source_rows']) {
      const nested = value[key]
      if (Array.isArray(nested)) {
        return normalizeIngestionRows(nested)
      }
    }
    return { ok: true, rows: [{ ...value }] }
  }
  return { ok: false, message: 'Expected a row object or an array of row objects.' }
}

function normalizeIngestionRows(value: unknown[]): ParsedIngestionRows {
  const rows: Array<Record<string, unknown>> = []
  for (const [index, row] of value.entries()) {
    if (!isRecord(row)) {
      return { ok: false, message: `Row ${index + 1} must be a JSON object.` }
    }
    rows.push({ ...row })
  }
  return { ok: true, rows }
}

function PlanPanel({
  plan,
  executionConfigured,
  configuredOutputs,
  executionState,
  onExecute,
}: {
  plan: GenerationPlan
  executionConfigured: boolean
  configuredOutputs: string[]
  executionState: ExecutionState
  onExecute: () => void
}) {
  const configuredOutputSet = new Set(configuredOutputs)
  const planOutputsConfigured = plan.steps.every((step) =>
    configuredOutputSet.has(step.output),
  )
  const canExecute = plan.canExecute && executionConfigured && planOutputsConfigured
  const executing = executionState.kind === 'submitting'
  return (
    <section className="mt-6 rounded-lg border border-slate-800 bg-slate-900/60 p-5">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <span
            className={clsx(
              'rounded-full px-3 py-0.5 text-xs font-medium',
              plan.canExecute
                ? 'bg-emerald-500/20 text-emerald-300'
                : 'bg-amber-500/20 text-amber-200',
            )}
          >
            Plan: {plan.canExecute ? 'Can execute' : 'Blocked'}
          </span>
          <span className="text-sm text-slate-400">
            {plan.steps.length} step{plan.steps.length === 1 ? '' : 's'} ·
            target_mode={plan.targetMode} · limit={plan.limit}
          </span>
        </div>
        <button
          type="button"
          onClick={onExecute}
          disabled={!canExecute || executing}
          title={
            !plan.canExecute
              ? 'Plan is blocked; cannot execute.'
              : !executionConfigured
                ? 'Host execution services not configured.'
                : !planOutputsConfigured
                  ? 'One or more planned outputs are missing host execution services.'
                  : 'Execute this plan through host services.'
          }
          className={clsx(
            'flex items-center gap-2 rounded-md border px-3 py-1.5 text-xs font-medium disabled:cursor-not-allowed disabled:opacity-50',
            canExecute
              ? 'border-cyan-500/40 bg-cyan-500/10 text-cyan-200 hover:bg-cyan-500/20'
              : 'border-slate-700 bg-slate-800/40 text-slate-400',
          )}
        >
          {executing ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <Play className="h-3.5 w-3.5" />
          )}
          Execute {canExecute ? '' : '(disabled)'}
        </button>
      </div>

      <div className="space-y-3">
        {plan.steps.map((step) => (
          <PlanStepCard key={step.output} step={step} />
        ))}
      </div>
    </section>
  )
}

function PlanStepCard({ step }: { step: GenerationPlanStep }) {
  return (
    <div className="rounded-md border border-slate-800 bg-slate-950/40 p-3">
      <div className="mb-2 flex items-center gap-3">
        <span className="font-mono text-sm text-slate-100">{step.output}</span>
        <span
          className={clsx(
            'rounded px-2 py-0.5 text-[10px] uppercase tracking-wide',
            step.status === 'runnable'
              ? 'bg-emerald-500/10 text-emerald-300'
              : 'bg-amber-500/10 text-amber-200',
          )}
        >
          {step.status}
        </span>
        <span className="text-xs text-slate-500">{step.runner}</span>
      </div>
      {step.status === 'blocked' && step.reason && (
        <div className="mb-2 rounded bg-amber-500/5 px-2 py-1 text-xs text-amber-200">
          Blocked: {step.reason}
        </div>
      )}
      {Object.keys(step.config).length > 0 && (
        <details className="text-xs text-slate-400">
          <summary className="cursor-pointer text-slate-500 hover:text-slate-300">
            Config ({Object.keys(step.config).length} fields)
          </summary>
          <pre className="mt-2 overflow-x-auto rounded bg-slate-950/80 p-2 font-mono text-[11px] text-slate-300">
            {JSON.stringify(step.config, null, 2)}
          </pre>
        </details>
      )}
    </div>
  )
}

function ExecutionPanel({ result }: { result: ContentOpsExecutionResult }) {
  const tone =
    result.status === 'completed'
      ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300'
      : result.status === 'partial' || result.status === 'blocked'
        ? 'border-amber-500/40 bg-amber-500/10 text-amber-200'
        : 'border-rose-500/40 bg-rose-500/10 text-rose-200'
  return (
    <section className="mt-6 rounded-lg border border-slate-800 bg-slate-900/60 p-5">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <span className={clsx('rounded-full border px-3 py-0.5 text-xs font-medium', tone)}>
            Execution: {result.status}
          </span>
          <span className="text-sm text-slate-400">
            {result.steps.length} step{result.steps.length === 1 ? '' : 's'}
          </span>
        </div>
      </div>

      <div className="space-y-3">
        {result.steps.map((step) => (
          <div
            key={`${step.output}-${step.runner}`}
            className="rounded-md border border-slate-800 bg-slate-950/40 p-3"
          >
            <div className="mb-2 flex items-center gap-3">
              <span className="font-mono text-sm text-slate-100">{step.output}</span>
              <span
                className={clsx(
                  'rounded px-2 py-0.5 text-[10px] uppercase tracking-wide',
                  step.status === 'completed'
                    ? 'bg-emerald-500/10 text-emerald-300'
                    : step.status === 'skipped'
                      ? 'bg-slate-700 text-slate-400'
                      : 'bg-rose-500/10 text-rose-200',
                )}
              >
                {step.status}
              </span>
              <span className="text-xs text-slate-500">{step.runner}</span>
            </div>
            {step.error && (
              <div className="mb-2 rounded bg-rose-500/5 px-2 py-1 text-xs text-rose-200">
                Error: {step.error}
              </div>
            )}
            {step.reasoning && <ReasoningAuditBadge audit={step.reasoning} />}
            <ExecutionStepSummary output={step.output} result={step.result} />
            {Object.keys(step.result).length > 0 && (
              <details className="text-xs text-slate-400">
                <summary className="cursor-pointer text-slate-500 hover:text-slate-300">
                  Result ({Object.keys(step.result).length} fields)
                </summary>
                <pre className="mt-2 overflow-x-auto rounded bg-slate-950/80 p-2 font-mono text-[11px] text-slate-300">
                  {JSON.stringify(step.result, null, 2)}
                </pre>
              </details>
            )}
          </div>
        ))}
      </div>

      {result.errors.length > 0 && (
        <Section label="Execution errors">
          <pre className="overflow-x-auto rounded-md bg-slate-950/80 p-3 font-mono text-xs text-rose-100">
            {JSON.stringify(result.errors, null, 2)}
          </pre>
        </Section>
      )}
    </section>
  )
}

function ExecutionStepSummary({
  output,
  result,
}: {
  output: string
  result: Record<string, unknown>
}) {
  if (output === 'signal_extraction') {
    return <SignalExtractionSummary result={result} />
  }
  if (output === 'email_campaign') {
    return <GeneratedAssetSummary result={result} generatedLabel="Drafts generated" />
  }
  if (['blog_post', 'report', 'landing_page', 'sales_brief'].includes(output)) {
    return <GeneratedAssetSummary result={result} generatedLabel="Assets generated" />
  }
  if (output === 'faq_markdown') {
    return <FAQMarkdownExecutionSummary result={result} />
  }
  return null
}

function FAQMarkdownExecutionSummary({ result }: { result: Record<string, unknown> }) {
  const generated = typeof result.generated === 'number' ? result.generated : null
  const sourceCount =
    typeof result.source_count === 'number' ? result.source_count : null
  const ticketSourceCount =
    typeof result.ticket_source_count === 'number'
      ? result.ticket_source_count
      : null
  const warningCount = Array.isArray(result.warnings)
    ? result.warnings.length
    : null
  const savedIds = Array.isArray(result.saved_ids)
    ? result.saved_ids.filter((id): id is string => typeof id === 'string')
    : []
  const mappings = faqExecutionTermMappings(result.items)
  const shownMappings = mappings.slice(0, 3)

  if (
    generated === null &&
    sourceCount === null &&
    ticketSourceCount === null &&
    warningCount === null &&
    savedIds.length === 0 &&
    mappings.length === 0
  ) {
    return null
  }

  return (
    <div className="mb-3 space-y-2 rounded-md border border-slate-800 bg-slate-900/70 px-3 py-2 text-xs text-slate-300">
      <div className="flex flex-wrap items-center gap-3">
        {generated !== null && (
          <span>
            FAQ items:{' '}
            <span className="font-medium text-slate-100">{generated}</span>
          </span>
        )}
        {sourceCount !== null && (
          <span>
            Source rows:{' '}
            <span className="font-medium text-slate-100">{sourceCount}</span>
          </span>
        )}
        {ticketSourceCount !== null && (
          <span>
            Ticket sources:{' '}
            <span className="font-medium text-slate-100">
              {ticketSourceCount}
            </span>
          </span>
        )}
        {warningCount !== null && (
          <span>
            Warnings:{' '}
            <span className="font-medium text-slate-100">{warningCount}</span>
          </span>
        )}
        <span>
          Vocabulary gaps:{' '}
          <span className="font-medium text-slate-100">{mappings.length}</span>
        </span>
      </div>
      {shownMappings.length > 0 && (
        <ul className="space-y-1">
          {shownMappings.map((mapping, index) => (
            <li
              key={`${mapping.customerTerm}-${mapping.documentationTerm}-${index}`}
              className="rounded border border-slate-800 bg-slate-950/50 px-2 py-1"
            >
              <span className="font-medium text-slate-100">
                {mapping.customerTerm || 'Customer term'}
              </span>
              {' -> '}
              <span className="text-cyan-200">
                {mapping.documentationTerm || 'Documentation term'}
              </span>
              <span className="ml-2 text-slate-500">
                {faqMappingMeta(mapping).join(' · ')}
              </span>
            </li>
          ))}
        </ul>
      )}
      {mappings.length > shownMappings.length && (
        <div className="text-slate-500">
          +{mappings.length - shownMappings.length} more vocabulary gaps in raw
          result details
        </div>
      )}
      {savedIds.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5">
          <span>Saved:</span>
          {savedIds.map((id) => (
            <span
              key={id}
              className="max-w-full break-all rounded bg-slate-950/60 px-1.5 py-0.5 font-mono text-slate-100"
            >
              {id}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

type FAQExecutionTermMapping = {
  customerTerm: string
  documentationTerm: string
  sourceIdCount: number | null
  zeroResultSourceCount: number | null
  opportunityScore: number | null
}

function faqExecutionTermMappings(value: unknown): FAQExecutionTermMapping[] {
  return recordArray(value).flatMap((item) =>
    recordArray(item.term_mappings)
      .map((mapping) => ({
        customerTerm: stringField(mapping, 'customer_term') ?? '',
        documentationTerm: stringField(mapping, 'documentation_term') ?? '',
        sourceIdCount: numberField(mapping, 'source_id_count'),
        zeroResultSourceCount: numberField(mapping, 'zero_result_source_count'),
        opportunityScore: numberField(mapping, 'opportunity_score'),
      }))
      .filter((mapping) => mapping.customerTerm || mapping.documentationTerm),
  )
}

function recordArray(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value) ? value.filter(isRecord) : []
}

function faqMappingMeta(mapping: FAQExecutionTermMapping): string[] {
  return [
    numberMeta(mapping.sourceIdCount, 'source'),
    numberMeta(mapping.zeroResultSourceCount, 'zero-result'),
    mapping.opportunityScore === null
      ? ''
      : `score ${mapping.opportunityScore}`,
  ].filter(Boolean)
}

function numberMeta(value: number | null, label: string): string {
  if (value === null) return ''
  return `${value} ${label}${value === 1 ? '' : 's'}`
}

function GeneratedAssetSummary({
  result,
  generatedLabel,
}: {
  result: Record<string, unknown>
  generatedLabel: string
}) {
  const requested = typeof result.requested === 'number' ? result.requested : null
  const generated = typeof result.generated === 'number' ? result.generated : null
  const skipped = typeof result.skipped === 'number' ? result.skipped : null
  const savedIds = Array.isArray(result.saved_ids)
    ? result.saved_ids.filter((id): id is string => typeof id === 'string')
    : []
  const errorCount = Array.isArray(result.errors) ? result.errors.length : null
  const reasoningContextsUsed =
    typeof result.reasoning_contexts_used === 'number'
      ? result.reasoning_contexts_used
      : null

  if (
    requested === null &&
    generated === null &&
    skipped === null &&
    savedIds.length === 0 &&
    reasoningContextsUsed === null &&
    errorCount === null
  ) {
    return null
  }

  return (
    <div className="mb-3 space-y-2 rounded-md border border-slate-800 bg-slate-900/70 px-3 py-2 text-xs text-slate-300">
      <div className="flex flex-wrap items-center gap-3">
        {requested !== null && (
          <span>
            Requested:{' '}
            <span className="font-medium text-slate-100">{requested}</span>
          </span>
        )}
        {generated !== null && (
          <span>
            {generatedLabel}:{' '}
            <span className="font-medium text-slate-100">{generated}</span>
          </span>
        )}
        {skipped !== null && (
          <span>
            Skipped:{' '}
            <span className="font-medium text-slate-100">{skipped}</span>
          </span>
        )}
        {errorCount !== null && (
          <span>
            Errors:{' '}
            <span className="font-medium text-slate-100">{errorCount}</span>
          </span>
        )}
        {reasoningContextsUsed !== null && (
          <span>
            Reasoning used:{' '}
            <span className="font-medium text-slate-100">
              {reasoningContextsUsed}
            </span>
          </span>
        )}
      </div>
      {savedIds.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5">
          <span>Saved:</span>
          {savedIds.map((id) => (
            <span
              key={id}
              className="max-w-full break-all rounded bg-slate-950/60 px-1.5 py-0.5 font-mono text-slate-100"
            >
              {id}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

function ReasoningAuditBadge({ audit }: { audit: ContentOpsStepReasoningAudit }) {
  if (audit.requirement === 'absent' && !audit.serviceSupportsReasoning) {
    return null
  }
  const ready = audit.providerConfigured && audit.serviceSupportsReasoning
  const label = ready
    ? 'Reasoning provider attached'
    : audit.serviceSupportsReasoning
      ? 'Reasoning provider absent'
      : 'Reasoning seam unavailable'
  return (
    <div
      className={clsx(
        'mb-3 inline-flex max-w-full flex-wrap items-center gap-2 rounded-md border px-2 py-1 text-xs',
        ready
          ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-200'
          : 'border-slate-700 bg-slate-900/70 text-slate-400',
      )}
    >
      <span>{label}</span>
      <span className="font-mono text-[11px] opacity-80">
        {audit.requirement}
      </span>
      {typeof audit.contextsUsed === 'number' && (
        <span className="font-mono text-[11px] opacity-80">
          used {audit.contextsUsed}
        </span>
      )}
      {audit.consumedContexts && audit.consumedContexts.length > 0 && (
        <ReasoningContextList contexts={audit.consumedContexts} />
      )}
    </div>
  )
}

function ReasoningContextList({
  contexts,
}: {
  contexts: CampaignReasoningContextView[]
}) {
  return (
    <div className="basis-full space-y-1 pt-1">
      {contexts.slice(0, 3).map((context, index) => (
        <details
          key={index}
          className="group rounded border border-emerald-500/20 bg-slate-950/50 px-2 py-1 text-[11px] text-emerald-100"
        >
          <summary className="cursor-pointer list-none">
            <div className="flex items-start justify-between gap-3">
              <div className="flex min-w-0 gap-2">
                <ChevronRight className="mt-0.5 h-3 w-3 shrink-0 text-emerald-200/60 transition-transform group-open:rotate-90" />
                <div>
                  <div className="font-medium">
                    {context.summary || `Reasoning context ${index + 1}`}
                  </div>
                  <div className="mt-0.5 text-emerald-200/70">
                    {reasoningContextCounts(context).join(' · ')}
                  </div>
                </div>
              </div>
              <span className="shrink-0 text-[10px] uppercase tracking-wide text-emerald-200/60 group-open:hidden">
                Details
              </span>
              <span className="hidden shrink-0 text-[10px] uppercase tracking-wide text-emerald-200/60 group-open:inline">
                Hide
              </span>
            </div>
          </summary>
          <ReasoningContextDetails context={context} />
        </details>
      ))}
      {contexts.length > 3 && (
        <div className="text-[11px] text-emerald-200/70">
          +{contexts.length - 3} more contexts
        </div>
      )}
    </div>
  )
}

function ReasoningContextDetails({
  context,
}: {
  context: CampaignReasoningContextView
}) {
  const hasDetails =
    hasItems(context.topTheses) ||
    hasItems(context.proofPoints) ||
    hasAnchorExamples(context.anchorExamples) ||
    hasItems(context.witnessHighlights) ||
    hasItems(context.accountSignals) ||
    hasItems(context.timingWindows) ||
    hasStringItems(context.coverageLimits) ||
    hasObjectItems(context.referenceIds) ||
    hasObjectItems(context.scopeSummary) ||
    hasObjectItems(context.deltaSummary) ||
    hasObjectItems(context.extra)

  if (!hasDetails) {
    return (
      <div className="mt-2 border-t border-emerald-500/10 pt-2 text-emerald-200/70">
        No structured reasoning detail returned.
      </div>
    )
  }

  return (
    <div className="mt-2 space-y-2 border-t border-emerald-500/10 pt-2">
      <ReasoningObjectList label="Top theses" rows={context.topTheses} />
      <ReasoningObjectList label="Proof points" rows={context.proofPoints} />
      <ReasoningAnchorExamples anchorExamples={context.anchorExamples} />
      <ReasoningObjectList label="Witness highlights" rows={context.witnessHighlights} />
      <ReasoningObjectList label="Account signals" rows={context.accountSignals} />
      <ReasoningObjectList label="Timing windows" rows={context.timingWindows} />
      <ReasoningReferenceIds referenceIds={context.referenceIds} />
      <ReasoningStringList label="Coverage limits" values={context.coverageLimits} />
      <ReasoningKeyValueBlock label="Scope" value={context.scopeSummary} />
      <ReasoningKeyValueBlock label="Delta" value={context.deltaSummary} />
      <ReasoningKeyValueBlock label="Other" value={context.extra} />
    </div>
  )
}

function reasoningContextCounts(context: CampaignReasoningContextView): string[] {
  const counts = [
    { label: 'theses', count: context.topTheses?.length ?? 0 },
    { label: 'proof points', count: context.proofPoints?.length ?? 0 },
    { label: 'anchors', count: anchorExampleCount(context.anchorExamples) },
    { label: 'witnesses', count: context.witnessHighlights?.length ?? 0 },
    { label: 'account signals', count: context.accountSignals?.length ?? 0 },
    { label: 'timing windows', count: context.timingWindows?.length ?? 0 },
  ].filter(({ count }) => count > 0)

  if (counts.length === 0) {
    return ['no structured detail']
  }
  return counts.map(({ label, count }) => `${label} ${count}`)
}

function anchorExampleCount(
  value?: Record<string, Array<Record<string, unknown>>>,
): number {
  if (!value) return 0
  return Object.values(value).reduce((total, rows) => total + rows.length, 0)
}

function ReasoningAnchorExamples({
  anchorExamples,
}: {
  anchorExamples?: Record<string, Array<Record<string, unknown>>>
}) {
  if (!hasAnchorExamples(anchorExamples)) return null
  return (
    <div>
      <div className="mb-1 font-medium text-emerald-100">Anchor examples</div>
      <div className="space-y-2">
        {Object.entries(anchorExamples)
          .filter(([, rows]) => rows.length > 0)
          .map(([label, rows]) => (
            <div key={label} className="space-y-1">
              <ReasoningObjectList label={labelFromKey(label)} rows={rows} />
            </div>
          ))}
      </div>
    </div>
  )
}

function ReasoningObjectList({
  label,
  rows,
}: {
  label: string
  rows?: Array<Record<string, unknown>>
}) {
  if (!hasItems(rows)) return null
  return (
    <div>
      <div className="mb-1 font-medium text-emerald-100">{label}</div>
      <div className="space-y-1">
        {rows.map((row, index) => (
          <div
            key={index}
            className="rounded border border-slate-800/80 bg-slate-950/70 px-2 py-1 text-emerald-50/90"
          >
            {objectEntries(row).map(([key, value]) => (
              <div key={key} className="grid gap-1 sm:grid-cols-[120px_1fr]">
                <span className="text-emerald-200/60">{labelFromKey(key)}</span>
                <span className="break-words text-emerald-50/90">
                  {formatReasoningValue(value)}
                </span>
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  )
}

function ReasoningReferenceIds({
  referenceIds,
}: {
  referenceIds?: Record<string, string[]>
}) {
  if (!referenceIds || Object.keys(referenceIds).length === 0) return null
  return (
    <div>
      <div className="mb-1 font-medium text-emerald-100">Reference ids</div>
      <div className="space-y-1">
        {Object.entries(referenceIds).map(([key, values]) => (
          <div key={key} className="grid gap-1 sm:grid-cols-[120px_1fr]">
            <span className="text-emerald-200/60">{labelFromKey(key)}</span>
            <span className="break-words font-mono text-[10px] text-emerald-50/90">
              {Array.isArray(values) ? values.join(', ') : ''}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

function ReasoningStringList({
  label,
  values,
}: {
  label: string
  values?: string[]
}) {
  if (!hasStringItems(values)) return null
  return (
    <div>
      <div className="mb-1 font-medium text-emerald-100">{label}</div>
      <ul className="space-y-1">
        {values.map((value, index) => (
          <li
            key={`${value}-${index}`}
            className="rounded border border-slate-800/80 bg-slate-950/70 px-2 py-1 text-emerald-50/90"
          >
            {value}
          </li>
        ))}
      </ul>
    </div>
  )
}

function ReasoningKeyValueBlock({
  label,
  value,
}: {
  label: string
  value?: Record<string, unknown>
}) {
  if (!value || Object.keys(value).length === 0) return null
  return (
    <div>
      <div className="mb-1 font-medium text-emerald-100">{label}</div>
      <div className="rounded border border-slate-800/80 bg-slate-950/70 px-2 py-1">
        {objectEntries(value).map(([key, item]) => (
          <div key={key} className="grid gap-1 sm:grid-cols-[120px_1fr]">
            <span className="text-emerald-200/60">{labelFromKey(key)}</span>
            <span className="break-words text-emerald-50/90">
              {formatReasoningValue(item)}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

function hasItems(rows?: Array<Record<string, unknown>>): rows is Array<Record<string, unknown>> {
  return Array.isArray(rows) && rows.length > 0
}

function hasStringItems(values?: string[]): values is string[] {
  return Array.isArray(values) && values.length > 0
}

function hasAnchorExamples(
  value?: Record<string, Array<Record<string, unknown>>>,
): value is Record<string, Array<Record<string, unknown>>> {
  return Boolean(value && Object.values(value).some((rows) => rows.length > 0))
}

function hasObjectItems(value?: Record<string, unknown> | Record<string, string[]>): boolean {
  return Boolean(value && Object.keys(value).length > 0)
}

function objectEntries(value: Record<string, unknown>): Array<[string, unknown]> {
  return Object.entries(value).filter(([, item]) => item !== null && item !== undefined)
}

function labelFromKey(key: string): string {
  return key
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase())
}

function formatReasoningValue(value: unknown): string {
  if (value === null || value === undefined) return ''
  if (typeof value === 'string') return value
  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value)
  }
  if (Array.isArray(value)) {
    return value.map(formatReasoningValue).filter(Boolean).join(', ')
  }
  if (typeof value === 'object') {
    return JSON.stringify(value)
  }
  return String(value)
}

function reasoningStatusHint(
  reasoning: ContentOpsCatalog['reasoning'],
): string {
  const values = [
    ...(reasoning.modes ?? []),
    ...(reasoning.packs ?? []),
    ...reasoningCapabilityValues(reasoning.capabilities),
  ]
    .map((value) => String(value).trim())
    .filter(Boolean)

  if (values.length === 0) return ''
  const shown = values.slice(0, 3)
  const suffix = values.length > shown.length ? ` +${values.length - shown.length} more` : ''
  const hint = `${shown.join(', ')}${suffix}`
  return hint.length > 42 ? `(${values.length} capabilities)` : `(${hint})`
}

function reasoningCapabilityValues(
  capabilities: ContentOpsCatalog['reasoning']['capabilities'],
): string[] {
  if (!capabilities) return []
  if (Array.isArray(capabilities)) {
    return capabilities.map((value) => String(value))
  }
  return Object.entries(capabilities)
    .map(([name, status]) => {
      const label = name.replace(/_/g, ' ')
      if (status.active && status.ready) return `${label} active`
      if (status.ready) return `${label} ready`
      if (status.configured) {
        const missing = status.missing?.filter(Boolean).join('/')
        return missing ? `${label} missing ${missing}` : `${label} configured`
      }
      return ''
    })
    .filter(Boolean)
}

function SignalExtractionSummary({ result }: { result: Record<string, unknown> }) {
  const generated = typeof result.generated === 'number' ? result.generated : null
  const targetMode =
    typeof result.target_mode === 'string' ? result.target_mode : null
  const warningCount = Array.isArray(result.warnings)
    ? result.warnings.length
    : null
  const opportunities = Array.isArray(result.opportunities)
    ? result.opportunities.filter(isRecord).slice(0, 3)
    : []

  if (
    generated === null &&
    targetMode === null &&
    warningCount === null &&
    opportunities.length === 0
  ) {
    return null
  }

  return (
    <div className="mb-3 space-y-2 rounded-md border border-slate-800 bg-slate-900/70 px-3 py-2 text-xs text-slate-300">
      <div className="flex flex-wrap items-center gap-3">
        {generated !== null && (
          <span>
            Opportunities:{' '}
            <span className="font-medium text-slate-100">{generated}</span>
          </span>
        )}
        {targetMode && (
          <span>
            Target mode:{' '}
            <span className="font-mono text-slate-100">{targetMode}</span>
          </span>
        )}
        {warningCount !== null && (
          <span>
            Warnings:{' '}
            <span className="font-medium text-slate-100">{warningCount}</span>
          </span>
        )}
      </div>
      {opportunities.length > 0 && (
        <ul className="space-y-1 text-slate-400">
          {opportunities.map((opportunity, index) => (
            <li key={opportunityKey(opportunity, index)}>
              {opportunityLabel(opportunity)}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function opportunityKey(
  opportunity: Record<string, unknown>,
  index: number,
): string {
  const key = opportunity.target_id ?? opportunity.source_id ?? opportunity.contact_email
  return typeof key === 'string' && key ? key : String(index)
}

function opportunityLabel(opportunity: Record<string, unknown>): string {
  const company = stringField(opportunity, 'company_name') ?? stringField(opportunity, 'company')
  const vendor = stringField(opportunity, 'vendor_name') ?? stringField(opportunity, 'vendor')
  const email = stringField(opportunity, 'contact_email')
  const id = stringField(opportunity, 'target_id') ?? stringField(opportunity, 'source_id')
  return [company, vendor, email, id].filter(Boolean).join(' | ') || 'Opportunity'
}

function stringField(
  value: Record<string, unknown>,
  key: string,
): string | null {
  const field = value[key]
  return typeof field === 'string' && field ? field : null
}

function numberField(
  value: Record<string, unknown>,
  key: string,
): number | null {
  const field = value[key]
  return typeof field === 'number' && Number.isFinite(field) ? field : null
}

function executionDetailMessage(detail: unknown): string {
  if (typeof detail === 'string') return detail
  try {
    return JSON.stringify(detail)
  } catch {
    return 'Execution request failed.'
  }
}
