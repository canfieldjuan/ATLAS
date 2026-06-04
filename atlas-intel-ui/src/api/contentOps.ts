/**
 * AI Content Ops API adapter.
 *
 * Typed fetch wrappers + wire-shape types for the
 * `/content-ops/*` and `/content-assets/*` routes the backend exposes:
 *
 *   GET  /content-ops/control-surfaces
 *   GET  /content-ops/brand-voice-profiles
 *   POST /content-ops/brand-voice-profiles
 *   POST /content-ops/brand-voice-profiles/sample-url
 *   PUT  /content-ops/brand-voice-profiles/{profile_id}
 *   DELETE /content-ops/brand-voice-profiles/{profile_id}
 *   GET  /content-ops/zendesk-credentials
 *   POST /content-ops/zendesk-credentials
 *   DELETE /content-ops/zendesk-credentials/{credential_id}
 *   POST /content-ops/preview
 *   POST /content-ops/plan
 *   POST /content-ops/ingestion/files/inspect
 *   POST /content-ops/ingestion/files/import
 *   POST /content-ops/ingestion/inspect (deprecated inline fallback)
 *   POST /content-ops/ingestion/import (deprecated inline fallback)
 *   POST /content-ops/execute
 *   GET  /content-assets/{asset}/drafts
 *   GET  /content-assets/{asset}/drafts/export
 *   PATCH /content-assets/landing_page/drafts/{id}
 *   POST /content-assets/landing_page/drafts/{id}/repair
 *   POST /content-assets/faq_markdown/drafts/{id}/publish-macros
 *   POST /content-assets/{asset}/drafts/review
 *   POST /content-assets/{asset}/drafts/review-batch
 *
 * Wire types are 1:1 with the backend JSON (snake_case), matching
 * the convention in `client.ts` / `b2bClient.ts`. camelCase
 * translation belongs at the domain layer (next slice;
 * `src/content/contentOps/`).
 *
 * Contract reference: `docs/frontend/content_ops_frontend_contract.md`
 * (PR #401, backend HEAD `a4020c1`).
 */

import { tryRefreshToken } from '../auth/AuthContext'
import { API_BASE } from './config'

// Mount under /api/v1/content-ops to match the Vite dev proxy
// (/api/*) and the existing backend mount convention used by
// `client.ts` and `b2bClient.ts` (`/api/v1/...`). The host
// application is expected to mount the content-ops router there
// via `ContentOpsControlSurfaceApiConfig(prefix="/api/v1/content-ops")`
// or equivalent.
const BASE = `${API_BASE}/api/v1/content-ops`
const ASSETS_BASE = `${API_BASE}/api/v1/content-assets`

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

// GET /content-ops/control-surfaces

export interface ContentOpsOutputDefinition {
  id: string
  label: string
  description: string
  implemented: boolean
  estimated_unit_cost_usd: number
  default_parse_retry_attempts: number
  default_quality_repair_attempts: number
  estimated_retry_adjusted_unit_cost_usd: number
  required_inputs: string[]
  default_max_items: number
  reasoning_requirement: 'absent' | 'optional_host_context' | string
  // Per-request, computed from host-injected services:
  execution_configured: boolean
  can_execute: boolean
}

export interface ContentOpsPreset {
  id: string
  label: string
  description: string
  outputs: string[]
}

export interface ContentOpsInputContract {
  key: string
  label: string
  type: 'integer' | string
  asset?: string
  group?: string
  placeholder?: string
  min?: number
  max?: number
  default?: string | number | boolean | null
}

export interface ContentOpsCatalogResponse {
  outputs: ContentOpsOutputDefinition[]
  presets: ContentOpsPreset[]
  execution: {
    configured: boolean
    configured_outputs: string[]
  }
  reasoning: {
    configured: boolean
    source?: 'db' | 'file' | 'none' | string
    modes?: Array<string | number | boolean>
    packs?: Array<string | number | boolean>
    capabilities?:
      | Array<string | number | boolean>
      | Record<string, ContentOpsReasoningCapabilityStatus>
  }
  ingestion_profiles: string[]
  ingestion_limits: ContentOpsIngestionLimits
  input_contracts: Record<string, ContentOpsInputContract>
}

export interface ContentOpsIngestionLimits {
  inline_rows: {
    max_rows: number
    deprecated: boolean
  }
  file_upload: {
    max_file_bytes: number
    max_rows: number
    supported_formats: string[]
  }
  max_source_text_chars: number
  max_sample_limit: number
}

export interface ContentOpsReasoningCapabilityStatus {
  configured?: boolean
  ready?: boolean
  active?: boolean
  missing?: string[]
}

// GET /content-ops/usage/summary/tenant

export interface ContentOpsUsageSummaryParams {
  days?: number
  asset_type?: string
  run_id?: string
  request_id?: string
}

export interface ContentOpsUsageSummaryBreakdownResponse {
  provider?: string
  model?: string
  asset_type?: string
  cache_mode?: string
  cache_reason?: string
  cache_result?: string
  cache_store_result?: string
  cost_usd: number
  cache_savings_usd: number
  calls: number
  input_tokens: number
  output_tokens: number
}

export interface ContentOpsUsageSummaryResponse {
  period_days: number
  filters: {
    account_id?: string | null
    asset_type?: string | null
    run_id?: string | null
    request_id?: string | null
  }
  summary: {
    total_cost_usd: number
    total_calls: number
    failed_calls: number
    input_tokens: number
    billable_input_tokens: number
    output_tokens: number
    total_tokens: number
    cached_tokens: number
    cache_write_tokens: number
    total_cache_savings_usd: number
    cache_hit_calls: number
    avg_duration_ms: number
    latest_call_at: string | null
  }
  by_model: ContentOpsUsageSummaryBreakdownResponse[]
  by_asset_type: ContentOpsUsageSummaryBreakdownResponse[]
  by_cache_status?: ContentOpsUsageSummaryBreakdownResponse[]
}

// GET/POST/DELETE /content-ops/zendesk-credentials

export interface ContentOpsZendeskCredential {
  id: string
  account_id: string
  email: string
  api_token_prefix: string
  subdomain: string
  base_url: string
  label: string
  added_at: string
  last_used_at?: string | null
  revoked_at?: string | null
}

export interface UpsertContentOpsZendeskCredentialRequest {
  email: string
  api_token: string
  subdomain?: string
  base_url?: string
  label?: string
}

// GET/POST/PUT/DELETE /content-ops/brand-voice-profiles

export interface ContentOpsBrandVoiceProfile {
  id: string
  account_id: string
  name: string
  descriptors: string[]
  exemplars: string[]
  banned_terms: string[]
  preferred_pov?: string | null
  reading_level?: string | null
  metadata: Record<string, unknown>
  created_at: string
  updated_at: string
  archived_at?: string | null
}

export interface UpsertContentOpsBrandVoiceProfileRequest {
  name: string
  descriptors?: string[]
  exemplars?: string[]
  banned_terms?: string[]
  preferred_pov?: string | null
  reading_level?: string | null
  metadata?: Record<string, unknown>
}

export interface ContentOpsBrandVoiceSampleUrlRequest {
  url: string
}

export interface ContentOpsBrandVoiceSampleUrlResponse {
  url: string
  title?: string | null
  text: string
  source_character_count: number
}

// POST /content-ops/preview, /plan, /execute share this body shape.

export interface ContentOpsRequestBody {
  target_mode?: string                             // default "vendor_retention"
  preset?: string | null
  outputs?: string[]
  limit?: number                                   // 1..1000
  max_cost_usd?: number | null
  account_usage_budget_usd?: number | null
  account_usage_budget_days?: number
  content_ops_cache_policy?: string | null
  brand_voice_profile_id?: string | null
  inputs?: Record<string, unknown>
  ingestion_profile?: string                       // default "domain_specific"
  require_quality_gates?: boolean                  // default true
  allow_unimplemented_outputs?: boolean            // default false
}

export interface ContentOpsInputProviderWarning {
  code?: string
  message?: string
  [key: string]: unknown
}

export interface ContentOpsInputProviderDiagnostics {
  provider: string
  metadata: Record<string, unknown>
  warnings: ContentOpsInputProviderWarning[]
}

export interface ContentOpsUsageBudgetEvaluationResponse {
  budget_usd: number
  period_days: number
  current_cost_usd: number
  estimated_cost_usd: number
  projected_cost_usd: number
  exceeded: boolean
}

// POST /content-ops/ingestion/inspect body and response

export interface ContentOpsIngestionInspectRequest {
  rows: Array<Record<string, unknown>>
  source_rows?: boolean                             // default false
  source?: string | null                            // default "api"
  target_mode?: string | null                       // default "vendor_retention"
  max_source_text_chars?: number                    // 1..10000
  sample_limit?: number                             // 0..25
  default_fields?: Record<string, unknown>
  include_source_material?: boolean                 // default false
}

export interface ContentOpsIngestionImportRequest
  extends ContentOpsIngestionInspectRequest {
  replace_existing?: boolean                         // default false
  dry_run?: boolean                                  // default false
}

export interface ContentOpsIngestionFileInspectRequest {
  file: File
  source_rows?: boolean
  source?: string | null
  target_mode?: string | null
  file_format?: 'auto' | 'json' | 'jsonl' | 'csv'
  max_source_text_chars?: number
  sample_limit?: number
  default_fields?: Record<string, unknown>
  include_source_material?: boolean
}

export interface ContentOpsIngestionFileImportRequest
  extends ContentOpsIngestionFileInspectRequest {
  replace_existing?: boolean
  dry_run?: boolean
}

export interface ContentOpsIngestionWarning {
  code: string
  message: string
  row_index?: number
  field?: string
}

export interface ContentOpsIngestionDiagnosticsResponse {
  ok: boolean
  mode: 'opportunities' | 'source_rows'
  source: string
  opportunity_count: number
  warning_count: number
  warning_counts: Record<string, number>
  missing_field_counts: Record<string, number>
  source_type_counts: Record<string, number>
  samples: Array<Record<string, unknown>>
  source_material?: Array<Record<string, unknown>>
  warnings: ContentOpsIngestionWarning[]
}

export interface ContentOpsIngestionImportResultResponse {
  inserted: number
  skipped: number
  dry_run: boolean
  replace_existing: boolean
  target_ids: string[]
  warnings: ContentOpsIngestionWarning[]
  source?: string | null
}

export interface ContentOpsIngestionImportResponse {
  diagnostics: ContentOpsIngestionDiagnosticsResponse
  import: ContentOpsIngestionImportResultResponse
}

export type ContentOpsIngestionImportOutcome =
  | { kind: 'success'; response: ContentOpsIngestionImportResponse }
  | { kind: 'not_ready'; diagnostics: ContentOpsIngestionDiagnosticsResponse }
  | { kind: 'request_invalid'; detail: string }
  | { kind: 'validation_error'; detail: unknown }
  | { kind: 'services_unavailable'; detail: string }

// POST /content-ops/preview response

export interface ContentOpsPreviewResponse {
  can_run: boolean
  outputs: string[]
  estimated_cost_usd: number
  missing_inputs: string[]
  blocked_outputs: string[]
  warnings: string[]
  normalized_request: ContentOpsRequestBody | null
  input_provider?: ContentOpsInputProviderDiagnostics
  usage_budget?: ContentOpsUsageBudgetEvaluationResponse
}

// POST /content-ops/plan response

export type GenerationPlanStepStatus = 'runnable' | 'blocked'

export interface GenerationPlanStep {
  output: string                                   // e.g. "email_campaign"
  runner: string                                   // e.g. "CampaignGenerationService.generate"
  status: GenerationPlanStepStatus
  config: Record<string, unknown>                  // runner-specific config snapshot
  reason: string                                   // populated when status="blocked"
}

export interface GenerationPlanResponse {
  can_execute: boolean
  target_mode: string
  limit: number
  steps: GenerationPlanStep[]
  preview: ContentOpsPreviewResponse
  input_provider?: ContentOpsInputProviderDiagnostics
}

// POST /content-ops/execute response (wire shape)

export type ContentOpsExecutionStatus =
  | 'completed'
  | 'partial'
  | 'failed'
  | 'blocked'

export type ContentOpsStepStatus = 'completed' | 'failed' | 'skipped'

export interface ContentOpsStepReasoningAudit {
  requirement: 'absent' | 'optional_host_context' | string
  service_supports_reasoning: boolean
  provider_configured: boolean
  contexts_used?: number
  consumed_contexts?: CampaignReasoningContextView[]
}

export interface CampaignReasoningContextView {
  summary?: string
  anchor_examples?: Record<string, Array<Record<string, unknown>>>
  witness_highlights?: Array<Record<string, unknown>>
  reference_ids?: Record<string, string[]>
  top_theses?: Array<Record<string, unknown>>
  account_signals?: Array<Record<string, unknown>>
  timing_windows?: Array<Record<string, unknown>>
  proof_points?: Array<Record<string, unknown>>
  coverage_limits?: string[]
  scope_summary?: Record<string, unknown>
  delta_summary?: Record<string, unknown>
  [key: string]: unknown
}

export interface ContentOpsStepExecution {
  output: string
  runner: string
  status: ContentOpsStepStatus
  result: Record<string, unknown>
  error: string                                    // populated when status="failed"
  reasoning?: ContentOpsStepReasoningAudit
}

export interface ContentOpsExecutionResult {
  status: ContentOpsExecutionStatus
  plan: GenerationPlanResponse
  steps: ContentOpsStepExecution[]
  errors: Array<Record<string, unknown>>
  request_id?: string
  usage_summary?: ContentOpsUsageSummaryResponse
  input_provider?: ContentOpsInputProviderDiagnostics
}

// HTTP-code-aware execute outcome. The /execute route maps the
// backend `ContentOpsExecutionResult.status` field onto HTTP codes
// (200 / 207 / 400 / 502); a plain `res.json()` would lose that.
// 422 / 503 / 400-from-ValueError are also surfaced as outcome
// kinds so the UI can render one banner per kind without
// HTTP-code knowledge leaking into screens.

export type ContentOpsExecuteOutcome =
  | { kind: 'completed'; result: ContentOpsExecutionResult }
  | { kind: 'partial'; result: ContentOpsExecutionResult }
  | { kind: 'failed'; result: ContentOpsExecutionResult }
  | { kind: 'blocked'; result: ContentOpsExecutionResult }
  | { kind: 'validation_error'; detail: unknown }                // 422
  | { kind: 'services_unavailable'; detail: string }             // 503
  | { kind: 'request_invalid'; detail: string }                  // 400 from ValueError

// GET /content-assets/{asset}/drafts and review/export helpers.

export type GeneratedAssetType =
  | 'blog_post'
  | 'report'
  | 'landing_page'
  | 'sales_brief'
  | 'social_post'
  | 'ad_copy'
  | 'quote_card'
  | 'stat_card'
  | 'faq_markdown'

export interface GeneratedAssetRepairHistoryEntry {
  attempt?: number
  passed?: boolean
  blockers?: string[] | string
  repair_issues?: string[] | string
  [key: string]: unknown
}

export interface GeneratedAssetDraftMetadata {
  generation_quality_repair_history?: GeneratedAssetRepairHistoryEntry[] | string
  quality_repair_history?: GeneratedAssetRepairHistoryEntry[] | string
  [key: string]: unknown
}

export interface GeneratedAssetRepairResult {
  requested?: number
  generated?: number
  skipped?: number
  saved_ids?: string[]
  errors?: Array<Record<string, unknown>>
  quality_repair_history?: GeneratedAssetRepairHistoryEntry[] | string
  [key: string]: unknown
}

export interface GeneratedAssetDraft {
  id?: string
  title?: string
  slug?: string
  status?: string
  target_id?: string
  target_mode?: string
  report_type?: string
  topic_type?: string
  campaign_name?: string
  brief_type?: string
  channel?: string
  format?: string
  theme?: string
  quote?: string
  attribution?: string
  supporting_text?: string
  text?: string
  primary_text?: string
  source_id?: string
  source_type?: string
  company_name?: string
  vendor_name?: string
  pain_points?: string[] | string
  pain_point_count?: number
  description?: string
  summary?: string
  headline?: string
  content?: string
  markdown?: string
  items?: Array<Record<string, unknown>> | string
  tags?: string[] | string
  hero?: Record<string, unknown>
  sections?: Array<Record<string, unknown>> | string
  cta?: Record<string, unknown> | string
  metadata?: GeneratedAssetDraftMetadata | string
  reference_ids?: string[] | string
  generation_total_tokens?: number
  generation_input_tokens?: number
  generation_output_tokens?: number
  generation_parse_attempts?: number
  generation_quality_repair_attempts?: number
  generation_quality_repair_history?: GeneratedAssetRepairHistoryEntry[] | string
  quality_repair_history?: GeneratedAssetRepairHistoryEntry[] | string
  reasoning_context_used?: boolean
  reasoning_wedge?: string
  reasoning_confidence?: number | string
  section_count?: number
  reference_count?: number
  tag_count?: number
  chart_count?: number
  source_count?: number
  ticket_source_count?: number
  output_checks?: Record<string, unknown> | string
  passed_output_checks?: number
  seo_aeo_readiness?: GeneratedAssetReadiness | string
  geo_readiness?: GeneratedAssetReadiness | string
  repair_result?: GeneratedAssetRepairResult | string
  structured_data?: Record<string, unknown> | string
  robots?: string
  persona?: string
  value_prop?: string
  [key: string]: unknown
}

export interface GeneratedAssetReadiness {
  status?: string
  passed?: number
  total?: number
  missing?: string[]
  checks?: Record<string, unknown>
  [key: string]: unknown
}

export interface GeneratedAssetListParams {
  status?: string
  target_mode?: string
  report_type?: string
  campaign_name?: string
  slug?: string
  topic_type?: string
  brief_type?: string
  channel?: string
  theme?: string
  id?: string | string[]
  format?: string
  limit?: number
}

export interface GeneratedAssetListResponse {
  count: number
  limit: number
  filters: Record<string, unknown>
  rows: GeneratedAssetDraft[]
}

export interface GeneratedAssetReviewResponse {
  account_id?: string | null
  asset: GeneratedAssetType
  id: string
  status: string
  updated: boolean
}

export interface GeneratedAssetBatchReviewResponse {
  account_id?: string | null
  asset: GeneratedAssetType
  ids: string[]
  status: string
  updated: number
  updated_ids: string[]
  missing_ids: string[]
}

export interface GeneratedLandingPageDraftUpdate {
  title?: string
  slug?: string
  hero?: Record<string, unknown>
  sections?: Array<Record<string, unknown>>
  cta?: Record<string, unknown>
  meta?: Record<string, unknown>
  reference_ids?: string[]
}

export interface GeneratedAssetMacroPublishSkippedItem {
  question?: string
  reason?: string
  [key: string]: unknown
}

export interface GeneratedAssetMacroPublishResult {
  status?: string
  external_id?: string | null
  error?: string | null
  [key: string]: unknown
}

export interface GeneratedAssetMacroPublishSummary {
  account_id?: string | null
  asset: GeneratedAssetType
  faq_id: string
  found: boolean
  ok: boolean
  draft_status: string
  publishable_count: number
  skipped_count: number
  published_count: number
  updated_count: number
  failed_count: number
  pending_reconcile_count: number
  draft_status_updated: boolean
  skipped: GeneratedAssetMacroPublishSkippedItem[]
  results: GeneratedAssetMacroPublishResult[]
}

export interface GeneratedAssetMacroPublishAttempt {
  id: string
  faq_id: string
  draft_status: string
  ok: boolean
  publishable_count: number
  skipped_count: number
  published_count: number
  updated_count: number
  failed_count: number
  pending_reconcile_count: number
  draft_status_updated: boolean
  skipped: GeneratedAssetMacroPublishSkippedItem[]
  results: GeneratedAssetMacroPublishResult[]
  created_at: string
}

export interface GeneratedAssetMacroPublishAttemptsResponse {
  account_id?: string | null
  asset: GeneratedAssetType
  faq_id: string
  count: number
  limit: number
  attempts: GeneratedAssetMacroPublishAttempt[]
}

// ---------------------------------------------------------------------------
// Internal fetch plumbing
// ---------------------------------------------------------------------------

function authHeaders(): Record<string, string> {
  const token = localStorage.getItem('atlas_token')
  return token ? { Authorization: `Bearer ${token}` } : {}
}

function forceLogout(): never {
  localStorage.removeItem('atlas_token')
  localStorage.removeItem('atlas_refresh_token')
  window.location.href = '/login'
  throw new Error('Session expired')
}

function maybeFallbackApiPath(url: string): string | null {
  if (!url.includes('/api/v1/')) return null
  return url.replace('/api/v1/', '/api/')
}

async function fetchWithApiFallback(url: string, init?: RequestInit): Promise<Response> {
  const res = await fetch(url, init)
  if (res.status !== 404) return res
  const fallbackUrl = maybeFallbackApiPath(url)
  if (!fallbackUrl) return res
  return fetch(fallbackUrl, init)
}

async function rawJson<T>(res: Response): Promise<T> {
  return res.json() as Promise<T>
}

async function rawText(res: Response): Promise<string> {
  return res.text().catch(() => '')
}

async function withRefreshOn401(
  doFetch: () => Promise<Response>,
): Promise<Response> {
  const res = await doFetch()
  if (res.status !== 401) return res
  const newToken = await tryRefreshToken()
  if (!newToken) forceLogout()
  const retryRes = await doFetch()
  if (retryRes.status === 401) forceLogout()
  return retryRes
}

async function getJson<T>(
  path: string,
  params: object = {},
): Promise<T> {
  const url = `${BASE}${path}${queryString(params)}`
  const doFetch = () => fetchWithApiFallback(url, { headers: authHeaders() })
  const res = await withRefreshOn401(doFetch)
  if (!res.ok) {
    const body = await rawText(res)
    throw new Error(`API ${res.status}: ${body || res.statusText}`)
  }
  return rawJson<T>(res)
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const url = `${BASE}${path}`
  const doFetch = () =>
    fetchWithApiFallback(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...authHeaders() },
      body: JSON.stringify(body),
    })
  const res = await withRefreshOn401(doFetch)
  if (!res.ok) {
    const text = await rawText(res)
    throw new Error(`API ${res.status}: ${text || res.statusText}`)
  }
  return rawJson<T>(res)
}

async function putJson<T>(path: string, body: unknown): Promise<T> {
  const url = `${BASE}${path}`
  const doFetch = () =>
    fetchWithApiFallback(url, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json', ...authHeaders() },
      body: JSON.stringify(body),
    })
  const res = await withRefreshOn401(doFetch)
  if (!res.ok) {
    const text = await rawText(res)
    throw new Error(`API ${res.status}: ${text || res.statusText}`)
  }
  return rawJson<T>(res)
}

async function deleteJson(path: string): Promise<void> {
  const url = `${BASE}${path}`
  const doFetch = () =>
    fetchWithApiFallback(url, {
      method: 'DELETE',
      headers: authHeaders(),
    })
  const res = await withRefreshOn401(doFetch)
  if (!res.ok) {
    const text = await rawText(res)
    throw new Error(`API ${res.status}: ${text || res.statusText}`)
  }
}

async function postMultipart<T>(path: string, body: FormData): Promise<T> {
  const url = `${BASE}${path}`
  const doFetch = () =>
    fetchWithApiFallback(url, {
      method: 'POST',
      headers: authHeaders(),
      body,
    })
  const res = await withRefreshOn401(doFetch)
  if (!res.ok) {
    const text = await rawText(res)
    throw new Error(`API ${res.status}: ${text || res.statusText}`)
  }
  return rawJson<T>(res)
}

function ingestionFileFormData(
  body: ContentOpsIngestionFileInspectRequest,
): FormData {
  const formData = new FormData()
  formData.set('file', body.file)
  appendOptionalBoolean(formData, 'source_rows', body.source_rows)
  appendOptionalString(formData, 'source', body.source)
  appendOptionalString(formData, 'target_mode', body.target_mode)
  appendOptionalString(formData, 'file_format', body.file_format)
  appendOptionalNumber(
    formData,
    'max_source_text_chars',
    body.max_source_text_chars,
  )
  appendOptionalNumber(formData, 'sample_limit', body.sample_limit)
  appendOptionalBoolean(
    formData,
    'include_source_material',
    body.include_source_material,
  )
  if (body.default_fields) {
    formData.set('default_fields', JSON.stringify(body.default_fields))
  }
  return formData
}

function appendOptionalString(
  formData: FormData,
  key: string,
  value: string | null | undefined,
): void {
  const text = typeof value === 'string' ? value.trim() : ''
  if (text) formData.set(key, text)
}

function appendOptionalNumber(
  formData: FormData,
  key: string,
  value: number | null | undefined,
): void {
  if (typeof value === 'number' && Number.isFinite(value)) {
    formData.set(key, String(value))
  }
}

function appendOptionalBoolean(
  formData: FormData,
  key: string,
  value: boolean | null | undefined,
): void {
  if (typeof value === 'boolean') formData.set(key, String(value))
}

function queryString(params: object): string {
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null) continue
    if (Array.isArray(value)) {
      for (const item of value) {
        search.append(key, String(item))
      }
      continue
    }
    search.set(key, String(value))
  }
  const query = search.toString()
  return query ? `?${query}` : ''
}

async function getAssetJson<T>(
  asset: GeneratedAssetType,
  path: string,
  params: GeneratedAssetListParams = {},
): Promise<T> {
  const url = `${ASSETS_BASE}/${asset}${path}${queryString(params)}`
  const doFetch = () => fetchWithApiFallback(url, { headers: authHeaders() })
  const res = await withRefreshOn401(doFetch)
  if (!res.ok) {
    const body = await rawText(res)
    throw new Error(`API ${res.status}: ${body || res.statusText}`)
  }
  return rawJson<T>(res)
}

async function postAssetJson<T>(
  asset: GeneratedAssetType,
  path: string,
  body: Record<string, unknown>,
): Promise<T> {
  const url = `${ASSETS_BASE}/${asset}${path}`
  const doFetch = () =>
    fetchWithApiFallback(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...authHeaders() },
      body: JSON.stringify(body),
    })
  const res = await withRefreshOn401(doFetch)
  if (!res.ok) {
    const text = await rawText(res)
    throw new Error(`API ${res.status}: ${text || res.statusText}`)
  }
  return rawJson<T>(res)
}

async function patchAssetJson<T>(
  asset: GeneratedAssetType,
  path: string,
  body: Record<string, unknown>,
): Promise<T> {
  const url = `${ASSETS_BASE}/${asset}${path}`
  const doFetch = () =>
    fetchWithApiFallback(url, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json', ...authHeaders() },
      body: JSON.stringify(body),
    })
  const res = await withRefreshOn401(doFetch)
  if (!res.ok) {
    const text = await rawText(res)
    throw new Error(`API ${res.status}: ${text || res.statusText}`)
  }
  return rawJson<T>(res)
}

async function getAssetText(
  asset: GeneratedAssetType,
  path: string,
  params: GeneratedAssetListParams = {},
): Promise<string> {
  const url = `${ASSETS_BASE}/${asset}${path}${queryString(params)}`
  const doFetch = () => fetchWithApiFallback(url, { headers: authHeaders() })
  const res = await withRefreshOn401(doFetch)
  if (!res.ok) {
    const body = await rawText(res)
    throw new Error(`API ${res.status}: ${body || res.statusText}`)
  }
  return rawText(res)
}

async function getAssetBlob(
  asset: GeneratedAssetType,
  path: string,
  params: GeneratedAssetListParams = {},
): Promise<Blob> {
  const url = `${ASSETS_BASE}/${asset}${path}${queryString(params)}`
  const doFetch = () => fetchWithApiFallback(url, { headers: authHeaders() })
  const res = await withRefreshOn401(doFetch)
  if (!res.ok) {
    const body = await rawText(res)
    throw new Error(`API ${res.status}: ${body || res.statusText}`)
  }
  return res.blob()
}

async function getPublicAssetJson<T>(path: string): Promise<T> {
  const url = `${ASSETS_BASE}${path}`
  const res = await fetchWithApiFallback(url)
  if (!res.ok) {
    const body = await rawText(res)
    throw new Error(`API ${res.status}: ${body || res.statusText}`)
  }
  return rawJson<T>(res)
}

// ---------------------------------------------------------------------------
// Public fetch wrappers
// ---------------------------------------------------------------------------

/** GET /content-ops/control-surfaces -- catalog + presets + execution flags. */
export function fetchContentOpsControlSurfaces(): Promise<ContentOpsCatalogResponse> {
  return getJson<ContentOpsCatalogResponse>('/control-surfaces')
}

/** GET /content-ops/usage/summary/tenant -- account-scoped LLM usage rollup. */
export function fetchContentOpsTenantUsageSummary(
  params: ContentOpsUsageSummaryParams = {},
): Promise<ContentOpsUsageSummaryResponse> {
  return getJson<ContentOpsUsageSummaryResponse>(
    '/usage/summary/tenant',
    params,
  )
}

/** GET /content-ops/zendesk-credentials -- display-safe tenant credential list. */
export function fetchContentOpsZendeskCredentials(): Promise<
  ContentOpsZendeskCredential[]
> {
  return getJson<ContentOpsZendeskCredential[]>('/zendesk-credentials')
}

/** POST /content-ops/zendesk-credentials -- add or rotate tenant Zendesk credential. */
export function saveContentOpsZendeskCredential(
  body: UpsertContentOpsZendeskCredentialRequest,
): Promise<ContentOpsZendeskCredential> {
  return postJson<ContentOpsZendeskCredential>('/zendesk-credentials', body)
}

/** DELETE /content-ops/zendesk-credentials/{id} -- revoke tenant Zendesk credential. */
export function revokeContentOpsZendeskCredential(id: string): Promise<void> {
  return deleteJson(`/zendesk-credentials/${encodeURIComponent(id)}`)
}

/** GET /content-ops/brand-voice-profiles -- tenant saved profile list. */
export function fetchContentOpsBrandVoiceProfiles(): Promise<
  ContentOpsBrandVoiceProfile[]
> {
  return getJson<ContentOpsBrandVoiceProfile[]>('/brand-voice-profiles')
}

/** POST /content-ops/brand-voice-profiles -- create a tenant saved profile. */
export function createContentOpsBrandVoiceProfile(
  body: UpsertContentOpsBrandVoiceProfileRequest,
): Promise<ContentOpsBrandVoiceProfile> {
  return postJson<ContentOpsBrandVoiceProfile>('/brand-voice-profiles', body)
}

/** POST /content-ops/brand-voice-profiles/sample-url -- extract public page copy. */
export function fetchContentOpsBrandVoiceSampleUrl(
  body: ContentOpsBrandVoiceSampleUrlRequest,
): Promise<ContentOpsBrandVoiceSampleUrlResponse> {
  return postJson<ContentOpsBrandVoiceSampleUrlResponse>(
    '/brand-voice-profiles/sample-url',
    body,
  )
}

/** PUT /content-ops/brand-voice-profiles/{id} -- update a tenant saved profile. */
export function updateContentOpsBrandVoiceProfile(
  id: string,
  body: UpsertContentOpsBrandVoiceProfileRequest,
): Promise<ContentOpsBrandVoiceProfile> {
  return putJson<ContentOpsBrandVoiceProfile>(
    `/brand-voice-profiles/${encodeURIComponent(id)}`,
    body,
  )
}

/** DELETE /content-ops/brand-voice-profiles/{id} -- archive a tenant saved profile. */
export function deleteContentOpsBrandVoiceProfile(id: string): Promise<void> {
  return deleteJson(`/brand-voice-profiles/${encodeURIComponent(id)}`)
}

/** GET /content-assets/{asset}/drafts -- persisted generated assets. */
export function fetchGeneratedAssetDrafts(
  asset: GeneratedAssetType,
  params: GeneratedAssetListParams = {},
): Promise<GeneratedAssetListResponse> {
  return getAssetJson<GeneratedAssetListResponse>(asset, '/drafts', params)
}

/** GET /content-assets/{asset}/drafts/export?format=csv -- CSV body. */
export function exportGeneratedAssetDraftsCsv(
  asset: GeneratedAssetType,
  params: GeneratedAssetListParams = {},
): Promise<string> {
  return getAssetText(asset, '/drafts/export', { ...params, format: 'csv' })
}

/** GET /content-assets/{asset}/drafts/export?format=html -- static visual cards. */
export function exportGeneratedAssetDraftsHtml(
  asset: GeneratedAssetType,
  params: GeneratedAssetListParams = {},
): Promise<string> {
  return getAssetText(asset, '/drafts/export', { ...params, format: 'html' })
}

/** GET /content-assets/{asset}/drafts/export?format=png -- visual card screenshot. */
export function exportGeneratedAssetDraftsPng(
  asset: GeneratedAssetType,
  params: GeneratedAssetListParams = {},
): Promise<Blob> {
  return getAssetBlob(asset, '/drafts/export', { ...params, format: 'png' })
}

/** POST /content-assets/{asset}/drafts/review -- approve/reject a draft. */
export function reviewGeneratedAssetDraft(
  asset: GeneratedAssetType,
  id: string,
  status: 'approved' | 'rejected',
): Promise<GeneratedAssetReviewResponse> {
  return postAssetJson<GeneratedAssetReviewResponse>(asset, '/drafts/review', {
    id,
    status,
  })
}

/** POST /content-assets/{asset}/drafts/review-batch -- approve/reject drafts. */
export function reviewGeneratedAssetDrafts(
  asset: GeneratedAssetType,
  ids: string[],
  status: 'approved' | 'rejected',
): Promise<GeneratedAssetBatchReviewResponse> {
  return postAssetJson<GeneratedAssetBatchReviewResponse>(
    asset,
    '/drafts/review-batch',
    { ids, status },
  )
}

/** PATCH /content-assets/landing_page/drafts/{id} -- edit a generated landing page draft. */
export function updateGeneratedLandingPageDraft(
  id: string,
  body: GeneratedLandingPageDraftUpdate,
): Promise<GeneratedAssetDraft> {
  return patchAssetJson<GeneratedAssetDraft>(
    'landing_page',
    `/drafts/${encodeURIComponent(id)}`,
    { ...body },
  )
}

/** POST /content-assets/landing_page/drafts/{id}/repair -- repair a generated landing page draft. */
export function repairGeneratedLandingPageDraft(
  id: string,
): Promise<GeneratedAssetDraft> {
  const url = `${ASSETS_BASE}/landing_page/drafts/${encodeURIComponent(id)}/repair`
  const doFetch = () =>
    fetchWithApiFallback(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...authHeaders() },
      body: '{}',
    })
  return withRefreshOn401(doFetch).then(async (res) => {
    if (!res.ok) {
      throw new Error(await repairErrorMessage(res))
    }
    return rawJson<GeneratedAssetDraft>(res)
  })
}

/** POST /content-assets/faq_markdown/drafts/{id}/publish-macros -- publish approved FAQ macros. */
export function publishGeneratedFaqMacros(
  id: string,
): Promise<GeneratedAssetMacroPublishSummary> {
  return postAssetJson<GeneratedAssetMacroPublishSummary>(
    'faq_markdown',
    `/drafts/${encodeURIComponent(id)}/publish-macros`,
    {},
  )
}

/** GET /content-assets/faq_markdown/drafts/{id}/publish-macro-attempts -- recent publish history. */
export function fetchGeneratedFaqMacroPublishAttempts(
  id: string,
  params: { limit?: number } = {},
): Promise<GeneratedAssetMacroPublishAttemptsResponse> {
  return getAssetJson<GeneratedAssetMacroPublishAttemptsResponse>(
    'faq_markdown',
    `/drafts/${encodeURIComponent(id)}/publish-macro-attempts`,
    params,
  )
}

async function repairErrorMessage(res: Response): Promise<string> {
  const text = await rawText(res)
  const body = jsonFromText(text)
  if (body !== null) {
    const detail = recordValue(body)?.detail
    if (typeof detail === 'string' && detail.trim()) {
      return `API ${res.status}: ${detail.trim()}`
    }
    const detailRecord = recordValue(detail)
    if (detailRecord) {
      const message = textValue(detailRecord.message)
      const blockers = repairResultBlockers(detailRecord.repair_result)
      const suffix = blockers.length > 0 ? `: ${blockers.join(', ')}` : ''
      return `API ${res.status}: ${message || 'Landing page repair failed'}${suffix}`
    }
  }
  return `API ${res.status}: ${text || res.statusText}`
}

function jsonFromText(text: string): unknown {
  if (!text.trim()) return null
  try {
    return JSON.parse(text)
  } catch {
    return null
  }
}

function repairResultBlockers(value: unknown): string[] {
  const result = recordValue(value)
  const errors = Array.isArray(result?.errors) ? result.errors : []
  const firstError = recordValue(errors[0])
  return valueList(firstError?.blockers).slice(0, 4)
}

function recordValue(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null
}

function valueList(value: unknown): string[] {
  if (Array.isArray(value)) return value.map(textValue).filter(Boolean)
  const text = textValue(value)
  return text ? text.split(',').map((item) => item.trim()).filter(Boolean) : []
}

function textValue(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

/** GET /content-assets/landing_page/public/{id} -- approved public landing page. */
export function fetchPublicLandingPageDraft(
  id: string,
): Promise<GeneratedAssetDraft> {
  return getPublicAssetJson<GeneratedAssetDraft>(
    `/landing_page/public/${encodeURIComponent(id)}`,
  )
}

/** POST /content-ops/preview -- preflight validation. */
export function previewContentOpsRun(
  body: ContentOpsRequestBody,
): Promise<ContentOpsPreviewResponse> {
  return postJson<ContentOpsPreviewResponse>('/preview', body)
}

/** POST /content-ops/plan -- build runnable plan. */
export function planContentOpsRun(
  body: ContentOpsRequestBody,
): Promise<GenerationPlanResponse> {
  return postJson<GenerationPlanResponse>('/plan', body)
}

/** POST /content-ops/ingestion/inspect -- inspect inline opportunity/source rows. */
export function inspectContentOpsIngestion(
  body: ContentOpsIngestionInspectRequest,
): Promise<ContentOpsIngestionDiagnosticsResponse> {
  return postJson<ContentOpsIngestionDiagnosticsResponse>('/ingestion/inspect', body)
}

/** POST /content-ops/ingestion/files/inspect -- inspect an uploaded ingestion file. */
export function inspectContentOpsIngestionFile(
  body: ContentOpsIngestionFileInspectRequest,
): Promise<ContentOpsIngestionDiagnosticsResponse> {
  return postMultipart<ContentOpsIngestionDiagnosticsResponse>(
    '/ingestion/files/inspect',
    ingestionFileFormData(body),
  )
}

/** POST /content-ops/ingestion/import -- deprecated inline compatibility path. */
export async function importContentOpsIngestion(
  body: ContentOpsIngestionImportRequest,
): Promise<ContentOpsIngestionImportOutcome> {
  const url = `${BASE}/ingestion/import`
  const doFetch = () =>
    fetchWithApiFallback(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...authHeaders() },
      body: JSON.stringify(body),
    })
  const res = await withRefreshOn401(doFetch)
  return ingestionImportOutcomeFromResponse(res)
}

/** POST /content-ops/ingestion/files/import -- import an uploaded ingestion file. */
export async function importContentOpsIngestionFile(
  body: ContentOpsIngestionFileImportRequest,
): Promise<ContentOpsIngestionImportOutcome> {
  const url = `${BASE}/ingestion/files/import`
  const formData = ingestionFileFormData(body)
  appendOptionalBoolean(formData, 'replace_existing', body.replace_existing)
  appendOptionalBoolean(formData, 'dry_run', body.dry_run)
  const doFetch = () =>
    fetchWithApiFallback(url, {
      method: 'POST',
      headers: authHeaders(),
      body: formData,
    })
  const res = await withRefreshOn401(doFetch)
  return ingestionImportOutcomeFromResponse(res)
}

async function ingestionImportOutcomeFromResponse(
  res: Response,
): Promise<ContentOpsIngestionImportOutcome> {
  if (res.status === 200) {
    return {
      kind: 'success',
      response: await rawJson<ContentOpsIngestionImportResponse>(res),
    }
  }
  if (res.status === 400) {
    const text = await rawText(res)
    try {
      const detail = (JSON.parse(text) as { detail?: unknown })?.detail
      if (isIngestionNotReadyDetail(detail)) {
        return { kind: 'not_ready', diagnostics: detail.diagnostics }
      }
      return {
        kind: 'request_invalid',
        detail: typeof detail === 'string' ? detail : text,
      }
    } catch {
      return { kind: 'request_invalid', detail: text || res.statusText }
    }
  }
  if (res.status === 422) {
    const text = await rawText(res)
    let detail: unknown = text
    try {
      detail = (JSON.parse(text) as { detail?: unknown })?.detail ?? text
    } catch {
      // keep raw text
    }
    return { kind: 'validation_error', detail }
  }
  if (res.status === 503) {
    const text = await rawText(res)
    let detail = text || res.statusText
    try {
      const parsed = JSON.parse(text) as { detail?: unknown }
      if (typeof parsed?.detail === 'string') detail = parsed.detail
    } catch {
      // keep raw text
    }
    return { kind: 'services_unavailable', detail }
  }
  const text = await rawText(res)
  throw new Error(`API ${res.status}: ${text || res.statusText}`)
}

/**
 * POST /content-ops/execute -- run the plan via host-injected services.
 *
 * Returns a discriminated outcome. The backend's status-to-HTTP
 * mapping (200 / 207 / 400 / 502) is part of the contract, so this
 * wrapper deliberately handles non-2xx codes that carry a typed
 * payload rather than throwing on every `!res.ok`.
 */
export async function executeContentOpsRun(
  body: ContentOpsRequestBody,
): Promise<ContentOpsExecuteOutcome> {
  const url = `${BASE}/execute`
  const doFetch = () =>
    fetchWithApiFallback(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...authHeaders() },
      body: JSON.stringify(body),
    })
  const res = await withRefreshOn401(doFetch)

  if (res.status === 200) {
    const result = await rawJson<ContentOpsExecutionResult>(res)
    return { kind: 'completed', result }
  }
  if (res.status === 207) {
    const result = await rawJson<ContentOpsExecutionResult>(res)
    // FastAPI wraps the result in `{ detail: ... }` when raised via
    // HTTPException; gracefully unwrap when present.
    return {
      kind: 'partial',
      result: unwrapDetail<ContentOpsExecutionResult>(result),
    }
  }
  if (res.status === 502) {
    const result = await rawJson<ContentOpsExecutionResult>(res)
    return {
      kind: 'failed',
      result: unwrapDetail<ContentOpsExecutionResult>(result),
    }
  }
  if (res.status === 400) {
    // Backend returns either a sanitized execution result (status="blocked")
    // OR a string detail (ValueError from request_from_mapping).
    const text = await rawText(res)
    try {
      const parsed = JSON.parse(text) as { detail?: unknown }
      const detail = parsed?.detail
      if (detail && typeof detail === 'object' && 'status' in detail) {
        return {
          kind: 'blocked',
          result: detail as ContentOpsExecutionResult,
        }
      }
      return {
        kind: 'request_invalid',
        detail: typeof detail === 'string' ? detail : text,
      }
    } catch {
      return { kind: 'request_invalid', detail: text || res.statusText }
    }
  }
  if (res.status === 422) {
    const text = await rawText(res)
    let detail: unknown = text
    try {
      detail = (JSON.parse(text) as { detail?: unknown })?.detail ?? text
    } catch {
      // keep raw text
    }
    return { kind: 'validation_error', detail }
  }
  if (res.status === 503) {
    const text = await rawText(res)
    let detail = text || res.statusText
    try {
      const parsed = JSON.parse(text) as { detail?: unknown }
      if (typeof parsed?.detail === 'string') detail = parsed.detail
    } catch {
      // keep raw text
    }
    return { kind: 'services_unavailable', detail }
  }
  // Anything else is a hard failure -- surface like the rest of the API.
  const text = await rawText(res)
  throw new Error(`API ${res.status}: ${text || res.statusText}`)
}

function unwrapDetail<T>(payload: T | { detail: T }): T {
  if (
    payload &&
    typeof payload === 'object' &&
    'detail' in payload &&
    (payload as { detail: unknown }).detail
  ) {
    return (payload as { detail: T }).detail
  }
  return payload as T
}

function isIngestionNotReadyDetail(
  value: unknown,
): value is {
  reason: 'ingestion_not_ready'
  diagnostics: ContentOpsIngestionDiagnosticsResponse
} {
  if (!value || typeof value !== 'object') return false
  const detail = value as Record<string, unknown>
  const diagnostics = detail.diagnostics
  return (
    detail.reason === 'ingestion_not_ready' &&
    !!diagnostics &&
    typeof diagnostics === 'object' &&
    'ok' in diagnostics &&
    'warnings' in diagnostics
  )
}
