/**
 * AI Content Ops API adapter.
 *
 * Typed fetch wrappers + wire-shape types for the four
 * `/content-ops/*` routes the backend exposes:
 *
 *   GET  /content-ops/control-surfaces
 *   POST /content-ops/preview
 *   POST /content-ops/plan
 *   POST /content-ops/execute
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

export interface ContentOpsCatalogResponse {
  outputs: ContentOpsOutputDefinition[]
  presets: ContentOpsPreset[]
  execution: {
    configured: boolean
    configured_outputs: string[]
  }
  reasoning: {
    configured: boolean
  }
  ingestion_profiles: string[]
}

// POST /content-ops/preview, /plan, /execute share this body shape.

export interface ContentOpsRequestBody {
  target_mode?: string                             // default "vendor_retention"
  preset?: string | null
  outputs?: string[]
  limit?: number                                   // 1..1000
  max_cost_usd?: number | null
  inputs?: Record<string, unknown>
  ingestion_profile?: string                       // default "domain_specific"
  require_quality_gates?: boolean                  // default true
  allow_unimplemented_outputs?: boolean            // default false
}

// POST /content-ops/preview response

export interface ContentOpsPreviewResponse {
  can_run: boolean
  outputs: string[]
  estimated_cost_usd: number
  missing_inputs: string[]
  blocked_outputs: string[]
  warnings: string[]
  normalized_request: ContentOpsRequestBody | null
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

async function getJson<T>(path: string): Promise<T> {
  const url = `${BASE}${path}`
  const doFetch = () => fetchWithApiFallback(url, { headers: authHeaders() })
  const res = await withRefreshOn401(doFetch)
  if (!res.ok) {
    const body = await rawText(res)
    throw new Error(`API ${res.status}: ${body || res.statusText}`)
  }
  return rawJson<T>(res)
}

async function postJson<T>(path: string, body: ContentOpsRequestBody): Promise<T> {
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

// ---------------------------------------------------------------------------
// Public fetch wrappers
// ---------------------------------------------------------------------------

/** GET /content-ops/control-surfaces -- catalog + presets + execution flags. */
export function fetchContentOpsControlSurfaces(): Promise<ContentOpsCatalogResponse> {
  return getJson<ContentOpsCatalogResponse>('/control-surfaces')
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
