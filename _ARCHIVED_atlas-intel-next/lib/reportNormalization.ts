import type { ReportDetail, ReportSubscription, VendorProfile } from '@/lib/types'

const JSONISH_PATTERN = /^\s*[\[{].*[\]}]\s*$/s
const LIST_STRING_KEYS = new Set([
  'integration_stack',
  'archetype_key_signals',
  'falsification_conditions',
  'uncertainty_sources',
  'top_alternatives',
  'displacement_triggers',
  'proof_points',
  'top_feature_gaps',
  'key_signals',
  'shared',
  'challenger_exclusive',
  'incumbent_exclusive',
  'shared_pain_categories',
  'shared_alternatives',
  'shared_vendors',
  'discovery_questions',
  'landmine_questions',
  'commonly_switched_from',
])

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function tryParseJsonish(raw: string): unknown {
  const text = raw.trim()
  if (!JSONISH_PATTERN.test(text)) return raw
  try {
    return JSON.parse(text) as unknown
  } catch {
    return raw
  }
}

function uniqueStrings(values: string[]): string[] {
  const seen = new Set<string>()
  const out: string[] = []
  for (const value of values) {
    const item = value.trim()
    if (!item || seen.has(item)) continue
    seen.add(item)
    out.push(item)
  }
  return out
}

function salvageStringList(raw: string, key: string): string[] | null {
  const text = raw.trim()
  if (!text) return null
  const looksListLike = text.startsWith('[') || text.includes('","') || text.includes('",')
  if (!LIST_STRING_KEYS.has(key) && !looksListLike) return null

  const quotedTokens = Array.from(text.matchAll(/"([^"]+)"/g))
    .map((match) => (match[1] ?? '').trim())
    .filter(Boolean)
  if (quotedTokens.length > 0) {
    return uniqueStrings(quotedTokens)
  }

  const stripped = text
    .replace(/^\s*\[/, '')
    .replace(/\]\s*$/, '')
  const splitTokens = stripped
    .split(',')
    .map((item) => item.trim().replace(/^"+|"+$/g, '').trim())
    .filter(Boolean)
  if (splitTokens.length > 0) {
    return uniqueStrings(splitTokens)
  }

  return null
}

function normalizeInsights(value: unknown[]): unknown[] {
  return value.map((item) => {
    if (typeof item === 'string') {
      return { insight: item, evidence: '' }
    }
    if (isRecord(item)) {
      return {
        insight: String(item.insight ?? item.name ?? item.summary ?? ''),
        evidence: String(item.evidence ?? item.metric ?? item.detail ?? ''),
      }
    }
    return item
  })
}

export function normalizeUnknown(value: unknown, key = ''): unknown {
  if (typeof value === 'string') {
    const parsed = tryParseJsonish(value)
    if (parsed !== value) return normalizeUnknown(parsed, key)
    const salvaged = salvageStringList(value, key)
    return salvaged ?? value
  }

  if (Array.isArray(value)) {
    const normalized = value.map((item) => normalizeUnknown(item, key))
    if (key === 'key_insights') return normalizeInsights(normalized)
    return normalized
  }

  if (!isRecord(value)) return value

  const normalized: Record<string, unknown> = {}
  for (const [entryKey, entryValue] of Object.entries(value)) {
    normalized[entryKey] = normalizeUnknown(entryValue, entryKey)
  }
  return normalized
}

export function normalizeReportObject(data: Record<string, unknown> | null | undefined): Record<string, unknown> {
  if (!data) return {}
  const normalized = normalizeUnknown(data, 'root')
  return isRecord(normalized) ? normalized : {}
}

export function normalizeReportDetail(report: ReportDetail): ReportDetail {
  return {
    ...report,
    intelligence_data: normalizeUnknown(report.intelligence_data, 'intelligence_data') as ReportDetail['intelligence_data'],
    data_density: normalizeUnknown(report.data_density, 'data_density') as ReportDetail['data_density'],
  }
}

export function normalizeReportSubscription(
  subscription: ReportSubscription,
): ReportSubscription {
  return {
    ...subscription,
    scope_label: typeof subscription.scope_label === 'string' ? subscription.scope_label : '',
    recipient_emails: Array.isArray(subscription.recipient_emails)
      ? subscription.recipient_emails
          .map((value) => String(value).trim())
          .filter(Boolean)
      : [],
    delivery_note: typeof subscription.delivery_note === 'string' ? subscription.delivery_note : '',
    last_delivery_status: (
      subscription.last_delivery_status === 'sent'
      || subscription.last_delivery_status === 'partial'
      || subscription.last_delivery_status === 'skipped'
      || subscription.last_delivery_status === 'dry_run'
      || subscription.last_delivery_status === 'failed'
    )
      ? subscription.last_delivery_status
      : null,
    last_delivery_at: typeof subscription.last_delivery_at === 'string' ? subscription.last_delivery_at : null,
    last_delivery_summary: typeof subscription.last_delivery_summary === 'string' ? subscription.last_delivery_summary : '',
    last_delivery_error: typeof subscription.last_delivery_error === 'string' ? subscription.last_delivery_error : '',
    last_delivery_report_count: typeof subscription.last_delivery_report_count === 'number'
      ? subscription.last_delivery_report_count
      : 0,
  }
}

export function normalizeVendorProfile(profile: VendorProfile): VendorProfile {
  return {
    ...profile,
    churn_signal: normalizeUnknown(profile.churn_signal, 'churn_signal') as VendorProfile['churn_signal'],
    high_intent_companies: normalizeUnknown(profile.high_intent_companies, 'high_intent_companies') as VendorProfile['high_intent_companies'],
    pain_distribution: normalizeUnknown(profile.pain_distribution, 'pain_distribution') as VendorProfile['pain_distribution'],
  }
}
