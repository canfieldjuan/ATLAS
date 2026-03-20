import type { ReportDetail, VendorProfile } from '../types'

const JSONISH_PATTERN = /^\s*[\[{].*[\]}]\s*$/s

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
    return parsed === value ? value : normalizeUnknown(parsed, key)
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

export function normalizeVendorProfile(profile: VendorProfile): VendorProfile {
  return {
    ...profile,
    churn_signal: normalizeUnknown(profile.churn_signal, 'churn_signal') as VendorProfile['churn_signal'],
    high_intent_companies: normalizeUnknown(profile.high_intent_companies, 'high_intent_companies') as VendorProfile['high_intent_companies'],
    pain_distribution: normalizeUnknown(profile.pain_distribution, 'pain_distribution') as VendorProfile['pain_distribution'],
  }
}
