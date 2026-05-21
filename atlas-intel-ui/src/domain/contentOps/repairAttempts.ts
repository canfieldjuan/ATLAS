export const LANDING_PAGE_QUALITY_REPAIR_INPUT =
  'landing_page_quality_repair_attempts'

export const MAX_LANDING_PAGE_QUALITY_REPAIR_ATTEMPTS = 10

export const INVALID_LANDING_PAGE_QUALITY_REPAIR_VALUE = '__invalid__'

export const LANDING_PAGE_QUALITY_REPAIR_OPTIONS = Array.from(
  { length: MAX_LANDING_PAGE_QUALITY_REPAIR_ATTEMPTS + 1 },
  (_, index) => String(index),
)

export function normalizeLandingPageRepairAttemptValue(
  value: unknown,
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
    normalized < 0 ||
    normalized > MAX_LANDING_PAGE_QUALITY_REPAIR_ATTEMPTS
  ) {
    return { ok: false }
  }
  return { ok: true, value: normalized }
}
