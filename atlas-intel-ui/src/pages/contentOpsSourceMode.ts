const SOURCE_TYPE_INPUT = 'source_type'
const SOURCE_MATERIAL_TYPE_INPUT = 'source_material_type'
const SOURCE_FAQ_IDS_INPUT = 'source_faq_ids'
const SOURCE_B2B_DISPLACEMENT_VENDORS_INPUT = 'b2b_displacement_vendors'

export type ContentOpsSourceMode = 'support_ticket' | 'reviews' | 'competitive'

export type ParsedInputsJsonObject =
  | { ok: true; value: Record<string, unknown> }
  | { ok: false; message: string }

export type UpdatedInputsJson =
  | { ok: true; value: string }
  | { ok: false; message: string }

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export function parseInputsJsonObject(value: string): ParsedInputsJsonObject {
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

export function sourceModeDraftValue(
  parsed: ParsedInputsJsonObject,
): ContentOpsSourceMode {
  if (!parsed.ok) return 'support_ticket'
  const raw =
    parsed.value[SOURCE_TYPE_INPUT] ?? parsed.value[SOURCE_MATERIAL_TYPE_INPUT]
  const value = String(raw ?? '').trim().toLowerCase().replace(/-/g, '_')
  if (value === 'review' || value === 'reviews') return 'reviews'
  if (
    value === 'competitive' ||
    value === 'competition' ||
    value === 'competitor' ||
    value === 'competitors' ||
    value === 'competitive_displacement' ||
    value === 'competitive_signal' ||
    value === 'competitive_signals' ||
    value === 'displacement' ||
    value === 'displacement_edge' ||
    value === 'displacement_edges'
  ) {
    return 'competitive'
  }
  return 'support_ticket'
}

export function updateSourceModeInputJson(
  current: string,
  mode: ContentOpsSourceMode,
): UpdatedInputsJson {
  const parsed = parseInputsJsonObject(current)
  if (!parsed.ok) return parsed

  const next = { ...parsed.value }
  delete next[SOURCE_MATERIAL_TYPE_INPUT]
  if (mode === 'reviews') {
    next[SOURCE_TYPE_INPUT] = 'reviews'
    delete next[SOURCE_FAQ_IDS_INPUT]
    delete next[SOURCE_B2B_DISPLACEMENT_VENDORS_INPUT]
  } else if (mode === 'competitive') {
    next[SOURCE_TYPE_INPUT] = 'competitive'
    delete next[SOURCE_FAQ_IDS_INPUT]
  } else {
    delete next[SOURCE_TYPE_INPUT]
    delete next[SOURCE_B2B_DISPLACEMENT_VENDORS_INPUT]
  }
  return { ok: true, value: `${JSON.stringify(next, null, 2)}\n` }
}

export function b2bDisplacementVendorsDraftValue(
  parsed: ParsedInputsJsonObject,
): string[] {
  if (!parsed.ok) return []
  const value = parsed.value[SOURCE_B2B_DISPLACEMENT_VENDORS_INPUT]
  if (Array.isArray(value)) return uniqueNonBlankStrings(value)
  if (typeof value === 'string') return uniqueNonBlankStrings([value])
  return []
}

export function updateB2BDisplacementVendorsInputJson(
  current: string,
  vendorNames: string[],
): UpdatedInputsJson {
  const parsed = parseInputsJsonObject(current)
  if (!parsed.ok) return parsed

  const next = { ...parsed.value }
  const values = uniqueNonBlankStrings(vendorNames)
  if (values.length === 0) {
    delete next[SOURCE_B2B_DISPLACEMENT_VENDORS_INPUT]
  } else {
    next[SOURCE_B2B_DISPLACEMENT_VENDORS_INPUT] = values
  }

  return { ok: true, value: `${JSON.stringify(next, null, 2)}\n` }
}

function uniqueNonBlankStrings(values: unknown[]): string[] {
  const seen = new Set<string>()
  const items: string[] = []
  for (const item of values) {
    if (typeof item === 'undefined' || item === null) continue
    const trimmed = String(item).trim()
    if (trimmed === '' || seen.has(trimmed)) continue
    seen.add(trimmed)
    items.push(trimmed)
  }
  return items
}

export {
  SOURCE_B2B_DISPLACEMENT_VENDORS_INPUT,
  SOURCE_FAQ_IDS_INPUT,
  SOURCE_MATERIAL_TYPE_INPUT,
  SOURCE_TYPE_INPUT,
}
