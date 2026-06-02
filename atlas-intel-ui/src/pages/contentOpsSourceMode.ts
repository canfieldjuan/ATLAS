const SOURCE_TYPE_INPUT = 'source_type'
const SOURCE_MATERIAL_TYPE_INPUT = 'source_material_type'
const SOURCE_FAQ_IDS_INPUT = 'source_faq_ids'

export type ContentOpsSourceMode = 'support_ticket' | 'reviews'

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
  } else {
    delete next[SOURCE_TYPE_INPUT]
  }
  return { ok: true, value: `${JSON.stringify(next, null, 2)}\n` }
}

export {
  SOURCE_FAQ_IDS_INPUT,
  SOURCE_MATERIAL_TYPE_INPUT,
  SOURCE_TYPE_INPUT,
}
