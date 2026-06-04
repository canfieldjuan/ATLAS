import type {
  ContentOpsBrandVoiceProfile,
  UpsertContentOpsBrandVoiceProfileRequest,
} from '../../api/contentOps'

export type BrandVoiceProfileEditorState = {
  mode: 'create' | 'edit'
  profileId: string | null
  name: string
  descriptorsText: string
  exemplarsText: string
  bannedTermsText: string
  preferredPov: string
  readingLevel: string
  metadata: Record<string, unknown>
}

export type BrandVoiceProfileEditorPatch = Partial<
  Omit<BrandVoiceProfileEditorState, 'mode' | 'profileId'>
>

export function blankBrandVoiceProfileEditorState(): BrandVoiceProfileEditorState {
  return {
    mode: 'create',
    profileId: null,
    name: '',
    descriptorsText: '',
    exemplarsText: '',
    bannedTermsText: '',
    preferredPov: '',
    readingLevel: '',
    metadata: { source: 'content_ops_ui' },
  }
}

export function brandVoiceProfileEditorStateFromProfile(
  profile: ContentOpsBrandVoiceProfile,
): BrandVoiceProfileEditorState {
  return {
    mode: 'edit',
    profileId: profile.id,
    name: profile.name,
    descriptorsText: brandVoiceProfileListText(profile.descriptors),
    exemplarsText: brandVoiceProfileListText(profile.exemplars),
    bannedTermsText: brandVoiceProfileListText(profile.banned_terms),
    preferredPov: profile.preferred_pov ?? '',
    readingLevel: profile.reading_level ?? '',
    metadata: { ...profile.metadata },
  }
}

export function brandVoiceProfileEditorRequest(
  editor: BrandVoiceProfileEditorState,
): UpsertContentOpsBrandVoiceProfileRequest {
  const preferredPov = cleanOptionalText(editor.preferredPov)
  const readingLevel = cleanOptionalText(editor.readingLevel)
  return {
    name: editor.name.trim(),
    descriptors: brandVoiceProfileListItems(editor.descriptorsText, 8),
    exemplars: brandVoiceProfileListItems(editor.exemplarsText, 3),
    banned_terms: brandVoiceProfileListItems(editor.bannedTermsText, 20),
    preferred_pov: preferredPov,
    reading_level: readingLevel,
    metadata: { ...editor.metadata },
  }
}

export function canSaveBrandVoiceProfileEditor(
  editor: BrandVoiceProfileEditorState,
): boolean {
  if (!editor.name.trim()) return false
  const body = brandVoiceProfileEditorRequest(editor)
  return Boolean(
    (body.descriptors?.length ?? 0) > 0 ||
      (body.exemplars?.length ?? 0) > 0 ||
      (body.banned_terms?.length ?? 0) > 0 ||
      body.preferred_pov ||
      body.reading_level,
  )
}

export function deriveBrandVoiceProfileEditorPatch(
  sampleText: string,
  options: { fallbackName?: string } = {},
): BrandVoiceProfileEditorPatch {
  const normalized = normalizeSampleText(sampleText)
  const sentences = sampleSentences(normalized)
  return {
    name: brandVoiceSampleName(options.fallbackName ?? ''),
    descriptorsText: sampleDescriptors(normalized, sentences).join('\n'),
    exemplarsText: sampleExemplars(sentences, normalized).join('\n'),
    bannedTermsText: '',
    preferredPov: samplePreferredPov(normalized),
    readingLevel: sampleReadingLevel(sentences),
    metadata: {
      source: 'content_ops_ui_sample',
      sample_character_count: normalized.length,
    },
  }
}

export function applyBrandVoiceProfileEditorPatch(
  editor: BrandVoiceProfileEditorState,
  patch: BrandVoiceProfileEditorPatch,
): BrandVoiceProfileEditorState {
  return {
    ...editor,
    name: editor.name.trim() ? editor.name : patch.name ?? editor.name,
    descriptorsText: editor.descriptorsText.trim()
      ? editor.descriptorsText
      : patch.descriptorsText ?? editor.descriptorsText,
    exemplarsText: editor.exemplarsText.trim()
      ? editor.exemplarsText
      : patch.exemplarsText ?? editor.exemplarsText,
    bannedTermsText: editor.bannedTermsText.trim()
      ? editor.bannedTermsText
      : patch.bannedTermsText ?? editor.bannedTermsText,
    preferredPov: editor.preferredPov.trim()
      ? editor.preferredPov
      : patch.preferredPov ?? editor.preferredPov,
    readingLevel: editor.readingLevel.trim()
      ? editor.readingLevel
      : patch.readingLevel ?? editor.readingLevel,
    metadata: { ...editor.metadata, ...patch.metadata },
  }
}

export function brandVoiceProfileListItems(text: string, limit: number): string[] {
  return text
    .split(/\r?\n/)
    .map((item) => item.trim())
    .filter(Boolean)
    .slice(0, limit)
}

function brandVoiceProfileListText(values: string[]): string {
  return values.join('\n')
}

function cleanOptionalText(value: string): string | null {
  const cleaned = value.trim()
  return cleaned ? cleaned : null
}

function normalizeSampleText(text: string): string {
  return text.replace(/\s+/g, ' ').trim()
}

function sampleSentences(text: string): string[] {
  return text
    .split(/(?<=[.!?])\s+/)
    .map((sentence) => sentence.trim())
    .filter((sentence) => sentence.length >= 24)
}

function sampleExemplars(sentences: string[], normalized: string): string[] {
  const candidates = sentences.length > 0 ? sentences : [normalized]
  return candidates
    .map((sentence) => sentence.slice(0, 280).trim())
    .filter(Boolean)
    .slice(0, 3)
}

function sampleDescriptors(text: string, sentences: string[]): string[] {
  const lower = text.toLowerCase()
  const descriptors: string[] = []
  const avgWords = averageSentenceWords(sentences)
  if (avgWords > 0 && avgWords <= 16) descriptors.push('concise')
  if (/\b(you|your|customers?|teams?)\b/.test(lower)) {
    descriptors.push('customer-focused')
  }
  if (/\b(can't|don't|won't|we're|you're|it's|they're)\b/.test(lower)) {
    descriptors.push('conversational')
  }
  if (/\b(api|data|security|workflow|integration|automation)\b/.test(lower)) {
    descriptors.push('technical')
  }
  if (/[!?]/.test(text)) descriptors.push('energetic')
  if (descriptors.length === 0) descriptors.push('plainspoken')
  return [...new Set(descriptors)].slice(0, 8)
}

function samplePreferredPov(text: string): string {
  const lower = text.toLowerCase()
  const second = countMatches(lower, /\b(you|your|yours)\b/g)
  const first = countMatches(lower, /\b(we|our|ours|us)\b/g)
  const third = countMatches(lower, /\b(they|their|customer|customers|team|teams)\b/g)
  if (second >= first && second >= third && second > 0) return 'second_person'
  if (first >= third && first > 0) return 'first_person'
  if (third > 0) return 'third_person'
  return ''
}

function sampleReadingLevel(sentences: string[]): string {
  const avgWords = averageSentenceWords(sentences)
  if (avgWords === 0) return ''
  return avgWords <= 18 ? 'plain' : 'detailed'
}

function averageSentenceWords(sentences: string[]): number {
  if (sentences.length === 0) return 0
  const totalWords = sentences.reduce((total, sentence) => {
    return total + sentence.split(/\s+/).filter(Boolean).length
  }, 0)
  return totalWords / sentences.length
}

function countMatches(text: string, pattern: RegExp): number {
  return text.match(pattern)?.length ?? 0
}

function brandVoiceSampleName(name: string): string {
  const cleaned = name
    .replace(/\.[^.]+$/, '')
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
  return cleaned
}
