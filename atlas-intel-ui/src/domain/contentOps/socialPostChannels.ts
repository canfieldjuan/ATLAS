import type { ContentOpsRequest } from './types'

export const SOCIAL_POST_OUTPUT = 'social_post' as const

// Mirror extracted_content_pipeline/social_post_generation.py.
export const SOCIAL_POST_CHANNEL_OPTIONS = [
  { id: 'linkedin', label: 'LinkedIn' },
  { id: 'x', label: 'X' },
  { id: 'facebook', label: 'Facebook' },
  { id: 'instagram', label: 'Instagram' },
  { id: 'threads', label: 'Threads' },
] as const

export type SocialPostChannelId =
  (typeof SOCIAL_POST_CHANNEL_OPTIONS)[number]['id']

export const DEFAULT_SOCIAL_POST_CHANNELS: SocialPostChannelId[] = [
  'linkedin',
]

const SOCIAL_POST_CHANNEL_IDS = new Set<string>(
  SOCIAL_POST_CHANNEL_OPTIONS.map((option) => option.id),
)

export function normalizeSocialPostChannels(
  channels: readonly string[] | null | undefined,
): SocialPostChannelId[] {
  const normalized: SocialPostChannelId[] = []
  for (const channel of channels ?? []) {
    if (
      SOCIAL_POST_CHANNEL_IDS.has(channel) &&
      !normalized.includes(channel as SocialPostChannelId)
    ) {
      normalized.push(channel as SocialPostChannelId)
    }
  }
  return normalized.length > 0
    ? normalized
    : [...DEFAULT_SOCIAL_POST_CHANNELS]
}

export function requestWithSocialPostChannels(
  request: ContentOpsRequest,
  channels: readonly string[],
): ContentOpsRequest {
  const inputs = { ...request.inputs }
  delete inputs.social_post_channels

  if (!request.outputs.includes(SOCIAL_POST_OUTPUT)) {
    delete inputs.social_channels
    return { ...request, inputs }
  }

  return {
    ...request,
    inputs: {
      ...inputs,
      social_channels: normalizeSocialPostChannels(channels),
    },
  }
}
