import type { ContentOpsInputContractView } from './types'

export interface ContentOpsInputDisplayFallback {
  label: string
  placeholder: string
}

export function inputContractDisplay(
  contract: Pick<ContentOpsInputContractView, 'label' | 'placeholder'> | undefined,
  fallback: ContentOpsInputDisplayFallback,
): ContentOpsInputDisplayFallback {
  return {
    label: contract?.label ?? fallback.label,
    placeholder: contract?.placeholder ?? fallback.placeholder,
  }
}
