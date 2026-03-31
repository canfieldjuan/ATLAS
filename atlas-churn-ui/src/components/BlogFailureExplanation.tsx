import CampaignFailureExplanation from './CampaignFailureExplanation'
import type { CampaignFailureExplanation as BlogFailureExplanationData } from '../types'

interface BlogFailureExplanationProps {
  explanation?: BlogFailureExplanationData | null
  boundaryLabel?: string | null
}

export default function BlogFailureExplanation({
  explanation,
  boundaryLabel,
}: BlogFailureExplanationProps) {
  return (
    <CampaignFailureExplanation
      explanation={explanation}
      boundaryLabel={boundaryLabel}
    />
  )
}
