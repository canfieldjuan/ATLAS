import CampaignQualityTrends from './CampaignQualityTrends'
import type { BlogQualityTrends as BlogQualityTrendsData } from '../types'

interface BlogQualityTrendsProps {
  data?: BlogQualityTrendsData | null
  loading?: boolean
  title?: string
}

export default function BlogQualityTrends({
  data,
  loading = false,
  title = 'Blog Quality Trends',
}: BlogQualityTrendsProps) {
  return (
    <CampaignQualityTrends
      data={data}
      loading={loading}
      title={title}
    />
  )
}
