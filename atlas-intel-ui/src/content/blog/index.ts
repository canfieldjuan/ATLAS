export interface ChartSpec {
  chart_id: string
  chart_type: 'bar' | 'horizontal_bar' | 'radar' | 'line'
  title: string
  data: Record<string, any>[]
  config: Record<string, any>
}

export interface BlogPost {
  slug: string
  title: string
  description: string
  date: string
  author: string
  tags: string[]
  content: string
  charts?: ChartSpec[]
  topic_type?: string
}

import amazonReviewMonitoringTools from './amazon-review-monitoring-tools'

export const POSTS: BlogPost[] = [
  amazonReviewMonitoringTools,
].sort((a, b) => b.date.localeCompare(a.date))
