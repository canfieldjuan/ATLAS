export interface BlogPost {
  slug: string
  title: string
  description: string
  date: string
  author: string
  tags: string[]
  content: string
}

import amazonReviewMonitoringTools from './amazon-review-monitoring-tools'

export const POSTS: BlogPost[] = [
  amazonReviewMonitoringTools,
].sort((a, b) => b.date.localeCompare(a.date))
