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

import amazonReviewMonitoringTools from './amazon-review-monitoring-tools-2026-03'
import migrationComputerAccessoriesPeripherals202603 from './migration-computer-accessories-peripherals-2026-03'
import migrationComputerComponents202603 from './migration-computer-components-2026-03'

export const POSTS: BlogPost[] = [
  amazonReviewMonitoringTools,
  migrationComputerAccessoriesPeripherals202603,
  migrationComputerComponents202603,
].sort((a, b) => b.date.localeCompare(a.date))
