import migrationFromFreshdesk202603 from './migration-from-freshdesk-2026-03'
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
  data_context?: Record<string, any>
}

export const POSTS: BlogPost[] = [
  migrationFromFreshdesk202603,
].sort((a, b) => b.date.localeCompare(a.date))
