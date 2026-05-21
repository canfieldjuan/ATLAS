export type ChartValue = string | number | null | undefined
export type ChartDatum = Record<string, ChartValue>

export interface ChartSeries {
  dataKey: string
}

export interface ChartSpec {
  chart_id: string
  title: string
  data: ChartDatum[]
  config: {
    bars?: ChartSeries[]
    x_key?: string
  }
}

export interface BlogSourceMetadata {
  file: string
  slug: string
  title: string
  description: string
  date: string
  author: string
  seoTitle: string
  seoDescription: string
  content: string
  charts: ChartSpec[]
}

export function chartPlaceholderIds(content: string): string[]
export function collectBlogSourceMetadata(rootDir: string): BlogSourceMetadata[]
