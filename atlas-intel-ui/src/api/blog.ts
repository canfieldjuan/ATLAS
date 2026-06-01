import { API_BASE } from './config'
import type { BlogPost, ChartSpec, FaqItem } from '../content/blog'

const BASE = `${API_BASE}/api/v1/blog`

interface PublicBlogPostWire {
  slug?: string
  title?: string
  description?: string
  date?: string
  author?: string
  tags?: unknown
  content?: string
  charts?: unknown
  topic_type?: string
  data_context?: Record<string, unknown>
  seo_title?: string | null
  seo_description?: string | null
  target_keyword?: string | null
  secondary_keywords?: unknown
  faq?: unknown
  related_slugs?: unknown
}

interface PublicBlogListWire {
  posts?: PublicBlogPostWire[]
  total?: number
}

interface PublicBlogDetailWire {
  post?: PublicBlogPostWire | null
}

export async function fetchPublicBlogPosts(): Promise<BlogPost[]> {
  const response = await getPublicBlogJson<PublicBlogListWire>('/published')
  return Array.isArray(response.posts)
    ? response.posts
        .map(publicBlogPostFromWire)
        .filter((post): post is BlogPost => Boolean(post))
    : []
}

export async function fetchPublicBlogPost(slug: string): Promise<BlogPost | null> {
  const response = await getPublicBlogJson<PublicBlogDetailWire>(
    `/published/${encodeURIComponent(slug)}`,
  )
  return response.post ? publicBlogPostFromWire(response.post) : null
}

function publicBlogPostFromWire(value: PublicBlogPostWire): BlogPost | null {
  const slug = textValue(value.slug)
  const title = textValue(value.title)
  const content = textValue(value.content)
  if (!slug || !title || !content) return null
  return {
    slug,
    title,
    description: textValue(value.description) || title,
    date: dateValue(value.date),
    author: textValue(value.author) || 'Atlas Intelligence',
    tags: stringList(value.tags),
    content,
    charts: chartList(value.charts),
    topic_type: textValue(value.topic_type) || undefined,
    data_context: value.data_context ?? undefined,
    seo_title: textValue(value.seo_title) || undefined,
    seo_description: textValue(value.seo_description) || undefined,
    target_keyword: textValue(value.target_keyword) || undefined,
    secondary_keywords: stringList(value.secondary_keywords),
    faq: faqList(value.faq),
    related_slugs: stringList(value.related_slugs),
  }
}

function maybeFallbackApiPath(url: string): string | null {
  if (!url.includes('/api/v1/')) return null
  return url.replace('/api/v1/', '/api/')
}

async function getPublicBlogJson<T>(path: string): Promise<T> {
  const url = `${BASE}${path}`
  let response = await fetch(url)
  if (response.status === 404) {
    const fallback = maybeFallbackApiPath(url)
    if (fallback) response = await fetch(fallback)
  }
  if (!response.ok) {
    const text = await response.text().catch(() => '')
    throw new Error(`API ${response.status}: ${text || response.statusText}`)
  }
  return response.json() as Promise<T>
}

function textValue(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function dateValue(value: unknown): string {
  const text = textValue(value)
  return text ? text.slice(0, 10) : new Date().toISOString().slice(0, 10)
}

function stringList(value: unknown): string[] {
  if (Array.isArray(value)) return value.map(textValue).filter(Boolean)
  const text = textValue(value)
  return text ? text.split(',').map((item) => item.trim()).filter(Boolean) : []
}

function chartList(value: unknown): ChartSpec[] {
  return Array.isArray(value) ? value.filter(isChartSpec) : []
}

function isChartSpec(value: unknown): value is ChartSpec {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false
  const chart = value as Record<string, unknown>
  const chartType = textValue(chart.chart_type)
  return Boolean(
    textValue(chart.chart_id) &&
      ['bar', 'horizontal_bar', 'radar', 'line'].includes(chartType) &&
      textValue(chart.title) &&
      Array.isArray(chart.data) &&
      chart.config &&
      typeof chart.config === 'object' &&
      !Array.isArray(chart.config),
  )
}

function faqList(value: unknown): FaqItem[] {
  if (!Array.isArray(value)) return []
  return value
    .map((item) => {
      if (!item || typeof item !== 'object' || Array.isArray(item)) return null
      const record = item as Record<string, unknown>
      const question = textValue(record.question)
      const answer = textValue(record.answer)
      return question && answer ? { question, answer } : null
    })
    .filter((item): item is FaqItem => Boolean(item))
}
