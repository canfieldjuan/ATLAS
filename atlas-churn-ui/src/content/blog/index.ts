type ChartValue = string | number | null | undefined
type ChartDatum = Record<string, ChartValue>

interface ChartSeries {
  dataKey: string
  color?: string
}

interface ChartConfig {
  bars?: ChartSeries[]
  x_key?: string
  [key: string]: unknown
}

interface AffiliatePartner {
  name?: string
  product_name?: string
  slug?: string
}

interface BlogDataContext {
  affiliate_url?: string
  affiliate_partner?: AffiliatePartner
  [key: string]: unknown
}

export interface ChartSpec {
  chart_id: string
  chart_type: 'bar' | 'horizontal_bar' | 'radar' | 'line'
  title: string
  data: ChartDatum[]
  config: ChartConfig
}

export interface FaqItem {
  question: string
  answer: string
}

/**
 * Canonical topic_type values. One per blueprint function in
 * atlas_brain/autonomous/tasks/b2b_blog_post_generation.py. New
 * blueprint functions must add their topic_type value here too --
 * adding a new producer without updating this union surfaces as a
 * compile error in any post file using the new type, which is the
 * intended forcing function.
 *
 * Mapping (blueprint function -> topic_type):
 *   _blueprint_vendor_alternative      -> 'vendor_alternative'
 *   _blueprint_vendor_showdown         -> 'vendor_showdown'
 *   _blueprint_churn_report            -> 'churn_report'
 *   _blueprint_migration_guide         -> 'migration_guide'
 *   _blueprint_vendor_deep_dive        -> 'vendor_deep_dive'
 *   _blueprint_market_landscape        -> 'market_landscape'
 *   _blueprint_pricing_reality_check   -> 'pricing_reality_check'
 *   _blueprint_switching_story         -> 'switching_story'
 *   _blueprint_pain_point_roundup      -> 'pain_point_roundup'
 *   _blueprint_best_fit_guide          -> 'best_fit_guide'
 */
export type BlogTopicType =
  | 'vendor_alternative'
  | 'vendor_showdown'
  | 'churn_report'
  | 'migration_guide'
  | 'vendor_deep_dive'
  | 'market_landscape'
  | 'pricing_reality_check'
  | 'switching_story'
  | 'pain_point_roundup'
  | 'best_fit_guide'

const VALID_TOPIC_TYPES: ReadonlySet<string> = new Set<BlogTopicType>([
  'vendor_alternative',
  'vendor_showdown',
  'churn_report',
  'migration_guide',
  'vendor_deep_dive',
  'market_landscape',
  'pricing_reality_check',
  'switching_story',
  'pain_point_roundup',
  'best_fit_guide',
])

/**
 * Coerce arbitrary input (typically from a draft API or a JSON parse)
 * to a BlogTopicType, returning undefined for anything else. Use this at
 * boundaries where untyped data crosses into the strict BlogPost type.
 */
export function coerceTopicType(value: unknown): BlogTopicType | undefined {
  if (typeof value !== 'string') return undefined
  return VALID_TOPIC_TYPES.has(value) ? (value as BlogTopicType) : undefined
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
  topic_type?: BlogTopicType
  data_context?: BlogDataContext
  seo_title?: string
  seo_description?: string
  target_keyword?: string
  secondary_keywords?: string[]
  faq?: FaqItem[]
  related_slugs?: string[]
  cta?: {
    headline: string
    body: string
    button_text: string
    report_type: string
    vendor_filter?: string | null
    category_filter?: string | null
  } | null
}

type BlogPostLoader = () => Promise<BlogPost>

const postLoaders = import.meta.glob('./*-20*.ts', { import: 'default' }) as Record<string, BlogPostLoader>
const postCache = new Map<string, Promise<BlogPost | null>>()
let allPostsPromise: Promise<BlogPost[]> | null = null

function sortPosts(posts: BlogPost[]): BlogPost[] {
  return [...posts].sort((left, right) => {
    if (left.date === right.date) return left.slug.localeCompare(right.slug)
    return right.date.localeCompare(left.date)
  })
}

function postPathForSlug(slug: string): string | null {
  const path = `./${slug}.ts`
  return path in postLoaders ? path : null
}

export async function loadPostBySlug(slug: string): Promise<BlogPost | null> {
  const cached = postCache.get(slug)
  if (cached) return cached

  const path = postPathForSlug(slug)
  if (!path) return null

  const promise = postLoaders[path]()
    .then((post) => post || null)
    .catch(() => null)
  postCache.set(slug, promise)
  return promise
}

export async function loadPostsBySlugs(slugs: string[]): Promise<BlogPost[]> {
  const uniqueSlugs = Array.from(new Set(slugs.filter(Boolean)))
  const posts = await Promise.all(uniqueSlugs.map((slug) => loadPostBySlug(slug)))
  return posts.filter((post): post is BlogPost => !!post)
}

export async function loadAllPosts(): Promise<BlogPost[]> {
  if (!allPostsPromise) {
    const paths = Object.keys(postLoaders).sort()
    allPostsPromise = Promise.all(paths.map((path) => postLoaders[path]())).then(sortPosts)
  }
  return allPostsPromise
}
