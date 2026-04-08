import shopifyDeepDive202604 from './shopify-deep-dive-2026-04'
import whyTeamsLeaveAzure202604 from './why-teams-leave-azure-2026-04'
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
  return [...posts  shopifyDeepDive202604,
  whyTeamsLeaveAzure202604,
].sort((left, right) => {
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
