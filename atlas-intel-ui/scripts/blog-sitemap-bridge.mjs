function clean(value) {
  return String(value || '').trim()
}

function stripTrailingSlash(value) {
  return value.replace(/\/+$/, '')
}

export function resolveGeneratedBlogPostsUrl(env = process.env) {
  const configured = clean(env.VITE_PUBLIC_BLOG_POSTS_URL)
  if (configured) return configured

  const apiBase = stripTrailingSlash(clean(env.VITE_API_BASE))
  return apiBase ? `${apiBase}/api/v1/blog/published?limit=200` : ''
}

function recordValue(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`Generated blog ${label} must be an object`)
  }
  return value
}

function requiredText(record, field, label) {
  const value = clean(record[field])
  if (!value) throw new Error(`Generated blog ${label} missing ${field}`)
  return value
}

function optionalText(record, field) {
  return clean(record[field])
}

function slugValue(record, label) {
  const slug = requiredText(record, 'slug', label)
  if (slug.includes('/') || slug.includes('\\') || slug.includes('..')) {
    throw new Error(`Generated blog ${label} has unsafe slug`)
  }
  return slug
}

function dateValue(record, label) {
  const date = requiredText(record, 'date', label).slice(0, 10)
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) {
    throw new Error(`Generated blog ${label} has invalid date`)
  }
  return date
}

function stringList(value, label, field) {
  if (value === null || value === undefined || value === '') return []
  if (Array.isArray(value)) return value.map(clean).filter(Boolean)
  if (typeof value === 'string') {
    return value.split(',').map(item => item.trim()).filter(Boolean)
  }
  throw new Error(`Generated blog ${label} ${field} must be an array or string`)
}

function chartList(value, label) {
  if (value === null || value === undefined) return []
  if (!Array.isArray(value)) throw new Error(`Generated blog ${label} charts must be an array`)
  return value.map((item, index) => {
    const chart = recordValue(item, `${label} chart ${index + 1}`)
    const chartType = requiredText(chart, 'chart_type', `${label} chart ${index + 1}`)
    if (!['bar', 'horizontal_bar', 'radar', 'line'].includes(chartType)) {
      throw new Error(`Generated blog ${label} chart ${index + 1} has invalid chart_type`)
    }
    if (!Array.isArray(chart.data)) {
      throw new Error(`Generated blog ${label} chart ${index + 1} data must be an array`)
    }
    recordValue(chart.config, `${label} chart ${index + 1} config`)
    return {
      chart_id: requiredText(chart, 'chart_id', `${label} chart ${index + 1}`),
      chart_type: chartType,
      title: requiredText(chart, 'title', `${label} chart ${index + 1}`),
      data: chart.data,
      config: chart.config,
    }
  })
}

function faqList(value, label) {
  if (value === null || value === undefined) return []
  if (!Array.isArray(value)) throw new Error(`Generated blog ${label} faq must be an array`)
  return value.map((item, index) => {
    const faq = recordValue(item, `${label} faq ${index + 1}`)
    return {
      question: requiredText(faq, 'question', `${label} faq ${index + 1}`),
      answer: requiredText(faq, 'answer', `${label} faq ${index + 1}`),
    }
  })
}

function generatedBlogPostFromWire(value, index) {
  const label = `post ${index + 1}`
  const record = recordValue(value, label)
  const robots = optionalText(record, 'robots').toLowerCase()
  if (robots.includes('noindex')) return null
  const title = requiredText(record, 'title', label)
  return {
    slug: slugValue(record, label),
    title,
    description: optionalText(record, 'description') || title,
    date: dateValue(record, label),
    author: optionalText(record, 'author') || 'Atlas Intelligence',
    tags: stringList(record.tags, label, 'tags'),
    content: requiredText(record, 'content', label),
    charts: chartList(record.charts, label),
    faq: faqList(record.faq, label),
    seoTitle: optionalText(record, 'seo_title') || optionalText(record, 'seoTitle'),
    seoDescription:
      optionalText(record, 'seo_description') || optionalText(record, 'seoDescription'),
  }
}

function blogLoc(publicSiteUrl, slug) {
  const siteBase = new URL(stripTrailingSlash(clean(publicSiteUrl)))
  return `${siteBase.origin}/blog/${encodeURIComponent(slug)}`
}

function warn(logger, message) {
  if (logger && typeof logger.warn === 'function') {
    logger.warn(message)
  }
}

async function fetchGeneratedBlogPosts({
  postsUrl,
  fetchImpl = globalThis.fetch,
  logger = console,
} = {}) {
  const feedUrl = clean(postsUrl)
  if (!feedUrl) return []
  if (typeof fetchImpl !== 'function') {
    warn(logger, 'Skipping generated blog feed: fetch is unavailable')
    return []
  }

  let envelope
  try {
    const response = await fetchImpl(feedUrl)
    if (!response || !response.ok) {
      const status = response?.status ? `HTTP ${response.status}` : 'no response'
      throw new Error(`Failed to fetch generated blog posts: ${status}`)
    }
    envelope = recordValue(await response.json(), 'response')
    if (!Array.isArray(envelope.posts)) {
      throw new Error('Generated blog response posts must be an array')
    }
  } catch (error) {
    warn(logger, `Skipping generated blog feed: ${error.message}`)
    return []
  }

  const posts = []
  for (const [index, item] of envelope.posts.entries()) {
    try {
      const post = generatedBlogPostFromWire(item, index)
      if (post) posts.push(post)
    } catch (error) {
      warn(logger, `Skipping generated blog post ${index + 1}: ${error.message}`)
    }
  }
  return posts
}

function excludeSlugSet(excludeSlugs) {
  return new Set((excludeSlugs || []).map(clean).filter(Boolean))
}

export async function fetchGeneratedBlogSitemapUrls({
  postsUrl,
  publicSiteUrl,
  excludeSlugs = [],
  fetchImpl = globalThis.fetch,
  logger = console,
} = {}) {
  const excluded = excludeSlugSet(excludeSlugs)
  const seen = new Set()
  const posts = await fetchGeneratedBlogPosts({ postsUrl, fetchImpl, logger })
  const urls = []
  for (const post of posts) {
    if (excluded.has(post.slug)) continue
    try {
      const loc = blogLoc(publicSiteUrl, post.slug)
      if (seen.has(loc)) continue
      seen.add(loc)
      urls.push({
        loc,
        lastmod: post.date,
        priority: '0.7',
        changefreq: 'monthly',
      })
    } catch (error) {
      warn(logger, `Skipping generated blog sitemap URL for ${post.slug}: ${error.message}`)
    }
  }
  return urls
}

export async function fetchGeneratedBlogPrerenderEntries({
  postsUrl,
  publicSiteUrl,
  excludeSlugs = [],
  fetchImpl = globalThis.fetch,
  logger = console,
} = {}) {
  const excluded = excludeSlugSet(excludeSlugs)
  const seen = new Set()
  const posts = await fetchGeneratedBlogPosts({ postsUrl, fetchImpl, logger })
  const entries = []
  for (const post of posts) {
    if (excluded.has(post.slug) || seen.has(post.slug)) continue
    seen.add(post.slug)
    try {
      entries.push({
        path: `/blog/${post.slug}`,
        loc: blogLoc(publicSiteUrl, post.slug),
        post,
      })
    } catch (error) {
      warn(logger, `Skipping generated blog prerender entry for ${post.slug}: ${error.message}`)
    }
  }
  return entries
}
