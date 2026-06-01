import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { marked } from 'marked'
import {
  existsSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
} from 'node:fs'
import { resolve, join } from 'node:path'
import {
  chartPlaceholderIds,
  collectBlogSourceMetadata,
  type ChartSpec,
  type ChartValue,
  type FaqItem,
} from './scripts/blog-source-metadata.mjs'
import {
  fetchLandingPagePrerenderEntries,
  fetchLandingPageSitemapUrls,
  resolveLandingPagePublicApiBase,
  resolveLandingPageSitemapUrl,
} from './scripts/landing-page-sitemap-bridge.mjs'
import {
  fetchGeneratedBlogPrerenderEntries,
  fetchGeneratedBlogSitemapUrls,
  resolveGeneratedBlogPostsUrl,
} from './scripts/blog-sitemap-bridge.mjs'

const BASE_URL = 'https://atlas-intel-ui-two.vercel.app'
const DEFAULT_OG_IMAGE = `${BASE_URL}/og-default.png`
const ATLAS_SAME_AS = [
  'https://twitter.com/atlasintel',
  'https://www.linkedin.com/company/atlas-intelligence',
]

interface SitemapUrl {
  loc: string
  lastmod?: string
  priority: string
  changefreq: string
}

function dedupeSitemapUrls(urls: SitemapUrl[]): SitemapUrl[] {
  const seen = new Set<string>()
  return urls.filter(url => {
    if (seen.has(url.loc)) return false
    seen.add(url.loc)
    return true
  })
}

// ---------------------------------------------------------------------------
// Sitemap plugin
// ---------------------------------------------------------------------------
function sitemapPlugin() {
  return {
    name: 'generate-sitemap',
    async closeBundle() {
      const posts = collectBlogSourceMetadata(import.meta.dirname)
      const staticBlogSlugs = posts.map(post => post.slug)
      const generatedBlogPostsUrl = resolveGeneratedBlogPostsUrl()
      const generatedBlogUrls = await fetchGeneratedBlogSitemapUrls({
        postsUrl: generatedBlogPostsUrl,
        publicSiteUrl: BASE_URL,
        excludeSlugs: staticBlogSlugs,
      })
      const generatedLandingPageUrls = await fetchLandingPageSitemapUrls({
        sitemapUrl: resolveLandingPageSitemapUrl(),
        publicSiteUrl: BASE_URL,
      })

      const today = new Date().toISOString().split('T')[0]
      const urls = dedupeSitemapUrls([
        { loc: `${BASE_URL}/landing`, priority: '1.0', changefreq: 'weekly' },
        { loc: `${BASE_URL}/blog`, priority: '0.9', changefreq: 'daily' },
        ...posts.map(post => ({
          loc: `${BASE_URL}/blog/${post.slug}`,
          lastmod: post.date,
          priority: '0.7',
          changefreq: 'monthly',
        })),
        ...generatedBlogUrls,
        ...generatedLandingPageUrls,
        { loc: `${BASE_URL}/`, priority: '0.3', changefreq: 'monthly' },
      ])

      const xml = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${urls.map(u => `  <url>
    <loc>${u.loc}</loc>
    <lastmod>${u.lastmod || today}</lastmod>
    <changefreq>${u.changefreq}</changefreq>
    <priority>${u.priority}</priority>
  </url>`).join('\n')}
</urlset>
`
      const distDir = resolve(import.meta.dirname, 'dist')
      if (existsSync(distDir)) {
        writeFileSync(join(distDir, 'sitemap.xml'), xml)
        console.log(`  Sitemap generated with ${urls.length} URLs`)
      }
    },
  }
}

// ---------------------------------------------------------------------------
// Pre-render plugin
//
// Strategy: at build time, clone dist/index.html into per-route directories
// with the correct static <head> meta baked in. Crawlers (Googlebot, GPTBot,
// PerplexityBot, ClaudeBot) fetch the HTML file directly and see proper title,
// description, canonical, og:image, og:type, and JSON-LD without executing JS.
//
// Authenticated routes stay as the regular SPA (index.html fallback).
// Only public routes are pre-rendered: /landing, /blog, /blog/:slug.
// ---------------------------------------------------------------------------

interface PrerenderedRoute {
  path: string        // URL path, e.g. /blog/my-slug
  title: string
  description: string
  canonical: string
  ogType: string
  jsonLd?: object
  bodyHtml?: string
  robots?: string
}

interface PublicLandingPagePrerenderEntry {
  path: string
  loc: string
  page: Record<string, unknown>
}

interface BlogPrerenderPost {
  slug: string
  title: string
  description: string
  date: string
  author: string
  content: string
  charts: ChartSpec[]
  faq: FaqItem[]
  seoTitle?: string
  seoDescription?: string
}

interface GeneratedBlogPrerenderEntry {
  path: string
  loc: string
  post: BlogPrerenderPost
}

interface LandingPageSectionView {
  id: string
  title: string
  body: string
}

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

function chartValue(value: ChartValue): string {
  if (value === null || value === undefined) return ''
  return String(value)
}

function buildChartFallbackHtml(chart: ChartSpec): string {
  const xKey = chart.config.x_key || 'name'
  const series = chart.config.bars?.map(item => item.dataKey) || []
  const headers = [xKey, ...series]
  const rows = chart.data
    .map(row => `          <tr>${headers
      .map(header => `<td>${escapeHtml(chartValue(row[header]))}</td>`)
      .join('')}</tr>`)
    .join('\n')

  return `<figure data-prerendered-chart="${escapeHtml(chart.chart_id)}">
        <figcaption>${escapeHtml(chart.title)}</figcaption>
        <table>
          <thead><tr>${headers.map(header => `<th>${escapeHtml(header)}</th>`).join('')}</tr></thead>
          <tbody>
${rows}
          </tbody>
        </table>
      </figure>`
}

function renderChartFallbacks(content: string, charts: ChartSpec[]): string {
  const chartMap = new Map(charts.map(chart => [chart.chart_id, chart]))
  const unknownChartIds = [...new Set(chartPlaceholderIds(content).filter(chartId => !chartMap.has(chartId)))]
  if (unknownChartIds.length) {
    throw new Error(`Missing chart fallback data for: ${unknownChartIds.join(', ')}`)
  }

  return content.replace(
    /<p>\s*\{\{chart:([^}]+)\}\}\s*<\/p>|\{\{chart:([^}]+)\}\}/g,
    (_match, htmlId: string | undefined, markdownId: string | undefined) => {
      const chartId = (htmlId || markdownId || '').trim()
      const chart = chartMap.get(chartId)
      if (!chart) {
        throw new Error(`Missing chart fallback data for: ${chartId}`)
      }
      return buildChartFallbackHtml(chart)
    },
  )
}

function escapeGeneratedMarkdownHtml(value: string): string {
  return value.replace(/[<>]/g, char => (char === '<' ? '&lt;' : '&gt;'))
}

function sanitizeGeneratedRenderedHtml(html: string): string {
  return html
    .replace(/\s(?:on[a-z]+|style)=(["']).*?\1/gi, '')
    .replace(/\s(href|src)=(["'])(.*?)\2/gi, (_match, attr: string, quote: string, value: string) => (
      safePublicHref(value) ? ` ${attr}=${quote}${value}${quote}` : ''
    ))
}

function buildFaqHtml(faq: FaqItem[]): string {
  if (!faq.length) return ''

  const items = faq
    .map(item => `        <div>
          <h3>${escapeHtml(item.question)}</h3>
          <p>${escapeHtml(item.answer)}</p>
        </div>`)
    .join('\n')

  return `
      <section data-prerendered-blog-faq="true">
        <h2>Frequently Asked Questions</h2>
${items}
      </section>`
}

function buildBlogBodyHtml(
  post: BlogPrerenderPost,
  { trustedHtml = true }: { trustedHtml?: boolean } = {},
): string {
  const content = trustedHtml ? post.content : escapeGeneratedMarkdownHtml(post.content)
  const renderedHtml = marked.parse(renderChartFallbacks(content, post.charts), { async: false }) as string
  const articleHtml = trustedHtml ? renderedHtml : sanitizeGeneratedRenderedHtml(renderedHtml)
  const faqHtml = buildFaqHtml(post.faq)
  return `
    <article data-prerendered-blog-article="true">
      <header>
        <h1>${escapeHtml(post.title)}</h1>
        <p>
          <time datetime="${escapeHtml(post.date)}">${escapeHtml(post.date)}</time>
          <span>${escapeHtml(post.author)}</span>
        </p>
      </header>
      <section data-prerendered-blog-content="true">
        ${articleHtml}
      </section>
      ${faqHtml}
    </article>`
}

function buildFaqJsonLd(faq: FaqItem[]): object | null {
  if (!faq.length) return null

  return {
    '@type': 'FAQPage',
    mainEntity: faq.map(item => ({
      '@type': 'Question',
      name: item.question,
      acceptedAnswer: {
        '@type': 'Answer',
        text: item.answer,
      },
    })),
  }
}

function buildHeadHtml(route: PrerenderedRoute): string {
  const { title, description, canonical, ogType, jsonLd } = route
  const lines = [
    `<meta name="description" content="${escapeHtml(description)}" />`,
    `<link rel="canonical" href="${escapeHtml(canonical)}" />`,
    `<meta property="og:title" content="${escapeHtml(title)}" />`,
    `<meta property="og:description" content="${escapeHtml(description)}" />`,
    `<meta property="og:url" content="${escapeHtml(canonical)}" />`,
    `<meta property="og:type" content="${escapeHtml(ogType)}" />`,
    `<meta property="og:site_name" content="Atlas Intelligence" />`,
    `<meta property="og:image" content="${DEFAULT_OG_IMAGE}" />`,
    `<meta property="og:image:width" content="1200" />`,
    `<meta property="og:image:height" content="630" />`,
    `<meta name="twitter:card" content="summary_large_image" />`,
    `<meta name="twitter:title" content="${escapeHtml(title)}" />`,
    `<meta name="twitter:description" content="${escapeHtml(description)}" />`,
    `<meta name="twitter:image" content="${DEFAULT_OG_IMAGE}" />`,
  ]
  if (route.robots) {
    lines.push(`<meta name="robots" content="${escapeHtml(route.robots)}" />`)
  }
  if (jsonLd) {
    lines.push(
      `<script type="application/ld+json">${jsonLdScriptContent(jsonLd)}</script>`,
    )
  }
  return lines.map(l => `    ${l}`).join('\n')
}

function jsonLdScriptContent(value: object): string {
  return JSON.stringify(value).replace(/</g, '\\u003c')
}

function recordValue(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null
}

function recordList(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value)
    ? value.filter((item): item is Record<string, unknown> =>
      Boolean(item && typeof item === 'object' && !Array.isArray(item)),
    )
    : []
}

function textValue(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function safePublicHref(value: string): string {
  const href = value.trim()
  const normalized = href.toLowerCase()
  if (
    normalized.startsWith('https://') ||
    normalized.startsWith('http://') ||
    normalized.startsWith('mailto:') ||
    normalized.startsWith('tel:') ||
    normalized.startsWith('/') ||
    normalized.startsWith('#')
  ) {
    return href
  }
  return ''
}

function landingPageSections(value: unknown): LandingPageSectionView[] {
  return recordList(value)
    .map((section) => ({
      id: textValue(section.id),
      title: textValue(section.title) || textValue(section.heading),
      body: textValue(section.body_markdown) || textValue(section.body),
    }))
    .filter((section) => section.title || section.body)
}

function paragraphHtml(value: string): string {
  return value
    .split(/\n{2,}/)
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => `<p>${escapeHtml(part).replace(/\n/g, '<br />')}</p>`)
    .join('\n')
}

function structuredDataWithCanonical(value: unknown, canonical: string): object | undefined {
  const raw = recordValue(value)
  if (!raw) return undefined
  const graph = recordList(raw['@graph'])
  if (graph.length === 0) return raw
  return {
    ...raw,
    '@graph': graph.map((node) => {
      const type = node['@type']
      if (type === 'WebPage') {
        return { ...node, '@id': `${canonical}#webpage`, url: canonical }
      }
      if (type === 'FAQPage') {
        return {
          ...node,
          '@id': `${canonical}#faq`,
          mainEntityOfPage: { '@id': `${canonical}#webpage` },
        }
      }
      return node
    }),
  }
}

function buildLandingPageBodyHtml(page: Record<string, unknown>): string {
  const hero = recordValue(page.hero)
  const cta = recordValue(page.cta)
  const sections = landingPageSections(page.sections)
  const title = textValue(hero?.headline) || textValue(page.title)
  const subheadline =
    textValue(hero?.subheadline) || textValue(page.value_prop)
  const persona = textValue(page.persona)
  const ctaLabel = textValue(cta?.label) || textValue(hero?.cta_label)
  const ctaUrl = safePublicHref(textValue(cta?.url) || textValue(hero?.cta_url))
  const sectionHtml = sections.map((section, index) => `
      <section data-prerendered-landing-page-section="${escapeHtml(section.id || String(index + 1))}">
        <h2>${escapeHtml(section.title || `Section ${index + 1}`)}</h2>
        ${paragraphHtml(section.body)}
      </section>`).join('\n')
  const ctaHtml = ctaLabel && ctaUrl
    ? `
      <p>
        <a href="${escapeHtml(ctaUrl)}">${escapeHtml(ctaLabel)}</a>
      </p>`
    : ''

  return `
    <article data-prerendered-landing-page="true">
      <header>
        ${persona ? `<p>${escapeHtml(persona)}</p>` : ''}
        <h1>${escapeHtml(title)}</h1>
        ${subheadline ? `<p>${escapeHtml(subheadline)}</p>` : ''}
        ${ctaHtml}
      </header>
      ${sectionHtml}
      ${ctaHtml ? `
      <section data-prerendered-landing-page-cta="true">
        <h2>${escapeHtml(textValue(page.title) || title)}</h2>
        ${subheadline ? `<p>${escapeHtml(subheadline)}</p>` : ''}
        ${ctaHtml}
      </section>` : ''}
    </article>`
}

function landingPageRoute(entry: PublicLandingPagePrerenderEntry): PrerenderedRoute {
  const page = entry.page
  const hero = recordValue(page.hero)
  const meta = recordValue(page.meta)
  const title = textValue(meta?.title_tag) || textValue(page.title) || 'Landing Page'
  const description =
    textValue(meta?.description) ||
    textValue(hero?.subheadline) ||
    textValue(page.value_prop) ||
    title
  return {
    path: entry.path,
    title,
    description,
    canonical: entry.loc,
    ogType: 'website',
    bodyHtml: buildLandingPageBodyHtml(page),
    jsonLd: structuredDataWithCanonical(page.structured_data, entry.loc),
    robots: textValue(page.robots) || 'noindex,follow',
  }
}

function blogPostRoute(
  post: BlogPrerenderPost,
  {
    path = `/blog/${post.slug}`,
    canonical = `${BASE_URL}/blog/${post.slug}`,
    trustedHtml = true,
  }: { path?: string; canonical?: string; trustedHtml?: boolean } = {},
): PrerenderedRoute {
  const seoTitle = post.seoTitle || post.title
  const seoDesc = post.seoDescription || post.description
  const faqJsonLd = buildFaqJsonLd(post.faq)
  const graph: object[] = [
    {
      '@type': 'BlogPosting',
      headline: seoTitle,
      description: seoDesc,
      datePublished: post.date,
      dateModified: post.date,
      image: DEFAULT_OG_IMAGE,
      author: {
        '@type': 'Organization',
        name: 'Atlas Intelligence',
        sameAs: ATLAS_SAME_AS,
      },
      publisher: {
        '@type': 'Organization',
        name: 'Atlas Intelligence',
        url: BASE_URL,
        sameAs: ATLAS_SAME_AS,
        logo: { '@type': 'ImageObject', url: DEFAULT_OG_IMAGE },
      },
      mainEntityOfPage: { '@type': 'WebPage', '@id': canonical },
    },
    {
      '@type': 'BreadcrumbList',
      itemListElement: [
        { '@type': 'ListItem', position: 1, name: 'Home', item: `${BASE_URL}/landing` },
        { '@type': 'ListItem', position: 2, name: 'Blog', item: `${BASE_URL}/blog` },
        { '@type': 'ListItem', position: 3, name: seoTitle, item: canonical },
      ],
    },
  ]
  if (faqJsonLd) graph.push(faqJsonLd)

  return {
    path,
    title: `${seoTitle} | Atlas Intelligence`,
    description: seoDesc,
    canonical,
    ogType: 'article',
    bodyHtml: buildBlogBodyHtml(post, { trustedHtml }),
    jsonLd: {
      '@context': 'https://schema.org',
      '@graph': graph,
    },
  }
}

function prerenderPlugin() {
  return {
    name: 'prerender-public-routes',
    async closeBundle() {
      const distDir = resolve(import.meta.dirname, 'dist')
      const indexHtmlPath = join(distDir, 'index.html')
      if (!existsSync(indexHtmlPath)) return

      const baseHtml = readFileSync(indexHtmlPath, 'utf-8')
      const staticPosts = collectBlogSourceMetadata(import.meta.dirname)
      const staticBlogSlugs = staticPosts.map(post => post.slug)
      const generatedBlogPostsUrl = resolveGeneratedBlogPostsUrl()
      const generatedBlogRoutes = (
        await fetchGeneratedBlogPrerenderEntries({
          postsUrl: generatedBlogPostsUrl,
          publicSiteUrl: BASE_URL,
          excludeSlugs: staticBlogSlugs,
        }) as GeneratedBlogPrerenderEntry[]
      ).map(entry => blogPostRoute(entry.post, {
        path: entry.path,
        canonical: entry.loc,
        trustedHtml: false,
      }))
      const landingPageRoutes = (
        await fetchLandingPagePrerenderEntries({
          sitemapUrl: resolveLandingPageSitemapUrl(),
          publicSiteUrl: BASE_URL,
          apiBaseUrl: resolveLandingPagePublicApiBase(),
        }) as PublicLandingPagePrerenderEntry[]
      ).map(landingPageRoute)

      const blogRoutes = staticPosts.map(post => blogPostRoute(post))

      const LANDING_JSON_LD = {
        '@context': 'https://schema.org',
        '@graph': [
          {
            '@type': 'Organization',
            name: 'Atlas Intelligence',
            url: BASE_URL,
            logo: DEFAULT_OG_IMAGE,
          },
          {
            '@type': 'WebSite',
            name: 'Atlas Intelligence',
            url: BASE_URL,
            potentialAction: {
              '@type': 'SearchAction',
              target: `${BASE_URL}/blog?q={search_term_string}`,
              'query-input': 'required name=search_term_string',
            },
          },
          {
            '@type': 'SoftwareApplication',
            name: 'Atlas Intelligence',
            applicationCategory: 'BusinessApplication',
            operatingSystem: 'Web',
            description:
              'Amazon review intelligence platform for tracking competitor complaints, safety signals, and customer migration patterns.',
            offers: {
              '@type': 'AggregateOffer',
              lowPrice: '49',
              highPrice: '399',
              priceCurrency: 'USD',
            },
            url: `${BASE_URL}/landing`,
          },
        ],
      }

      const routes: PrerenderedRoute[] = [
        {
          path: '/landing',
          title: 'Atlas Intelligence — Amazon Review Monitoring & Competitor Signals',
          description:
            'Track competitor complaints, safety signals, and customer migration patterns across your Amazon product category. Start free for 14 days.',
          canonical: `${BASE_URL}/landing`,
          ogType: 'website',
          jsonLd: LANDING_JSON_LD,
        },
        {
          path: '/blog',
          title: 'Blog | Atlas Intelligence',
          description:
            'Amazon seller intelligence, review monitoring strategies, and competitive analysis insights.',
          canonical: `${BASE_URL}/blog`,
          ogType: 'website',
        },
        ...blogRoutes,
        ...generatedBlogRoutes,
        ...landingPageRoutes,
      ]

      let count = 0
      for (const route of routes) {
        const headHtml = buildHeadHtml(route)
        // Replace the fallback <title> in index.html and inject all meta before </head>
        const rendered = baseHtml
          .replace(
            /<title>[^<]*<\/title>/,
            `<title>${escapeHtml(route.title)}</title>`,
          )
          .replace('</head>', `${headHtml}\n  </head>`)
          .replace('<div id="root"></div>', `<div id="root">${route.bodyHtml || ''}</div>`)

        // Write to dist/<path>/index.html
        const outDir = join(distDir, ...route.path.split('/').filter(Boolean))
        if (!existsSync(outDir)) mkdirSync(outDir, { recursive: true })
        writeFileSync(join(outDir, 'index.html'), rendered)
        count++
      }

      console.log(`  Pre-rendered ${count} public routes`)
    },
  }
}

export default defineConfig({
  plugins: [react(), sitemapPlugin(), prerenderPlugin()],
  server: {
    host: true,
    port: 5175,
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
