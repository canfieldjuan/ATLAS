import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { marked } from 'marked'
import {
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  writeFileSync,
} from 'node:fs'
import { resolve, join } from 'node:path'

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

interface BlogSourceMetadata {
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

type ChartValue = string | number | null | undefined
type ChartDatum = Record<string, ChartValue>

interface ChartSeries {
  dataKey: string
}

interface ChartSpec {
  chart_id: string
  title: string
  data: ChartDatum[]
  config: {
    bars?: ChartSeries[]
    x_key?: string
  }
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function parseStringField(content: string, field: string): string {
  const pattern = new RegExp(`^\\s*${escapeRegExp(field)}:\\s*'((?:\\\\'|[^'])*)'`, 'm')
  const match = content.match(pattern)
  return match ? match[1].replace(/\\'/g, "'") : ''
}

function parseTemplateField(content: string, field: string): string {
  const pattern = new RegExp(`^\\s*${escapeRegExp(field)}:\\s*\`([\\s\\S]*?)\`\\s*,`, 'm')
  const match = content.match(pattern)
  return match ? match[1] : ''
}

function parseArrayField(content: string, field: string): string {
  const fieldMatch = new RegExp(`^\\s*${escapeRegExp(field)}:\\s*\\[`, 'm').exec(content)
  if (!fieldMatch) return ''

  const start = fieldMatch.index + fieldMatch[0].lastIndexOf('[')
  let depth = 0
  let quote = ''
  let escaped = false

  for (let index = start; index < content.length; index += 1) {
    const char = content[index]

    if (quote) {
      if (escaped) {
        escaped = false
      } else if (char === '\\') {
        escaped = true
      } else if (char === quote) {
        quote = ''
      }
      continue
    }

    if (char === '"' || char === "'" || char === '`') {
      quote = char
      continue
    }

    if (char === '[') depth += 1
    if (char === ']') depth -= 1

    if (depth === 0) {
      return content.slice(start, index + 1)
    }
  }

  throw new Error(`Unclosed array field in blog source: ${field}`)
}

function parseChartsField(content: string, file: string): ChartSpec[] {
  const chartsLiteral = parseArrayField(content, 'charts')
  if (!chartsLiteral) return []

  try {
    return JSON.parse(chartsLiteral) as ChartSpec[]
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    throw new Error(`Invalid charts JSON in blog source ${file}: ${message}`)
  }
}

function collectBlogSourceMetadata(): BlogSourceMetadata[] {
  const blogDir = resolve(import.meta.dirname, 'src/content/blog')
  if (!existsSync(blogDir)) return []

  const posts: BlogSourceMetadata[] = []
  for (const file of readdirSync(blogDir).sort()) {
    if (!file.endsWith('.ts') || file === 'index.ts') continue
    const content = readFileSync(join(blogDir, file), 'utf-8')
    const slug = parseStringField(content, 'slug')
    const title = parseStringField(content, 'title')
    const description = parseStringField(content, 'description')
    const date = parseStringField(content, 'date')
    const author = parseStringField(content, 'author')
    const seoTitle = parseStringField(content, 'seo_title')
    const seoDescription = parseStringField(content, 'seo_description')
    const postContent = parseTemplateField(content, 'content')
    const charts = parseChartsField(content, file)

    if (!slug) throw new Error(`Missing slug in blog source: ${file}`)
    if (!title) throw new Error(`Missing title in blog source: ${file}`)
    if (!description) throw new Error(`Missing description in blog source: ${file}`)
    if (!date) throw new Error(`Missing date in blog source: ${file}`)
    if (!author) throw new Error(`Missing author in blog source: ${file}`)
    if (!postContent.trim()) throw new Error(`Missing content in blog source: ${file}`)

    posts.push({
      file,
      slug,
      title,
      description,
      date,
      author,
      seoTitle,
      seoDescription,
      content: postContent,
      charts,
    })
  }

  return posts.sort((a, b) => a.slug.localeCompare(b.slug))
}

// ---------------------------------------------------------------------------
// Sitemap plugin
// ---------------------------------------------------------------------------
function sitemapPlugin() {
  return {
    name: 'generate-sitemap',
    closeBundle() {
      const posts = collectBlogSourceMetadata()

      const today = new Date().toISOString().split('T')[0]
      const urls: SitemapUrl[] = [
        { loc: `${BASE_URL}/landing`, priority: '1.0', changefreq: 'weekly' },
        { loc: `${BASE_URL}/blog`, priority: '0.9', changefreq: 'daily' },
        ...posts.map(post => ({
          loc: `${BASE_URL}/blog/${post.slug}`,
          lastmod: post.date,
          priority: '0.7',
          changefreq: 'monthly',
        })),
        { loc: `${BASE_URL}/`, priority: '0.3', changefreq: 'monthly' },
      ]

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

function chartPlaceholderIds(content: string): string[] {
  return [...content.matchAll(/<p>\s*\{\{chart:([^}]+)\}\}\s*<\/p>|\{\{chart:([^}]+)\}\}/g)]
    .map(match => (match[1] || match[2] || '').trim())
    .filter(Boolean)
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

function buildBlogBodyHtml(post: BlogSourceMetadata): string {
  const articleHtml = marked.parse(renderChartFallbacks(post.content, post.charts), { async: false }) as string
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
    </article>`
}

function buildHeadHtml(route: PrerenderedRoute): string {
  const { title, description, canonical, ogType, jsonLd } = route
  const lines = [
    `<meta name="description" content="${description}" />`,
    `<link rel="canonical" href="${canonical}" />`,
    `<meta property="og:title" content="${title}" />`,
    `<meta property="og:description" content="${description}" />`,
    `<meta property="og:url" content="${canonical}" />`,
    `<meta property="og:type" content="${ogType}" />`,
    `<meta property="og:site_name" content="Atlas Intelligence" />`,
    `<meta property="og:image" content="${DEFAULT_OG_IMAGE}" />`,
    `<meta property="og:image:width" content="1200" />`,
    `<meta property="og:image:height" content="630" />`,
    `<meta name="twitter:card" content="summary_large_image" />`,
    `<meta name="twitter:title" content="${title}" />`,
    `<meta name="twitter:description" content="${description}" />`,
    `<meta name="twitter:image" content="${DEFAULT_OG_IMAGE}" />`,
  ]
  if (jsonLd) {
    lines.push(
      `<script type="application/ld+json">${JSON.stringify(jsonLd)}</script>`,
    )
  }
  return lines.map(l => `    ${l}`).join('\n')
}

function prerenderPlugin() {
  return {
    name: 'prerender-public-routes',
    closeBundle() {
      const distDir = resolve(import.meta.dirname, 'dist')
      const indexHtmlPath = join(distDir, 'index.html')
      if (!existsSync(indexHtmlPath)) return

      const baseHtml = readFileSync(indexHtmlPath, 'utf-8')

      const blogRoutes: PrerenderedRoute[] = []
      for (const post of collectBlogSourceMetadata()) {
        const seoTitle = post.seoTitle || post.title
        const seoDesc = post.seoDescription || post.description
        blogRoutes.push({
          path: `/blog/${post.slug}`,
          title: `${seoTitle} | Atlas Intelligence`,
          description: seoDesc,
          canonical: `${BASE_URL}/blog/${post.slug}`,
          ogType: 'article',
          bodyHtml: buildBlogBodyHtml(post),
          jsonLd: {
            '@context': 'https://schema.org',
            '@graph': [
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
                mainEntityOfPage: { '@type': 'WebPage', '@id': `${BASE_URL}/blog/${post.slug}` },
              },
              {
                '@type': 'BreadcrumbList',
                itemListElement: [
                  { '@type': 'ListItem', position: 1, name: 'Home', item: `${BASE_URL}/landing` },
                  { '@type': 'ListItem', position: 2, name: 'Blog', item: `${BASE_URL}/blog` },
                  { '@type': 'ListItem', position: 3, name: seoTitle, item: `${BASE_URL}/blog/${post.slug}` },
                ],
              },
            ],
          },
        })
      }

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
      ]

      let count = 0
      for (const route of routes) {
        const headHtml = buildHeadHtml(route)
        // Replace the fallback <title> in index.html and inject all meta before </head>
        const rendered = baseHtml
          .replace(
            /<title>[^<]*<\/title>/,
            `<title>${route.title}</title>`,
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
