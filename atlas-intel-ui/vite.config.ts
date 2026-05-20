import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
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
  seoTitle: string
  seoDescription: string
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function parseStringField(content: string, field: string): string {
  const pattern = new RegExp(`^\\s*${escapeRegExp(field)}:\\s*'((?:\\\\'|[^'])*)'`, 'm')
  const match = content.match(pattern)
  return match ? match[1].replace(/\\'/g, "'") : ''
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
    const seoTitle = parseStringField(content, 'seo_title')
    const seoDescription = parseStringField(content, 'seo_description')

    if (!slug) throw new Error(`Missing slug in blog source: ${file}`)
    if (!title) throw new Error(`Missing title in blog source: ${file}`)
    if (!description) throw new Error(`Missing description in blog source: ${file}`)
    if (!date) throw new Error(`Missing date in blog source: ${file}`)

    posts.push({
      file,
      slug,
      title,
      description,
      date,
      seoTitle,
      seoDescription,
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
