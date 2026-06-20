import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs'
import { resolve, join } from 'node:path'

const BASE_URL = 'https://churnsignals.co'
const DEFAULT_OG_IMAGE = `${BASE_URL}/og-default.png`

// FTC affiliate disclosure for no-JS agents -- AEO crawlers (GPTBot,
// PerplexityBot, ClaudeBot, Bingbot) and social unfurl bots that never run the
// React app. BlogArticleView renders the same disclosure for human readers via
// useEffect, but a crawler sees only the prerendered shell, so without this the
// FTC notice is invisible to exactly the audience the prerender exists for.
// The block sits OUTSIDE #root (React never touches it) and inside <noscript>
// (JS browsers never render it -- no flash, no duplicate of the React copy).
// Copy is kept in sync with BlogArticleView.tsx's disclosure text.
const AFFILIATE_DISCLOSURE_NOSCRIPT =
  '<noscript>' +
  '<div style="max-width:48rem;margin:0 auto;padding:0.75rem 1rem;' +
  'font-size:0.75rem;color:#64748b;border:1px solid #334155;border-radius:0.5rem">' +
  '<strong>Disclosure:</strong> This article may contain affiliate links. ' +
  'If you purchase through these links, we may earn a commission at no ' +
  'additional cost to you. Our analysis and recommendations are based on ' +
  'verified review data, not affiliate relationships. ' +
  'See our <a href="/methodology">methodology</a>.' +
  '</div>' +
  '</noscript>'

function sitemapPlugin() {
  return {
    name: 'generate-sitemap',
    closeBundle() {
      const indexPath = resolve(import.meta.dirname, 'src/content/blog/index.ts')
      if (!existsSync(indexPath)) return

      const indexContent = readFileSync(indexPath, 'utf-8')
      const slugRegex = /slug:\s*'([^']+)'/g
      const slugs: string[] = []
      let match
      while ((match = slugRegex.exec(indexContent)) !== null) {
        slugs.push(match[1])
      }

      const blogDir = resolve(import.meta.dirname, 'src/content/blog')
      for (const file of readdirSync(blogDir)) {
        if (file.endsWith('.ts') && file !== 'index.ts') {
          const content = readFileSync(join(blogDir, file), 'utf-8')
          const m = content.match(/slug:\s*'([^']+)'/)
          if (m && !slugs.includes(m[1])) slugs.push(m[1])
        }
      }

      const today = new Date().toISOString().split('T')[0]
      const urls = [
        { loc: `${BASE_URL}/`, priority: '1.0', changefreq: 'weekly' },
        { loc: `${BASE_URL}/blog`, priority: '0.9', changefreq: 'daily' },
        { loc: `${BASE_URL}/methodology`, priority: '0.6', changefreq: 'monthly' },
        { loc: `${BASE_URL}/landing`, priority: '0.8', changefreq: 'weekly' },
        ...slugs.map(slug => ({
          loc: `${BASE_URL}/blog/${slug}`,
          priority: '0.7',
          changefreq: 'monthly',
        })),
      ]

      const xml = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${urls.map(u => `  <url>
    <loc>${u.loc}</loc>
    <lastmod>${today}</lastmod>
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

interface PrerenderedRoute {
  path: string
  title: string
  description: string
  canonical: string
  ogType: string
  jsonLd?: object
  faqJsonLd?: object
  breadcrumbJsonLd?: object
  // Static HTML injected into <body> before #root (no-JS-only via <noscript>).
  bodyHtml?: string
}

function htmlEscape(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

function extractFaq(content: string): Array<{ question: string; answer: string }> {
  const faqMatch = content.match(/faq:\s*\[([\s\S]*?)\n\s*\],?/)
  if (!faqMatch) return []
  const items: Array<{ question: string; answer: string }> = []
  const itemRegex = /"question":\s*"((?:[^"\\]|\\.)*)"\s*,\s*"answer":\s*"((?:[^"\\]|\\.)*)"/g
  let m
  while ((m = itemRegex.exec(faqMatch[1])) !== null) {
    items.push({
      question: m[1].replace(/\\"/g, '"').replace(/\\\\/g, '\\'),
      answer: m[2].replace(/\\"/g, '"').replace(/\\\\/g, '\\'),
    })
  }
  return items
}

function buildHeadHtml(route: PrerenderedRoute): string {
  const t = htmlEscape(route.title)
  const d = htmlEscape(route.description)
  const c = htmlEscape(route.canonical)
  const lines = [
    `<meta name="description" content="${d}" />`,
    `<link rel="canonical" href="${c}" />`,
    `<meta property="og:title" content="${t}" />`,
    `<meta property="og:description" content="${d}" />`,
    `<meta property="og:url" content="${c}" />`,
    `<meta property="og:type" content="${htmlEscape(route.ogType)}" />`,
    `<meta property="og:site_name" content="Churn Signals" />`,
    `<meta property="og:image" content="${DEFAULT_OG_IMAGE}" />`,
    `<meta property="og:image:width" content="1200" />`,
    `<meta property="og:image:height" content="630" />`,
    `<meta name="twitter:card" content="summary_large_image" />`,
    `<meta name="twitter:title" content="${t}" />`,
    `<meta name="twitter:description" content="${d}" />`,
    `<meta name="twitter:image" content="${DEFAULT_OG_IMAGE}" />`,
  ]
  if (route.jsonLd) {
    lines.push(`<script id="seo-jsonld" type="application/ld+json">${JSON.stringify(route.jsonLd)}</script>`)
  }
  if (route.faqJsonLd) {
    lines.push(`<script id="seo-faq-jsonld" type="application/ld+json">${JSON.stringify(route.faqJsonLd)}</script>`)
  }
  if (route.breadcrumbJsonLd) {
    lines.push(`<script id="seo-breadcrumb-jsonld" type="application/ld+json">${JSON.stringify(route.breadcrumbJsonLd)}</script>`)
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

      const organization = {
        '@type': 'Organization',
        name: 'Churn Signals',
        url: BASE_URL,
        logo: DEFAULT_OG_IMAGE,
        sameAs: ['https://twitter.com/churnsignals', 'https://www.linkedin.com/company/churn-signals'],
      }
      const website = {
        '@type': 'WebSite',
        name: 'Churn Signals',
        url: BASE_URL,
        potentialAction: {
          '@type': 'SearchAction',
          target: `${BASE_URL}/blog?q={search_term_string}`,
          'query-input': 'required name=search_term_string',
        },
      }
      const homeGraph = { '@context': 'https://schema.org', '@graph': [organization, website] }

      const blogDir = resolve(import.meta.dirname, 'src/content/blog')
      const blogRoutes: PrerenderedRoute[] = []
      for (const file of readdirSync(blogDir)) {
        if (!file.endsWith('.ts') || file === 'index.ts') continue
        const content = readFileSync(join(blogDir, file), 'utf-8')
        const slug = (content.match(/slug:\s*'([^']+)'/) || [])[1]
        if (!slug) continue
        const seoTitle =
          (content.match(/seo_title:\s*'([^']+)'/) || [])[1] ||
          (content.match(/title:\s*'([^']+)'/) || [])[1] ||
          'Churn Signals'
        const seoDesc =
          (content.match(/seo_description:\s*'([^']+)'/) || [])[1] ||
          (content.match(/description:\s*'([^']+)'/) || [])[1] ||
          'B2B software churn intelligence from real enterprise reviews.'
        const date = (content.match(/date:\s*'([^']+)'/) || [])[1] || ''
        const title = `${seoTitle} | Churn Signals`
        const canonical = `${BASE_URL}/blog/${slug}`

        const blogPosting: Record<string, unknown> = {
          '@context': 'https://schema.org',
          '@type': 'BlogPosting',
          headline: seoTitle,
          description: seoDesc,
          datePublished: date,
          dateModified: date,
          image: DEFAULT_OG_IMAGE,
          author: { '@type': 'Organization', name: 'Churn Signals' },
          publisher: {
            '@type': 'Organization',
            name: 'Churn Signals',
            url: BASE_URL,
            logo: { '@type': 'ImageObject', url: DEFAULT_OG_IMAGE },
          },
          mainEntityOfPage: { '@type': 'WebPage', '@id': canonical },
        }

        const breadcrumbJsonLd = {
          '@context': 'https://schema.org',
          '@type': 'BreadcrumbList',
          itemListElement: [
            { '@type': 'ListItem', position: 1, name: 'Home', item: `${BASE_URL}/` },
            { '@type': 'ListItem', position: 2, name: 'Blog', item: `${BASE_URL}/blog` },
            { '@type': 'ListItem', position: 3, name: seoTitle, item: canonical },
          ],
        }

        // Parity with BlogArticleView.hasAffiliateContent: a non-empty
        // data_context.affiliate_url, or Monday.com's affiliate URL inlined in
        // the body. Posts that carry an affiliate link must ship the FTC
        // disclosure in the static HTML so no-JS crawlers see it.
        const hasAffiliate =
          /"affiliate_url"\s*:\s*"[^"]+"/.test(content) ||
          content.includes('try.monday.com')

        const faqItems = extractFaq(content)
        const faqJsonLd = faqItems.length > 0 ? {
          '@context': 'https://schema.org',
          '@type': 'FAQPage',
          mainEntity: faqItems.map(it => ({
            '@type': 'Question',
            name: it.question,
            acceptedAnswer: { '@type': 'Answer', text: it.answer },
          })),
        } : undefined

        blogRoutes.push({
          path: `/blog/${slug}`,
          title,
          description: seoDesc,
          canonical,
          ogType: 'article',
          jsonLd: blogPosting,
          faqJsonLd,
          breadcrumbJsonLd,
          bodyHtml: hasAffiliate ? AFFILIATE_DISCLOSURE_NOSCRIPT : undefined,
        })
      }

      const routes: PrerenderedRoute[] = [
        {
          path: '/',
          title: 'Churn Signals - B2B Software Churn Intelligence',
          description: 'B2B software churn intelligence from real enterprise reviews. Track vendor switching signals, competitive displacement, and high-intent leads.',
          canonical: `${BASE_URL}/`,
          ogType: 'website',
          jsonLd: homeGraph,
        },
        {
          path: '/blog',
          title: 'Blog | Churn Signals',
          description: 'Vendor deep dives, migration guides, and switching-signal analysis from enterprise software review data.',
          canonical: `${BASE_URL}/blog`,
          ogType: 'website',
        },
        {
          path: '/methodology',
          title: 'Our Methodology | Churn Signals',
          description: 'How we aggregate, classify, and surface churn signals from G2, Capterra, TrustRadius, and Reddit reviews.',
          canonical: `${BASE_URL}/methodology`,
          ogType: 'article',
        },
        {
          path: '/landing',
          title: 'Churn Signals - B2B Churn Intelligence',
          description: 'Track competitive displacement and high-intent buyer signals across B2B software categories.',
          canonical: `${BASE_URL}/landing`,
          ogType: 'website',
        },
        ...blogRoutes,
      ]

      // Strip the default SEO meta tags from the base shell before
      // injecting per-route versions. Without this, each prerendered
      // page ships duplicate `og:*` / `twitter:*` / `description` tags
      // -- the generic homepage values first, then the per-route ones
      // -- and some no-JS crawlers (the audience for prerender) read
      // the FIRST occurrence and end up with the generic metadata.
      // Codex flagged this on PR #642.
      //
      // We keep `<meta charset>`, `<meta viewport>`, `<meta theme-color>`,
      // `<link rel="icon">`, and the Vite-injected `<script>` / stylesheet
      // links untouched -- those are not duplicated by the prerender
      // plugin and have no per-route variant.
      const SEO_TAG_RE = new RegExp(
        [
          // <title>...</title>
          '\\s*<title>[^<]*<\\/title>',
          // <meta name="description" ...>
          '\\s*<meta\\s+name=["\']description["\'][^>]*\\/?>',
          // <meta property="og:..." ...> (anywhere in tag, attribute order varies)
          '\\s*<meta\\s+[^>]*property=["\']og:[^"\']+["\'][^>]*\\/?>',
          // <meta name="twitter:..." ...>
          '\\s*<meta\\s+[^>]*name=["\']twitter:[^"\']+["\'][^>]*\\/?>',
        ].join('|'),
        'gi',
      )
      const baseHtmlStripped = baseHtml.replace(SEO_TAG_RE, '')

      let count = 0
      for (const route of routes) {
        const headHtml = buildHeadHtml(route)
        // Inject the route's own title + the rest of the head block
        // before </head>. The base shell no longer has a <title> to
        // replace, so the new one lands inside the head HTML block.
        let rendered = baseHtmlStripped.replace(
          '</head>',
          `    <title>${htmlEscape(route.title)}</title>\n${headHtml}\n  </head>`,
        )
        // Inject the no-JS disclosure (when present) just before #root, so it
        // lives outside React's render tree and is visible only to no-JS
        // agents.
        if (route.bodyHtml) {
          rendered = rendered.replace(
            '<div id="root"></div>',
            `${route.bodyHtml}\n    <div id="root"></div>`,
          )
        }
        const segments = route.path.split('/').filter(Boolean)
        const outDir = segments.length === 0 ? distDir : join(distDir, ...segments)
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
    port: 5174,
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  test: {
    environment: 'jsdom',
    fileParallelism: false,
    setupFiles: './src/test/setup.ts',
  },
})
