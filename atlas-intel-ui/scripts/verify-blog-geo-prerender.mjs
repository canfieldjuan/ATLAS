import { existsSync, readFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { collectBlogSourceMetadata } from './blog-source-metadata.mjs'

const BASE_URL = 'https://atlas-intel-ui-two.vercel.app'
const rootDir = resolve(import.meta.dirname, '..')
const distDir = join(rootDir, 'dist')
const sitemapPath = join(distDir, 'sitemap.xml')

function fail(message) {
  throw new Error(message)
}

function readText(path) {
  if (!existsSync(path)) fail(`Missing file: ${path}`)
  return readFileSync(path, 'utf-8')
}

function collectBlogPosts() {
  try {
    return collectBlogSourceMetadata(rootDir)
  } catch (error) {
    fail(error.message)
  }
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function attrValue(tag, attr) {
  const pattern = new RegExp(`${escapeRegExp(attr)}=(["'])(.*?)\\1`, 'i')
  const match = tag.match(pattern)
  return match ? match[2] : ''
}

function findLink(html, rel) {
  const links = html.match(/<link\b[^>]*>/gi) || []
  return links.find(tag => attrValue(tag, 'rel').toLowerCase() === rel) || ''
}

function findMeta(html, attr, key) {
  const metas = html.match(/<meta\b[^>]*>/gi) || []
  return metas.find(tag => attrValue(tag, attr) === key) || ''
}

function findTitle(html, slug) {
  const matches = [...html.matchAll(/<title>([\s\S]*?)<\/title>/gi)]
  if (matches.length !== 1) {
    fail(`/blog/${slug} must have exactly one title tag`)
  }
  return decodeHtml(matches[0][1].trim())
}

function findH1(html, slug) {
  const matches = [...html.matchAll(/<h1\b[^>]*>([\s\S]*?)<\/h1>/gi)]
  if (matches.length !== 1) {
    fail(`/blog/${slug} must have exactly one crawler-visible h1`)
  }
  return normalizedText(matches[0][1])
}

function decodeHtml(value) {
  return value
    .replace(/&amp;/g, '&')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
}

function normalizedText(value) {
  return decodeHtml(value)
    .replace(/<script\b[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style\b[\s\S]*?<\/style>/gi, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

function sourceBodyText(value) {
  return normalizedText(
    value
      .replace(/<p>\s*\{\{chart:[^}]+\}\}\s*<\/p>|\{\{chart:[^}]+\}\}/g, ' ')
      .replace(/<h[1-6]\b[\s\S]*?<\/h[1-6]>/gi, ' ')
      .replace(/^#{1,6}\s+.*$/gm, ' ')
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
      .replace(/[#*_`|]+/g, ' '),
  )
}

function expectedBodyPhrase(post) {
  const text = sourceBodyText(post.content)
  const sentence = text
    .split(/(?<=[.!?])\s+/)
    .find(item => item.length >= 60 && /[A-Za-z]/.test(item))
  if (!sentence) fail(`/blog/${post.slug} source body needs a verifier phrase`)
  return sentence.slice(0, 80).trim()
}

function parseJsonLd(html, slug) {
  const scripts = [...html.matchAll(
    /<script\b[^>]*type=["']application\/ld\+json["'][^>]*>([\s\S]*?)<\/script>/gi,
  )]
  const objects = []
  for (const script of scripts) {
    try {
      objects.push(JSON.parse(script[1]))
    } catch (error) {
      fail(`Invalid JSON-LD for ${slug}: ${error.message}`)
    }
  }
  return objects
}

function flattenJsonLdTypes(objects) {
  const nodes = []
  for (const object of objects) {
    if (object && typeof object === 'object' && Array.isArray(object['@graph'])) {
      nodes.push(...object['@graph'])
    } else {
      nodes.push(object)
    }
  }
  return nodes.filter(node => node && typeof node === 'object')
}

function typeMatches(node, typeName) {
  const type = node['@type']
  if (Array.isArray(type)) return type.includes(typeName)
  return type === typeName
}

function assertNoNoindex(html, slug) {
  const robotsMeta = findMeta(html, 'name', 'robots')
  if (!robotsMeta) return
  const content = attrValue(robotsMeta, 'content').toLowerCase()
  if (content.includes('noindex')) {
    fail(`/blog/${slug} has a noindex robots directive`)
  }
}

function expectedTitle(post) {
  return post.seoTitle || post.title
}

function expectedDescription(post) {
  return post.seoDescription || post.description
}

function assertSame(actual, expected, label, slug) {
  if (actual !== expected) {
    fail(`/blog/${slug} ${label} must be ${expected}`)
  }
}

function assertOrganization(node, label, slug) {
  if (!node || node['@type'] !== 'Organization') {
    fail(`/blog/${slug} BlogPosting ${label} must be an Organization`)
  }
  assertSame(node.name, 'Atlas Intelligence', `BlogPosting ${label} name`, slug)
  const sameAs = Array.isArray(node.sameAs) ? node.sameAs : []
  if (!sameAs.includes('https://twitter.com/atlasintel')) {
    fail(`/blog/${slug} BlogPosting ${label} missing Twitter sameAs`)
  }
  if (!sameAs.includes('https://www.linkedin.com/company/atlas-intelligence')) {
    fail(`/blog/${slug} BlogPosting ${label} missing LinkedIn sameAs`)
  }
}

function assertBlogPosting(node, post, canonical, ogImageValue) {
  const { slug } = post
  for (const field of ['headline', 'description', 'image', 'author', 'publisher']) {
    if (!node[field]) fail(`/blog/${slug} BlogPosting missing ${field}`)
  }
  assertSame(node.headline, expectedTitle(post), 'BlogPosting headline', slug)
  assertSame(node.description, expectedDescription(post), 'BlogPosting description', slug)
  assertSame(node.datePublished, post.date, 'BlogPosting datePublished', slug)
  assertSame(node.dateModified, post.date, 'BlogPosting dateModified', slug)
  assertSame(node.image, ogImageValue, 'BlogPosting image', slug)

  const mainEntity = node.mainEntityOfPage
  if (!mainEntity || mainEntity['@id'] !== canonical) {
    fail(`/blog/${slug} BlogPosting mainEntityOfPage must be ${canonical}`)
  }

  assertOrganization(node.author, 'author', slug)
  assertOrganization(node.publisher, 'publisher', slug)
}

function assertMetaContent(html, attr, key, expected, slug) {
  const meta = findMeta(html, attr, key)
  const actual = decodeHtml(attrValue(meta, 'content'))
  if (actual !== expected) {
    fail(`/blog/${slug} ${key} must be ${expected}`)
  }
}

function assertSeoMeta(html, post) {
  const title = `${expectedTitle(post)} | Atlas Intelligence`
  const description = expectedDescription(post)

  if (findTitle(html, post.slug) !== title) {
    fail(`/blog/${post.slug} title tag must be ${title}`)
  }

  assertMetaContent(html, 'name', 'description', description, post.slug)
  assertMetaContent(html, 'property', 'og:title', title, post.slug)
  assertMetaContent(html, 'property', 'og:description', description, post.slug)
  assertMetaContent(html, 'name', 'twitter:title', title, post.slug)
  assertMetaContent(html, 'name', 'twitter:description', description, post.slug)

  const twitterImage = findMeta(html, 'name', 'twitter:image')
  const twitterImageValue = attrValue(twitterImage, 'content')
  if (!twitterImageValue || !twitterImageValue.startsWith(BASE_URL)) {
    fail(`/blog/${post.slug} twitter:image must be an absolute Atlas URL`)
  }
}

function assertCrawlerVisibleArticle(html, post) {
  if (!html.includes('data-prerendered-blog-article="true"')) {
    fail(`/blog/${post.slug} missing prerendered article body`)
  }
  if (!html.includes('data-prerendered-blog-content="true"')) {
    fail(`/blog/${post.slug} missing prerendered article content`)
  }

  const visibleText = normalizedText(html)
  const h1 = findH1(html, post.slug)
  if (h1 !== post.title) {
    fail(`/blog/${post.slug} h1 must be ${post.title}`)
  }
  if (!visibleText.includes(post.author)) {
    fail(`/blog/${post.slug} prerendered body missing source author`)
  }
  if (html.includes('{{chart:')) {
    fail(`/blog/${post.slug} prerendered body still contains chart placeholders`)
  }

  for (const chart of post.charts) {
    if (!html.includes(`data-prerendered-chart="${chart.chart_id}"`)) {
      fail(`/blog/${post.slug} missing prerendered chart fallback for ${chart.chart_id}`)
    }
    if (!visibleText.includes(chart.title)) {
      fail(`/blog/${post.slug} prerendered chart fallback missing title ${chart.title}`)
    }
  }

  const phrase = expectedBodyPhrase(post)
  if (!visibleText.includes(phrase)) {
    fail(`/blog/${post.slug} prerendered body missing source phrase: ${phrase}`)
  }

  if (visibleText.length < 1000) {
    fail(`/blog/${post.slug} prerendered body is too short for crawler-visible article content`)
  }
}

function assertBreadcrumbs(node, canonical, slug) {
  const items = Array.isArray(node.itemListElement) ? node.itemListElement : []
  if (items.length < 3) fail(`/blog/${slug} breadcrumb list is incomplete`)
  const last = items[items.length - 1]
  if (!last || last.item !== canonical) {
    fail(`/blog/${slug} breadcrumb final item must be ${canonical}`)
  }
}

function sitemapLastmod(sitemap, canonical) {
  const pattern = new RegExp(
    `<url>\\s*<loc>${escapeRegExp(canonical)}</loc>\\s*<lastmod>([^<]+)</lastmod>`,
  )
  const match = sitemap.match(pattern)
  return match ? match[1] : ''
}

function verifyBlogPage(post, sitemap) {
  const { slug } = post
  const canonical = `${BASE_URL}/blog/${slug}`
  const htmlPath = join(distDir, 'blog', slug, 'index.html')
  const html = readText(htmlPath)

  assertSeoMeta(html, post)
  assertCrawlerVisibleArticle(html, post)

  const canonicalLink = findLink(html, 'canonical')
  if (attrValue(canonicalLink, 'href') !== canonical) {
    fail(`/blog/${slug} canonical link must be ${canonical}`)
  }

  const ogUrl = findMeta(html, 'property', 'og:url')
  if (attrValue(ogUrl, 'content') !== canonical) {
    fail(`/blog/${slug} og:url must be ${canonical}`)
  }

  const ogType = findMeta(html, 'property', 'og:type')
  if (attrValue(ogType, 'content') !== 'article') {
    fail(`/blog/${slug} og:type must be article`)
  }

  const ogImage = findMeta(html, 'property', 'og:image')
  const ogImageValue = attrValue(ogImage, 'content')
  if (!ogImageValue || !ogImageValue.startsWith(BASE_URL)) {
    fail(`/blog/${slug} og:image must be an absolute Atlas URL`)
  }

  assertNoNoindex(html, slug)

  const nodes = flattenJsonLdTypes(parseJsonLd(html, slug))
  const blogPosting = nodes.find(node => typeMatches(node, 'BlogPosting'))
  if (!blogPosting) fail(`/blog/${slug} missing BlogPosting JSON-LD`)
  assertBlogPosting(blogPosting, post, canonical, ogImageValue)

  const breadcrumbs = nodes.find(node => typeMatches(node, 'BreadcrumbList'))
  if (!breadcrumbs) fail(`/blog/${slug} missing BreadcrumbList JSON-LD`)
  assertBreadcrumbs(breadcrumbs, canonical, slug)

  if (!sitemap.includes(`<loc>${canonical}</loc>`)) {
    fail(`/blog/${slug} missing from sitemap.xml`)
  }

  const lastmod = sitemapLastmod(sitemap, canonical)
  if (lastmod !== post.date) {
    fail(`/blog/${slug} sitemap lastmod must be ${post.date}`)
  }
}

const posts = collectBlogPosts()
const sitemap = readText(sitemapPath)
for (const post of posts) {
  verifyBlogPage(post, sitemap)
}

console.log(`Verified GEO prerender metadata for ${posts.length} blog pages`)
