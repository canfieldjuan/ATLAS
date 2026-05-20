import { existsSync, readFileSync, readdirSync } from 'node:fs'
import { join, resolve } from 'node:path'

const BASE_URL = 'https://atlas-intel-ui-two.vercel.app'
const rootDir = resolve(import.meta.dirname, '..')
const blogDir = join(rootDir, 'src/content/blog')
const distDir = join(rootDir, 'dist')
const sitemapPath = join(distDir, 'sitemap.xml')

function fail(message) {
  throw new Error(message)
}

function readText(path) {
  if (!existsSync(path)) fail(`Missing file: ${path}`)
  return readFileSync(path, 'utf-8')
}

function collectBlogSlugs() {
  const slugs = []
  for (const file of readdirSync(blogDir)) {
    if (!file.endsWith('.ts') || file === 'index.ts') continue
    const content = readText(join(blogDir, file))
    const match = content.match(/slug:\s*'([^']+)'/)
    if (!match) fail(`Missing slug in ${file}`)
    slugs.push(match[1])
  }
  if (!slugs.length) fail('No blog slugs found')
  return slugs.sort()
}

function attrValue(tag, attr) {
  const pattern = new RegExp(`${attr}=["']([^"']+)["']`, 'i')
  const match = tag.match(pattern)
  return match ? match[1] : ''
}

function findLink(html, rel) {
  const links = html.match(/<link\b[^>]*>/gi) || []
  return links.find(tag => attrValue(tag, 'rel').toLowerCase() === rel) || ''
}

function findMeta(html, attr, key) {
  const metas = html.match(/<meta\b[^>]*>/gi) || []
  return metas.find(tag => attrValue(tag, attr) === key) || ''
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

function assertBlogPosting(node, canonical, slug) {
  for (const field of ['headline', 'description', 'image', 'author', 'publisher']) {
    if (!node[field]) fail(`/blog/${slug} BlogPosting missing ${field}`)
  }
  const mainEntity = node.mainEntityOfPage
  if (!mainEntity || mainEntity['@id'] !== canonical) {
    fail(`/blog/${slug} BlogPosting mainEntityOfPage must be ${canonical}`)
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

function verifyBlogPage(slug, sitemap) {
  const canonical = `${BASE_URL}/blog/${slug}`
  const htmlPath = join(distDir, 'blog', slug, 'index.html')
  const html = readText(htmlPath)

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
  assertBlogPosting(blogPosting, canonical, slug)

  const breadcrumbs = nodes.find(node => typeMatches(node, 'BreadcrumbList'))
  if (!breadcrumbs) fail(`/blog/${slug} missing BreadcrumbList JSON-LD`)
  assertBreadcrumbs(breadcrumbs, canonical, slug)

  if (!sitemap.includes(`<loc>${canonical}</loc>`)) {
    fail(`/blog/${slug} missing from sitemap.xml`)
  }
}

const slugs = collectBlogSlugs()
const sitemap = readText(sitemapPath)
for (const slug of slugs) {
  verifyBlogPage(slug, sitemap)
}

console.log(`Verified GEO prerender metadata for ${slugs.length} blog pages`)
