import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const rootDir = fileURLToPath(new URL('..', import.meta.url))
const defaultDistDir = join(rootDir, 'dist')
const FALLBACK_TITLE_RE =
  /^Atlas Intelligence\s+(?:-|\u2014)\s+Amazon Review Monitoring & Competitor Signals$/

function readText(path) {
  return readFileSync(path, 'utf8')
}

function decodeXmlEntities(value) {
  return value
    .replace(/&amp;/g, '&')
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
}

function landingPageUrlsFromSitemap(sitemap) {
  const urls = []
  const locMatches = String(sitemap || '').matchAll(/<loc>\s*([\s\S]*?)\s*<\/loc>/gi)
  for (const match of locMatches) {
    const rawLoc = decodeXmlEntities(match[1])
    let parsed
    try {
      parsed = new URL(rawLoc)
    } catch (_error) {
      continue
    }
    if (!parsed.pathname.startsWith('/lp/')) continue
    urls.push({
      loc: `${parsed.origin}${parsed.pathname}`,
      path: parsed.pathname,
    })
  }
  return urls
}

function landingPageHtmlPath(pathname, distDir) {
  const parts = pathname.split('/').filter(Boolean)
  return join(distDir, ...parts, 'index.html')
}

function attrValue(tag, attr) {
  const pattern = new RegExp(`${attr}=["']([^"']*)["']`, 'i')
  return tag.match(pattern)?.[1] || ''
}

function findMeta(html, keyAttr, keyValue) {
  const pattern = new RegExp(
    `<meta\\b(?=[^>]*\\b${keyAttr}=["']${escapeRegex(keyValue)}["'])[^>]*>`,
    'i',
  )
  return html.match(pattern)?.[0] || ''
}

function findLink(html, rel) {
  const pattern = new RegExp(
    `<link\\b(?=[^>]*\\brel=["']${escapeRegex(rel)}["'])[^>]*>`,
    'i',
  )
  return html.match(pattern)?.[0] || ''
}

function escapeRegex(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function scriptJsonLdBlocks(html) {
  return [...html.matchAll(/<script\b[^>]*type=["']application\/ld\+json["'][^>]*>([\s\S]*?)<\/script>/gi)]
    .map(match => match[1].trim())
    .filter(Boolean)
}

function hasJsonLdGraph(html) {
  return scriptJsonLdBlocks(html).some((block) => {
    try {
      const parsed = JSON.parse(block)
      return parsed && parsed['@context'] === 'https://schema.org'
    } catch (_error) {
      return false
    }
  })
}

function isFallbackTitle(title) {
  return FALLBACK_TITLE_RE.test(title)
}

function verifyLandingPage(url, distDir, fail) {
  const htmlPath = landingPageHtmlPath(url.path, distDir)
  if (!existsSync(htmlPath)) {
    fail(`${url.path} is listed in sitemap.xml but missing ${htmlPath}`)
    return
  }

  const html = readText(htmlPath)
  const title = html.match(/<title>([^<]+)<\/title>/i)?.[1]?.trim() || ''
  if (!title || isFallbackTitle(title)) {
    fail(`${url.path} has missing or fallback <title>`)
  }

  const description = findMeta(html, 'name', 'description')
  if (!attrValue(description, 'content')) {
    fail(`${url.path} missing meta description`)
  }

  const canonical = findLink(html, 'canonical')
  if (attrValue(canonical, 'href') !== url.loc) {
    fail(`${url.path} canonical href must be ${url.loc}`)
  }

  const robots = findMeta(html, 'name', 'robots')
  if (attrValue(robots, 'content').toLowerCase() !== 'index,follow') {
    fail(`${url.path} must be index,follow in prerendered robots meta`)
  }

  const ogUrl = findMeta(html, 'property', 'og:url')
  if (attrValue(ogUrl, 'content') !== url.loc) {
    fail(`${url.path} og:url must be ${url.loc}`)
  }

  if (!hasJsonLdGraph(html)) {
    fail(`${url.path} missing valid Schema.org JSON-LD`)
  }

  if (!html.includes('data-prerendered-landing-page="true"')) {
    fail(`${url.path} missing prerendered landing-page body marker`)
  }

  if (!/<h1>[^<]+<\/h1>/i.test(html)) {
    fail(`${url.path} missing prerendered H1`)
  }

  if (!/<a\b[^>]*href=["'][^"']+["'][^>]*>[^<]+<\/a>/i.test(html)) {
    fail(`${url.path} missing prerendered CTA link`)
  }
}

export function verifyLandingPagePrerender({
  distDir = defaultDistDir,
  logger = console,
} = {}) {
  const sitemapPath = join(distDir, 'sitemap.xml')
  const failures = []
  const fail = message => failures.push(message)
  let checked = 0

  if (!existsSync(sitemapPath)) {
    fail(`Missing sitemap.xml at ${sitemapPath}`)
    return { checked, failures }
  }

  const urls = landingPageUrlsFromSitemap(readText(sitemapPath))
  for (const url of urls) {
    verifyLandingPage(url, distDir, fail)
  }
  checked = urls.length
  if (urls.length === 0) {
    logger.log('No generated landing-page sitemap entries found; skipping landing-page GEO prerender verification.')
  } else {
    logger.log(`Verified GEO prerender metadata for ${urls.length} generated landing page(s)`)
  }

  return { checked, failures }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const { failures } = verifyLandingPagePrerender()
  if (failures.length) {
    console.error('Landing-page GEO prerender verification failed:')
    for (const failure of failures) {
      console.error(`- ${failure}`)
    }
    process.exit(1)
  }
}
