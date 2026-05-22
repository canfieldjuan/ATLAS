function clean(value) {
  return String(value || '').trim()
}

function stripTrailingSlash(value) {
  return value.replace(/\/+$/, '')
}

export function resolveLandingPageSitemapUrl(env = process.env) {
  return clean(env.VITE_PUBLIC_LANDING_PAGE_SITEMAP_URL)
}

function decodeXmlEntities(value) {
  return value
    .replace(/&amp;/g, '&')
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
}

export function landingPageSitemapUrlsFromXml(xml, publicSiteUrl) {
  const siteBase = new URL(stripTrailingSlash(clean(publicSiteUrl)))
  const seen = new Set()
  const urls = []
  const locMatches = String(xml || '').matchAll(/<loc>\s*([\s\S]*?)\s*<\/loc>/gi)

  for (const match of locMatches) {
    const rawLoc = decodeXmlEntities(match[1])
    let parsed
    try {
      parsed = new URL(rawLoc)
    } catch (_error) {
      continue
    }

    if (!parsed.pathname.startsWith('/lp/')) continue

    const loc = `${siteBase.origin}${parsed.pathname}`
    if (seen.has(loc)) continue
    seen.add(loc)
    urls.push({
      loc,
      priority: '0.7',
      changefreq: 'weekly',
    })
  }

  return urls
}

export async function fetchLandingPageSitemapUrls({
  sitemapUrl,
  publicSiteUrl,
  fetchImpl = globalThis.fetch,
} = {}) {
  const feedUrl = clean(sitemapUrl)
  if (!feedUrl) return []
  if (typeof fetchImpl !== 'function') {
    throw new Error('Landing-page sitemap bridge requires a fetch implementation')
  }

  const response = await fetchImpl(feedUrl)
  if (!response || !response.ok) {
    const status = response?.status ? `HTTP ${response.status}` : 'no response'
    throw new Error(`Failed to fetch generated landing-page sitemap: ${status}`)
  }

  const xml = await response.text()
  return landingPageSitemapUrlsFromXml(xml, publicSiteUrl)
}
