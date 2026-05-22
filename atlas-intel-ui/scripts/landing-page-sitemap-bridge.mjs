function clean(value) {
  return String(value || '').trim()
}

function stripTrailingSlash(value) {
  return value.replace(/\/+$/, '')
}

export function resolveLandingPageSitemapUrl(env = process.env) {
  return clean(env.VITE_PUBLIC_LANDING_PAGE_SITEMAP_URL)
}

export function resolveLandingPagePublicApiBase(env = process.env) {
  const configured = stripTrailingSlash(clean(env.VITE_API_BASE))
  if (configured) return configured

  const sitemapUrl = resolveLandingPageSitemapUrl(env)
  if (!sitemapUrl) return ''
  try {
    return new URL(sitemapUrl).origin
  } catch (_error) {
    return ''
  }
}

function decodeXmlEntities(value) {
  return value
    .replace(/&amp;/g, '&')
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
}

function landingPagePathParts(pathname) {
  const parts = String(pathname || '').split('/').filter(Boolean)
  if (parts[0] !== 'lp' || !parts[1] || !parts[2]) return null
  return {
    id: parts[1],
    slug: parts.slice(2).join('/'),
  }
}

export function landingPageSitemapEntriesFromXml(xml, publicSiteUrl) {
  const siteBase = new URL(stripTrailingSlash(clean(publicSiteUrl)))
  const seen = new Set()
  const entries = []
  const locMatches = String(xml || '').matchAll(/<loc>\s*([\s\S]*?)\s*<\/loc>/gi)

  for (const match of locMatches) {
    const rawLoc = decodeXmlEntities(match[1])
    let parsed
    try {
      parsed = new URL(rawLoc)
    } catch (_error) {
      continue
    }

    const parts = landingPagePathParts(parsed.pathname)
    if (!parts) continue

    const path = parsed.pathname
    const loc = `${siteBase.origin}${path}`
    if (seen.has(loc)) continue
    seen.add(loc)
    entries.push({
      loc,
      path,
      id: decodeURIComponent(parts.id),
      slug: decodeURIComponent(parts.slug),
      priority: '0.7',
      changefreq: 'weekly',
    })
  }

  return entries
}

export function landingPageSitemapUrlsFromXml(xml, publicSiteUrl) {
  return landingPageSitemapEntriesFromXml(xml, publicSiteUrl).map((entry) => ({
    loc: entry.loc,
    priority: entry.priority,
    changefreq: entry.changefreq,
  }))
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

function landingPagePublicApiUrl(apiBaseUrl, id) {
  const base = stripTrailingSlash(clean(apiBaseUrl))
  if (!base) {
    throw new Error('Landing-page public prerender requires VITE_API_BASE or a sitemap URL origin')
  }
  return `${base}/api/v1/content-assets/landing_page/public/${encodeURIComponent(id)}`
}

export async function fetchLandingPagePrerenderEntries({
  sitemapUrl,
  publicSiteUrl,
  apiBaseUrl,
  fetchImpl = globalThis.fetch,
} = {}) {
  const feedUrl = clean(sitemapUrl)
  if (!feedUrl) return []
  if (typeof fetchImpl !== 'function') {
    throw new Error('Landing-page prerender requires a fetch implementation')
  }

  const sitemapResponse = await fetchImpl(feedUrl)
  if (!sitemapResponse || !sitemapResponse.ok) {
    const status = sitemapResponse?.status ? `HTTP ${sitemapResponse.status}` : 'no response'
    throw new Error(`Failed to fetch generated landing-page sitemap: ${status}`)
  }

  const xml = await sitemapResponse.text()
  const entries = landingPageSitemapEntriesFromXml(xml, publicSiteUrl)
  const resolvedApiBase = clean(apiBaseUrl) || resolveLandingPagePublicApiBase({
    VITE_PUBLIC_LANDING_PAGE_SITEMAP_URL: feedUrl,
  })

  const pages = []
  for (const entry of entries) {
    const response = await fetchImpl(landingPagePublicApiUrl(resolvedApiBase, entry.id))
    if (!response || !response.ok) {
      const status = response?.status ? `HTTP ${response.status}` : 'no response'
      throw new Error(`Failed to fetch generated landing-page ${entry.id}: ${status}`)
    }
    const page = await response.json()
    if (clean(page?.robots).toLowerCase() !== 'index,follow') continue
    pages.push({
      ...entry,
      page,
    })
  }

  return pages
}
