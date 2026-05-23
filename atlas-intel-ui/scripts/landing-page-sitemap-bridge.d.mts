export interface SitemapUrl {
  loc: string
  lastmod?: string
  priority: string
  changefreq: string
}

export interface LandingPageSitemapEntry extends SitemapUrl {
  path: string
  id: string
  slug: string
}

export interface LandingPagePrerenderEntry extends LandingPageSitemapEntry {
  page: Record<string, unknown>
}

export function resolveLandingPageSitemapUrl(
  env?: Record<string, string | undefined>,
): string

export function resolveLandingPagePublicApiBase(
  env?: Record<string, string | undefined>,
): string

export function landingPageSitemapEntriesFromXml(
  xml: string,
  publicSiteUrl: string,
): LandingPageSitemapEntry[]

export function landingPageSitemapUrlsFromXml(
  xml: string,
  publicSiteUrl: string,
): SitemapUrl[]

export function fetchLandingPageSitemapUrls(options?: {
  sitemapUrl?: string
  publicSiteUrl: string
  fetchImpl?: typeof fetch
}): Promise<SitemapUrl[]>

export function fetchLandingPagePrerenderEntries(options?: {
  sitemapUrl?: string
  publicSiteUrl: string
  apiBaseUrl?: string
  fetchImpl?: typeof fetch
}): Promise<LandingPagePrerenderEntry[]>
