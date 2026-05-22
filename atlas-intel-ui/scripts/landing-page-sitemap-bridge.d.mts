export interface SitemapUrl {
  loc: string
  lastmod?: string
  priority: string
  changefreq: string
}

export function resolveLandingPageSitemapUrl(
  env?: Record<string, string | undefined>,
): string

export function landingPageSitemapUrlsFromXml(
  xml: string,
  publicSiteUrl: string,
): SitemapUrl[]

export function fetchLandingPageSitemapUrls(options?: {
  sitemapUrl?: string
  publicSiteUrl: string
  fetchImpl?: typeof fetch
}): Promise<SitemapUrl[]>
