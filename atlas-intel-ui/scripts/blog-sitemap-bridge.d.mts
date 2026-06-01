export function resolveGeneratedBlogPostsUrl(env?: Record<string, string | undefined>): string

export function fetchGeneratedBlogSitemapUrls(args?: {
  postsUrl?: string
  publicSiteUrl?: string
  excludeSlugs?: string[]
  fetchImpl?: typeof fetch
  logger?: Pick<Console, 'warn'>
}): Promise<Array<{
  loc: string
  lastmod: string
  priority: string
  changefreq: string
}>>

export function fetchGeneratedBlogPrerenderEntries(args?: {
  postsUrl?: string
  publicSiteUrl?: string
  excludeSlugs?: string[]
  fetchImpl?: typeof fetch
  logger?: Pick<Console, 'warn'>
}): Promise<unknown[]>
