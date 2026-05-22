import assert from 'node:assert/strict'
import test from 'node:test'

import {
  fetchLandingPagePrerenderEntries,
  fetchLandingPageSitemapUrls,
  landingPageSitemapEntriesFromXml,
  landingPageSitemapUrlsFromXml,
  resolveLandingPagePublicApiBase,
  resolveLandingPageSitemapUrl,
} from './landing-page-sitemap-bridge.mjs'

const SITE_URL = 'https://atlas-intel-ui-two.vercel.app'

test('resolveLandingPageSitemapUrl uses explicit feed URL only', () => {
  assert.equal(
    resolveLandingPageSitemapUrl({
      VITE_PUBLIC_LANDING_PAGE_SITEMAP_URL:
        ' https://api.example.com/content-assets/landing_page/public/sitemap.xml ',
      VITE_API_BASE: 'https://api.example.com',
    }),
    'https://api.example.com/content-assets/landing_page/public/sitemap.xml',
  )
  assert.equal(
    resolveLandingPageSitemapUrl({ VITE_API_BASE: 'https://api.example.com' }),
    '',
  )
})

test('resolveLandingPagePublicApiBase prefers VITE_API_BASE and falls back to sitemap origin', () => {
  assert.equal(
    resolveLandingPagePublicApiBase({
      VITE_API_BASE: ' https://api.example.com/ ',
      VITE_PUBLIC_LANDING_PAGE_SITEMAP_URL:
        'https://other.example.com/api/v1/content-assets/landing_page/public/sitemap.xml',
    }),
    'https://api.example.com',
  )
  assert.equal(
    resolveLandingPagePublicApiBase({
      VITE_PUBLIC_LANDING_PAGE_SITEMAP_URL:
        'https://api.example.com/api/v1/content-assets/landing_page/public/sitemap.xml',
    }),
    'https://api.example.com',
  )
  assert.equal(resolveLandingPagePublicApiBase({}), '')
})

test('landingPageSitemapEntriesFromXml exposes lp path, id, and slug', () => {
  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<urlset>
  <url><loc>https://api.example.com/lp/111/acme-page</loc></url>
  <url><loc>https://api.example.com/lp/222/second-page?variant=a&amp;b=1</loc></url>
</urlset>`

  assert.deepEqual(landingPageSitemapEntriesFromXml(xml, SITE_URL), [
    {
      loc: `${SITE_URL}/lp/111/acme-page`,
      path: '/lp/111/acme-page',
      id: '111',
      slug: 'acme-page',
      priority: '0.7',
      changefreq: 'weekly',
    },
    {
      loc: `${SITE_URL}/lp/222/second-page`,
      path: '/lp/222/second-page',
      id: '222',
      slug: 'second-page',
      priority: '0.7',
      changefreq: 'weekly',
    },
  ])
})

test('landingPageSitemapUrlsFromXml keeps lp paths and rewrites to public site', () => {
  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<urlset>
  <url><loc>https://api.example.com/lp/111/acme-page</loc></url>
  <url><loc>https://api.example.com/blog/not-imported</loc></url>
  <url><loc>https://api.example.com/lp/222/second-page?variant=a&amp;b=1</loc></url>
</urlset>`

  assert.deepEqual(landingPageSitemapUrlsFromXml(xml, SITE_URL), [
    {
      loc: `${SITE_URL}/lp/111/acme-page`,
      priority: '0.7',
      changefreq: 'weekly',
    },
    {
      loc: `${SITE_URL}/lp/222/second-page`,
      priority: '0.7',
      changefreq: 'weekly',
    },
  ])
})

test('landingPageSitemapUrlsFromXml dedupes and skips invalid loc entries', () => {
  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<urlset>
  <url><loc>https://api.example.com/lp/111/acme-page</loc></url>
  <url><loc>not a url</loc></url>
  <url><loc>https://api.example.com/lp/111/acme-page</loc></url>
</urlset>`

  assert.deepEqual(landingPageSitemapUrlsFromXml(xml, `${SITE_URL}/`), [
    {
      loc: `${SITE_URL}/lp/111/acme-page`,
      priority: '0.7',
      changefreq: 'weekly',
    },
  ])
})

test('fetchLandingPageSitemapUrls fetches and parses configured feed', async () => {
  const urls = await fetchLandingPageSitemapUrls({
    sitemapUrl: 'https://api.example.com/content-assets/landing_page/public/sitemap.xml',
    publicSiteUrl: SITE_URL,
    fetchImpl: async url => {
      assert.equal(
        url,
        'https://api.example.com/content-assets/landing_page/public/sitemap.xml',
      )
      return {
        ok: true,
        async text() {
          return '<urlset><url><loc>https://api.example.com/lp/111/acme-page</loc></url></urlset>'
        },
      }
    },
  })

  assert.equal(urls.length, 1)
  assert.equal(urls[0].loc, `${SITE_URL}/lp/111/acme-page`)
})

test('fetchLandingPageSitemapUrls is a no-op without a feed URL', async () => {
  const urls = await fetchLandingPageSitemapUrls({
    sitemapUrl: '',
    publicSiteUrl: SITE_URL,
    fetchImpl: async () => {
      throw new Error('fetch should not run')
    },
  })

  assert.deepEqual(urls, [])
})

test('fetchLandingPageSitemapUrls fails when configured feed is unavailable', async () => {
  await assert.rejects(
    fetchLandingPageSitemapUrls({
      sitemapUrl: 'https://api.example.com/content-assets/landing_page/public/sitemap.xml',
      publicSiteUrl: SITE_URL,
      fetchImpl: async () => ({ ok: false, status: 503 }),
    }),
    /Failed to fetch generated landing-page sitemap: HTTP 503/,
  )
})

test('fetchLandingPagePrerenderEntries fetches public page payloads from api base', async () => {
  const calls = []
  const entries = await fetchLandingPagePrerenderEntries({
    sitemapUrl: 'https://api.example.com/api/v1/content-assets/landing_page/public/sitemap.xml',
    publicSiteUrl: SITE_URL,
    apiBaseUrl: 'https://api.example.com',
    fetchImpl: async url => {
      calls.push(url)
      if (url.endsWith('/sitemap.xml')) {
        return {
          ok: true,
          async text() {
            return '<urlset><url><loc>https://api.example.com/lp/111/acme-page</loc></url></urlset>'
          },
        }
      }
      assert.equal(
        url,
        'https://api.example.com/api/v1/content-assets/landing_page/public/111',
      )
      return {
        ok: true,
        async json() {
          return {
            id: '111',
            slug: 'acme-page',
            title: 'Acme Page',
            robots: 'index,follow',
          }
        },
      }
    },
  })

  assert.deepEqual(calls, [
    'https://api.example.com/api/v1/content-assets/landing_page/public/sitemap.xml',
    'https://api.example.com/api/v1/content-assets/landing_page/public/111',
  ])
  assert.equal(entries.length, 1)
  assert.equal(entries[0].path, '/lp/111/acme-page')
  assert.equal(entries[0].page.title, 'Acme Page')
})

test('fetchLandingPagePrerenderEntries skips noindex public page payloads', async () => {
  const entries = await fetchLandingPagePrerenderEntries({
    sitemapUrl: 'https://api.example.com/api/v1/content-assets/landing_page/public/sitemap.xml',
    publicSiteUrl: SITE_URL,
    apiBaseUrl: 'https://api.example.com',
    fetchImpl: async url => {
      if (url.endsWith('/sitemap.xml')) {
        return {
          ok: true,
          async text() {
            return '<urlset><url><loc>https://api.example.com/lp/111/acme-page</loc></url></urlset>'
          },
        }
      }
      return {
        ok: true,
        async json() {
          return {
            id: '111',
            slug: 'acme-page',
            title: 'Acme Page',
            robots: 'noindex,follow',
          }
        },
      }
    },
  })

  assert.deepEqual(entries, [])
})
