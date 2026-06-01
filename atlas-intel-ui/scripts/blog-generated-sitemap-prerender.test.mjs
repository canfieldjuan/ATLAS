import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

import {
  fetchGeneratedBlogPrerenderEntries,
  fetchGeneratedBlogSitemapUrls,
  resolveGeneratedBlogPostsUrl,
} from './blog-sitemap-bridge.mjs'

const SITE_URL = 'https://atlas-intel-ui-two.vercel.app'
const viteConfigSource = readFileSync(
  new URL('../vite.config.ts', import.meta.url),
  'utf-8',
)

function post(overrides = {}) {
  return {
    slug: 'generated-post',
    title: 'Generated Post',
    description: 'A generated post.',
    date: '2026-06-01T12:00:00Z',
    author: 'Atlas Intelligence',
    tags: ['content-ops'],
    content: '## Summary\n\nGenerated body with {{chart:demo-chart}}.',
    charts: [{
      chart_id: 'demo-chart',
      chart_type: 'bar',
      title: 'Demo Chart',
      data: [{ label: 'A', value: 1 }],
      config: { x_key: 'label', bars: [{ dataKey: 'value' }] },
    }],
    faq: [{ question: 'Can this be indexed?', answer: 'Yes.' }],
    seo_title: 'Generated SEO Title',
    seo_description: 'Generated SEO description.',
    ...overrides,
  }
}

function jsonResponse(payload, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    async json() {
      return payload
    },
  }
}

test('resolveGeneratedBlogPostsUrl prefers explicit feed and falls back to API base', () => {
  assert.equal(
    resolveGeneratedBlogPostsUrl({
      VITE_PUBLIC_BLOG_POSTS_URL: ' https://api.example.com/blog-feed ',
      VITE_API_BASE: 'https://fallback.example.com',
    }),
    'https://api.example.com/blog-feed',
  )
  assert.equal(
    resolveGeneratedBlogPostsUrl({ VITE_API_BASE: 'https://api.example.com/' }),
    'https://api.example.com/api/v1/blog/published?limit=200',
  )
  assert.equal(resolveGeneratedBlogPostsUrl({}), '')
})

test('fetchGeneratedBlogSitemapUrls maps valid generated posts and dedupes exclusions', async () => {
  const urls = await fetchGeneratedBlogSitemapUrls({
    postsUrl: 'https://api.example.com/api/v1/blog/published?limit=200',
    publicSiteUrl: SITE_URL,
    excludeSlugs: ['static-post'],
    fetchImpl: async (url) => {
      assert.equal(url, 'https://api.example.com/api/v1/blog/published?limit=200')
      return jsonResponse({
        posts: [
          post(),
          post({ slug: 'static-post' }),
          post({ slug: 'generated-post' }),
          post({ slug: 'private-post', robots: 'noindex,follow' }),
        ],
      })
    },
  })

  assert.deepEqual(urls, [{
    loc: `${SITE_URL}/blog/generated-post`,
    lastmod: '2026-06-01',
    priority: '0.7',
    changefreq: 'monthly',
  }])
})

test('fetchGeneratedBlogPrerenderEntries preserves generated post metadata', async () => {
  const entries = await fetchGeneratedBlogPrerenderEntries({
    postsUrl: 'https://api.example.com/api/v1/blog/published',
    publicSiteUrl: SITE_URL,
    fetchImpl: async () => jsonResponse({ posts: [post()] }),
  })

  assert.equal(entries.length, 1)
  assert.equal(entries[0].path, '/blog/generated-post')
  assert.equal(entries[0].loc, `${SITE_URL}/blog/generated-post`)
  assert.equal(entries[0].post.seoTitle, 'Generated SEO Title')
  assert.equal(entries[0].post.seoDescription, 'Generated SEO description.')
  assert.equal(entries[0].post.charts[0].chart_id, 'demo-chart')
  assert.equal(entries[0].post.faq[0].question, 'Can this be indexed?')
})

test('fetchGeneratedBlogPrerenderEntries is a no-op without a feed URL', async () => {
  const entries = await fetchGeneratedBlogPrerenderEntries({
    postsUrl: '',
    publicSiteUrl: SITE_URL,
    fetchImpl: async () => {
      throw new Error('fetch should not run')
    },
  })

  assert.deepEqual(entries, [])
})

test('fetchGeneratedBlogSitemapUrls fails when configured feed is unavailable', async () => {
  await assert.rejects(
    fetchGeneratedBlogSitemapUrls({
      postsUrl: 'https://api.example.com/api/v1/blog/published',
      publicSiteUrl: SITE_URL,
      fetchImpl: async () => jsonResponse({}, 503),
    }),
    /Failed to fetch generated blog posts: HTTP 503/,
  )
})

test('fetchGeneratedBlogSitemapUrls fails closed on malformed response envelope', async () => {
  await assert.rejects(
    fetchGeneratedBlogSitemapUrls({
      postsUrl: 'https://api.example.com/api/v1/blog/published',
      publicSiteUrl: SITE_URL,
      fetchImpl: async () => jsonResponse({ results: [post()] }),
    }),
    /Generated blog response posts must be an array/,
  )
})

test('fetchGeneratedBlogSitemapUrls fails closed on malformed public post', async () => {
  await assert.rejects(
    fetchGeneratedBlogSitemapUrls({
      postsUrl: 'https://api.example.com/api/v1/blog/published',
      publicSiteUrl: SITE_URL,
      fetchImpl: async () => jsonResponse({ posts: [post({ content: '' })] }),
    }),
    /Generated blog post 1 missing content/,
  )
})

test('fetchGeneratedBlogSitemapUrls rejects unsafe generated slugs', async () => {
  await assert.rejects(
    fetchGeneratedBlogSitemapUrls({
      postsUrl: 'https://api.example.com/api/v1/blog/published',
      publicSiteUrl: SITE_URL,
      fetchImpl: async () => jsonResponse({ posts: [post({ slug: '../escape' })] }),
    }),
    /Generated blog post 1 has unsafe slug/,
  )
})

test('vite build wires generated blogs into sitemap and prerender with escaped HTML', () => {
  assert.ok(viteConfigSource.includes('fetchGeneratedBlogSitemapUrls'))
  assert.ok(viteConfigSource.includes('...generatedBlogUrls'))
  assert.ok(viteConfigSource.includes('fetchGeneratedBlogPrerenderEntries'))
  assert.ok(viteConfigSource.includes('...generatedBlogRoutes'))
  assert.ok(viteConfigSource.includes('trustedHtml: false'))
  assert.ok(viteConfigSource.includes('escapeGeneratedMarkdownHtml(post.content)'))
})
