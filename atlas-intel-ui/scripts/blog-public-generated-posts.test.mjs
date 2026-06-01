import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

const blogApiSource = readFileSync(
  new URL('../src/api/blog.ts', import.meta.url),
  'utf8',
)
const blogPageSource = readFileSync(
  new URL('../src/pages/Blog.tsx', import.meta.url),
  'utf8',
)
const blogPostPageSource = readFileSync(
  new URL('../src/pages/BlogPost.tsx', import.meta.url),
  'utf8',
)
const publicLandingPageSource = readFileSync(
  new URL('../src/pages/PublicLandingPage.tsx', import.meta.url),
  'utf8',
)
const safeMarkdownSource = readFileSync(
  new URL('../src/lib/safeMarkdown.ts', import.meta.url),
  'utf8',
)

test('public blog adapter fetches list and detail from backend public routes', () => {
  assert.ok(blogApiSource.includes("const BASE = `${API_BASE}/api/v1/blog`"))
  assert.ok(blogApiSource.includes("getPublicBlogJson<PublicBlogListWire>('/published')"))
  assert.ok(blogApiSource.includes("`/published/${encodeURIComponent(slug)}`"))
  assert.ok(blogApiSource.includes('publicBlogPostFromWire'))
})

test('blog index merges generated posts ahead of static fallback posts', () => {
  assert.ok(blogPageSource.includes('fetchPublicBlogPosts'))
  assert.ok(blogPageSource.includes('useState<BlogPost[]>(POSTS)'))
  assert.ok(blogPageSource.includes('setPosts(mergeBlogPosts(generatedPosts, POSTS))'))
  assert.ok(blogPageSource.includes('function mergeBlogPosts'))
})

test('blog detail fetches generated post only when slug is not static', () => {
  assert.ok(blogPostPageSource.includes('const staticPost = POSTS.find'))
  assert.ok(blogPostPageSource.includes('const post = staticPost ?? generatedPost'))
  assert.ok(blogPostPageSource.includes('if (!slug || staticPost)'))
  assert.ok(blogPostPageSource.includes('fetchPublicBlogPost(slug)'))
  assert.ok(blogPostPageSource.includes('loadingGeneratedPost'))
  assert.ok(!blogPostPageSource.includes('setGeneratedPost(null)'))
  assert.ok(!blogPostPageSource.includes('setLoadingGeneratedPost(false)'))
})

test('blog detail sanitizes rendered markdown before html injection', () => {
  assert.ok(
    blogPostPageSource.includes(
      "import { renderSafeMarkdown, sanitizeRenderedHtml } from '../lib/safeMarkdown'",
    ),
  )
  assert.ok(
    publicLandingPageSource.includes("import { renderSafeMarkdown } from '../lib/safeMarkdown'"),
  )
  assert.ok(blogPostPageSource.includes('renderContentWithCharts(post.content, post.charts, !staticPost)'))
  assert.ok(safeMarkdownSource.includes('markdown.replace(/[<>]/g'))
  assert.ok(safeMarkdownSource.includes('char === \'<\' ? \'&lt;\' : \'&gt;\''))
  assert.ok(safeMarkdownSource.includes("element.removeAttribute(attr.name)"))
  assert.ok(safeMarkdownSource.includes("name.startsWith('on')"))
  assert.ok(safeMarkdownSource.includes("!safeUrl(attr.value)"))
  assert.ok(!blogPostPageSource.includes('const UNSAFE_HTML_RE'))
  assert.ok(!blogPostPageSource.includes('function sanitizeRenderedHtml'))
  assert.ok(!publicLandingPageSource.includes('function renderSafeMarkdown'))
})
