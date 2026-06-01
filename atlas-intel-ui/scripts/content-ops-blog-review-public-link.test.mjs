import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

const reviewSource = readFileSync(
  new URL('../src/pages/ContentOpsAssetsReview.tsx', import.meta.url),
  'utf8',
)

test('asset review detail drawer exposes approved generated blog public URLs', () => {
  assert.ok(reviewSource.includes('const publicUrl = publicAssetUrl(row, asset)'))
  assert.ok(reviewSource.includes('const publicUrlPending = publicAssetUrlPending(row, asset)'))
  assert.ok(reviewSource.includes("if (asset === 'blog_post')"))
  assert.ok(reviewSource.includes("return `${window.location.origin}/blog/${encodeURIComponent(slug)}`"))
  assert.ok(reviewSource.includes("Approved blog posts are available at /blog/:slug."))
  assert.ok(reviewSource.includes("Approve this {asset === 'blog_post' ? 'blog post' : 'landing page'}"))
})

test('asset review keeps landing-page public URLs id and slug based', () => {
  assert.ok(reviewSource.includes("if (asset === 'landing_page')"))
  assert.ok(reviewSource.includes("return `${window.location.origin}/lp/${encodeURIComponent(id)}/${encodeURIComponent(slug)}`"))
  assert.ok(reviewSource.includes("return asset === 'landing_page' && Boolean(assetId(row))"))
  assert.ok(!reviewSource.includes('publicLandingPageUrl('))
})
