import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

const newRunSource = readFileSync(
  new URL('../src/pages/ContentOpsNewRun.tsx', import.meta.url),
  'utf8',
)
const reviewSource = readFileSync(
  new URL('../src/pages/ContentOpsAssetsReview.tsx', import.meta.url),
  'utf8',
)
const apiSource = readFileSync(
  new URL('../src/api/contentOps.ts', import.meta.url),
  'utf8',
)

test('landing-page execution summary links saved drafts to asset review', () => {
  assert.ok(newRunSource.includes('function generatedAssetReviewHref'))
  assert.ok(newRunSource.includes('if (idFiltered && savedIds.length === 0) return null'))
  assert.ok(newRunSource.includes("params.set('asset', output)"))
  assert.ok(newRunSource.includes("params.set('status', 'draft')"))
  assert.ok(newRunSource.includes("params.append('id', id)"))
  assert.ok(newRunSource.includes('<Link'))
  assert.ok(newRunSource.includes('Review generated drafts'))
})

test('generated asset review initializes from landing-page deep-link params', () => {
  assert.ok(reviewSource.includes('useSearchParams'))
  assert.ok(reviewSource.includes('assetFromSearchParams(searchParams)'))
  assert.ok(reviewSource.includes('statusFromSearchParams(searchParams)'))
  assert.ok(reviewSource.includes('idFiltersFromSearchParams(searchParams)'))
  assert.ok(reviewSource.includes('assetSupportsIdFilter(asset) && focusedIds.length > 0'))
})

test('asset review passes focused ids through the existing drafts API', () => {
  assert.ok(reviewSource.includes('id: assetSupportsIdFilter(asset) && focusedIds.length > 0'))
  assert.ok(reviewSource.includes('fetchGeneratedAssetDrafts(asset, params)'))
  assert.ok(reviewSource.includes('Showing {focusedIds.length} generated draft'))
  assert.ok(reviewSource.includes('Show latest {activeAsset.label.toLowerCase()}'))
})

test('frontend asset API serializes id arrays as repeated query keys', () => {
  assert.ok(apiSource.includes('id?: string | string[]'))
  assert.ok(apiSource.includes('Array.isArray(value)'))
  assert.ok(apiSource.includes('search.append(key, String(item))'))
})
