import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

const newRunSource = readFileSync(
  new URL('../src/pages/ContentOpsNewRun.tsx', import.meta.url),
  'utf8',
)

test('new run page exposes saved FAQ source selection for blog and landing outputs', () => {
  assert.ok(newRunSource.includes("const SOURCE_FAQ_IDS_INPUT = 'source_faq_ids'"))
  assert.ok(newRunSource.includes("request.outputs.includes(BLOG_POST_OUTPUT)"))
  assert.ok(newRunSource.includes('landingPageOutputSelected || blogPostOutputSelected'))
  assert.ok(newRunSource.includes('<FaqSourceSelector'))
})

test('saved FAQ source selector writes selected IDs into inputs JSON', () => {
  assert.ok(newRunSource.includes('function updateSourceFaqIdsInputJson'))
  assert.ok(newRunSource.includes('next[SOURCE_FAQ_IDS_INPUT] = values'))
  assert.ok(newRunSource.includes('delete next[SOURCE_FAQ_IDS_INPUT]'))
  assert.ok(newRunSource.includes('handleFaqSourceSelectionChange'))
})

test('saved FAQ source selector uses existing generated-asset drafts API', () => {
  assert.ok(newRunSource.includes('fetchGeneratedAssetDrafts'))
  assert.ok(newRunSource.includes("fetchGeneratedAssetDrafts('faq_markdown'"))
  assert.ok(newRunSource.includes('Saved FAQ report sources'))
  assert.ok(newRunSource.includes('No draft FAQ reports found yet.'))
})

test('saved FAQ source selector keeps filtered selected IDs removable', () => {
  assert.ok(newRunSource.includes('function missingSelectedFaqSourceIds'))
  assert.ok(newRunSource.includes('Selected FAQ report not in recent list'))
  assert.ok(newRunSource.includes('Still included in'))
  assert.ok(newRunSource.includes('Uncheck to remove.'))
})
