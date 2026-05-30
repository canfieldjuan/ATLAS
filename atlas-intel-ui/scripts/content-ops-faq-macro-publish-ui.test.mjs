import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import ts from 'typescript'

const API_ORIGIN = 'https://api.example.test'

async function loadTsModule(path, replacements = []) {
  let source = readFileSync(new URL(path, import.meta.url), 'utf8')
  for (const [needle, replacement] of replacements) {
    source = source.replace(needle, replacement)
  }
  const compiled = ts.transpileModule(source, {
    compilerOptions: {
      module: ts.ModuleKind.ES2022,
      target: ts.ScriptTarget.ES2022,
      verbatimModuleSyntax: true,
    },
  }).outputText

  const moduleUrl = `data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`
  return import(moduleUrl)
}

const {
  fetchGeneratedFaqMacroPublishAttempts,
  publishGeneratedFaqMacros,
} = await loadTsModule('../src/api/contentOps.ts', [
  [
    "import { tryRefreshToken } from '../auth/AuthContext'\n",
    'const tryRefreshToken = async () => null\n',
  ],
  [
    "import { API_BASE } from './config'\n",
    `const API_BASE = '${API_ORIGIN}'\n`,
  ],
])

const reviewSource = readFileSync(
  new URL('../src/pages/ContentOpsAssetsReview.tsx', import.meta.url),
  'utf8',
)

function installBrowserStubs() {
  Object.defineProperty(globalThis, 'localStorage', {
    configurable: true,
    value: {
      getItem(key) {
        return key === 'atlas_token' ? 'test-token' : null
      },
      removeItem() {},
    },
  })
  Object.defineProperty(globalThis, 'window', {
    configurable: true,
    value: { location: { href: '' } },
  })
}

function installFetchResponder(payload, status = 200) {
  const calls = []
  globalThis.fetch = async (url, init = {}) => {
    calls.push({ url: String(url), init })
    return new Response(JSON.stringify(payload), {
      status,
      headers: { 'content-type': 'application/json' },
    })
  }
  return calls
}

test.beforeEach(() => {
  installBrowserStubs()
})

test('FAQ macro publish posts to encoded generated asset route', async () => {
  const payload = {
    account_id: 'account-1',
    asset: 'faq_markdown',
    faq_id: 'draft/id',
    found: true,
    ok: true,
    draft_status: 'published',
    publishable_count: 2,
    skipped_count: 0,
    published_count: 1,
    updated_count: 1,
    failed_count: 0,
    pending_reconcile_count: 0,
    draft_status_updated: true,
    skipped: [],
    results: [],
  }
  const calls = installFetchResponder(payload)

  const result = await publishGeneratedFaqMacros('draft/id')

  assert.deepEqual(result, payload)
  assert.equal(calls.length, 1)
  const [{ url, init }] = calls
  assert.equal(
    url,
    `${API_ORIGIN}/api/v1/content-assets/faq_markdown/drafts/draft%2Fid/publish-macros`,
  )
  assert.equal(init.method, 'POST')
  assert.deepEqual(init.headers, {
    'Content-Type': 'application/json',
    Authorization: 'Bearer test-token',
  })
  assert.deepEqual(JSON.parse(init.body), {})
})


test('FAQ macro publish history fetches encoded generated asset route', async () => {
  const payload = {
    account_id: 'account-1',
    asset: 'faq_markdown',
    faq_id: 'draft/id',
    count: 1,
    limit: 5,
    attempts: [
      {
        id: 'attempt-1',
        faq_id: 'draft/id',
        draft_status: 'published',
        ok: true,
        publishable_count: 2,
        skipped_count: 0,
        published_count: 1,
        updated_count: 1,
        failed_count: 0,
        pending_reconcile_count: 0,
        draft_status_updated: true,
        skipped: [],
        results: [{ status: 'published', external_id: 'macro-1' }],
        created_at: '2026-05-30T17:40:00Z',
      },
    ],
  }
  const calls = installFetchResponder(payload)

  const result = await fetchGeneratedFaqMacroPublishAttempts('draft/id', { limit: 5 })

  assert.deepEqual(result, payload)
  assert.equal(calls.length, 1)
  const [{ url, init }] = calls
  assert.equal(
    url,
    `${API_ORIGIN}/api/v1/content-assets/faq_markdown/drafts/draft%2Fid/publish-macro-attempts?limit=5`,
  )
  assert.equal(init.method, undefined)
  assert.deepEqual(init.headers, { Authorization: 'Bearer test-token' })
})


test('Generated Asset Review wires FAQ macro publish action and summary state', () => {
  assert.ok(reviewSource.includes('publishGeneratedFaqMacros'))
  assert.ok(reviewSource.includes('handlePublishFaqMacros'))
  assert.ok(reviewSource.includes("asset === 'faq_markdown'"))
  assert.ok(reviewSource.includes("status === 'approved'"))
  assert.ok(reviewSource.includes('Publish macros'))
  assert.ok(reviewSource.includes('MacroPublishResultBanner'))
  assert.ok(reviewSource.includes('fetchGeneratedFaqMacroPublishAttempts'))
  assert.ok(reviewSource.includes('MacroPublishHistoryPanel'))
  assert.ok(reviewSource.includes('Macro publish history'))
  assert.ok(reviewSource.includes('pending_reconcile_count'))
  assert.ok(reviewSource.includes('draft_status_updated'))
})
