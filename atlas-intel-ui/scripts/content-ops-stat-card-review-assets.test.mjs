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
  exportGeneratedAssetDraftsHtml,
  fetchGeneratedAssetDrafts,
  reviewGeneratedAssetDraft,
  reviewGeneratedAssetDrafts,
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
const newRunSource = readFileSync(
  new URL('../src/pages/ContentOpsNewRun.tsx', import.meta.url),
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

function installTextFetchResponder(payload, status = 200) {
  const calls = []
  globalThis.fetch = async (url, init = {}) => {
    calls.push({ url: String(url), init })
    return new Response(payload, {
      status,
      headers: { 'content-type': 'text/html' },
    })
  }
  return calls
}

test.beforeEach(() => {
  installBrowserStubs()
})

test('stat-card generated asset list fetches the typed content-assets route', async () => {
  const payload = {
    count: 1,
    limit: 5,
    filters: { status: 'draft', theme: 'customer_metric' },
    rows: [{
      id: 'stat-card-1',
      status: 'draft',
      theme: 'customer_metric',
      metric_label: 'NPS score',
      metric_value: 42,
      metric_display: '42',
      claim: 'NPS score: 42',
      headline: 'Customer metric for Zendesk',
      supporting_text: 'Use this stat to frame renewal dissatisfaction.',
      evidence: 'NPS score dropped to 42 after renewal.',
    }],
  }
  const calls = installFetchResponder(payload)

  const result = await fetchGeneratedAssetDrafts('stat_card', {
    status: 'draft',
    theme: 'customer_metric',
    limit: 5,
  })

  assert.deepEqual(result, payload)
  assert.equal(calls.length, 1)
  const [{ url, init }] = calls
  assert.equal(
    url,
    `${API_ORIGIN}/api/v1/content-assets/stat_card/drafts?status=draft&theme=customer_metric&limit=5`,
  )
  assert.equal(init.method, undefined)
  assert.deepEqual(init.headers, { Authorization: 'Bearer test-token' })
})

test('stat-card generated asset review actions post to typed routes', async () => {
  const calls = installFetchResponder({
    account_id: 'acct-1',
    asset: 'stat_card',
    id: 'stat-card-1',
    status: 'approved',
    updated: true,
  })

  await reviewGeneratedAssetDraft('stat_card', 'stat-card-1', 'approved')
  await reviewGeneratedAssetDrafts('stat_card', ['stat-card-1'], 'rejected')

  assert.equal(calls.length, 2)
  assert.equal(
    calls[0].url,
    `${API_ORIGIN}/api/v1/content-assets/stat_card/drafts/review`,
  )
  assert.deepEqual(JSON.parse(calls[0].init.body), {
    id: 'stat-card-1',
    status: 'approved',
  })
  assert.equal(
    calls[1].url,
    `${API_ORIGIN}/api/v1/content-assets/stat_card/drafts/review-batch`,
  )
  assert.deepEqual(JSON.parse(calls[1].init.body), {
    ids: ['stat-card-1'],
    status: 'rejected',
  })
})

test('stat-card visual export fetches html from the typed content-assets route', async () => {
  const calls = installTextFetchResponder('<!doctype html><h1>Stat Cards</h1>')

  const result = await exportGeneratedAssetDraftsHtml('stat_card', {
    status: 'approved',
    limit: 20,
  })

  assert.equal(result, '<!doctype html><h1>Stat Cards</h1>')
  assert.equal(calls.length, 1)
  const [{ url, init }] = calls
  assert.equal(
    url,
    `${API_ORIGIN}/api/v1/content-assets/stat_card/drafts/export?status=approved&limit=20&format=html`,
  )
  assert.equal(init.method, undefined)
  assert.deepEqual(init.headers, { Authorization: 'Bearer test-token' })
})

test('asset review UI exposes stat-card tab and preview branches', () => {
  assert.ok(reviewSource.includes("id: 'stat_card'"))
  assert.ok(reviewSource.includes("label: 'Stat Cards'"))
  assert.ok(reviewSource.includes("asset === 'stat_card'"))
  assert.ok(reviewSource.includes('assetSupportsVisualExport(asset)'))
  assert.ok(reviewSource.includes('Export HTML'))
  assert.ok(reviewSource.includes('exportGeneratedAssetDraftsHtml(asset, params)'))
  assert.ok(reviewSource.includes('textValue(row.claim)'))
  assert.ok(reviewSource.includes('textValue(row.evidence)'))
  assert.ok(reviewSource.includes('statCardPainPointCount(row)'))
  assert.ok(reviewSource.includes('pain: ${item}'))
})

test('completed stat-card runs link into generated asset review', () => {
  assert.match(
    newRunSource,
    /const GENERATED_ASSET_OUTPUTS:[\s\S]*'stat_card'[\s\S]*\]/,
  )
  assert.ok(newRunSource.includes('function generatedAssetReviewHref'))
  assert.ok(newRunSource.includes("params.set('asset', output)"))
  assert.ok(newRunSource.includes('Review generated drafts'))
})
