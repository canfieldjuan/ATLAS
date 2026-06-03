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

test.beforeEach(() => {
  installBrowserStubs()
})

test('social-post generated asset list fetches the typed content-assets route', async () => {
  const payload = {
    count: 1,
    limit: 5,
    filters: { status: 'draft', channel: 'linkedin' },
    rows: [{
      id: 'social-post-1',
      status: 'draft',
      channel: 'linkedin',
      text: 'Acme support teams can spot churn risk before renewal.',
    }],
  }
  const calls = installFetchResponder(payload)

  const result = await fetchGeneratedAssetDrafts('social_post', {
    status: 'draft',
    channel: 'linkedin',
    limit: 5,
  })

  assert.deepEqual(result, payload)
  assert.equal(calls.length, 1)
  const [{ url, init }] = calls
  assert.equal(
    url,
    `${API_ORIGIN}/api/v1/content-assets/social_post/drafts?status=draft&channel=linkedin&limit=5`,
  )
  assert.equal(init.method, undefined)
  assert.deepEqual(init.headers, { Authorization: 'Bearer test-token' })
})

test('social-post generated asset review actions post to typed routes', async () => {
  const calls = installFetchResponder({
    account_id: 'acct-1',
    asset: 'social_post',
    id: 'social-post-1',
    status: 'approved',
    updated: true,
  })

  await reviewGeneratedAssetDraft('social_post', 'social-post-1', 'approved')
  await reviewGeneratedAssetDrafts('social_post', ['social-post-1'], 'rejected')

  assert.equal(calls.length, 2)
  assert.equal(
    calls[0].url,
    `${API_ORIGIN}/api/v1/content-assets/social_post/drafts/review`,
  )
  assert.deepEqual(JSON.parse(calls[0].init.body), {
    id: 'social-post-1',
    status: 'approved',
  })
  assert.equal(
    calls[1].url,
    `${API_ORIGIN}/api/v1/content-assets/social_post/drafts/review-batch`,
  )
  assert.deepEqual(JSON.parse(calls[1].init.body), {
    ids: ['social-post-1'],
    status: 'rejected',
  })
})

test('asset review UI exposes social-post tab and text preview branches', () => {
  assert.ok(reviewSource.includes("id: 'social_post'"))
  assert.ok(reviewSource.includes("label: 'Social Posts'"))
  assert.ok(reviewSource.includes("asset === 'social_post'"))
  assert.ok(reviewSource.includes('textValue(row.text)'))
  assert.ok(reviewSource.includes('socialPostPainPointCount(row)'))
  assert.ok(reviewSource.includes("pain: ${item}"))
})

test('completed social-post runs link into generated asset review', () => {
  assert.match(
    newRunSource,
    /const GENERATED_ASSET_OUTPUTS:[\s\S]*'social_post'[\s\S]*\]/,
  )
  assert.ok(newRunSource.includes('function generatedAssetReviewHref'))
  assert.ok(newRunSource.includes("params.set('asset', output)"))
  assert.ok(newRunSource.includes('Review generated drafts'))
})
