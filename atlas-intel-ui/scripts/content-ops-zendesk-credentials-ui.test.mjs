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
  fetchContentOpsZendeskCredentials,
  saveContentOpsZendeskCredential,
  revokeContentOpsZendeskCredential,
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
    const body = status === 204 ? null : JSON.stringify(payload)
    return new Response(body, {
      status,
      headers: { 'content-type': 'application/json' },
    })
  }
  return calls
}

test.beforeEach(() => {
  installBrowserStubs()
})

test('credential list calls tenant Zendesk credential route', async () => {
  const payload = [{
    id: 'credential-1',
    account_id: 'account-1',
    email: 'agent@example.com',
    api_token_prefix: 'tok',
    subdomain: 'acme',
    base_url: '',
    label: 'Primary',
    added_at: '2026-05-29T12:00:00+00:00',
    last_used_at: null,
    revoked_at: null,
  }]
  const calls = installFetchResponder(payload)

  const result = await fetchContentOpsZendeskCredentials()

  assert.deepEqual(result, payload)
  assert.equal(calls.length, 1)
  const [{ url, init }] = calls
  assert.equal(
    url,
    `${API_ORIGIN}/api/v1/content-ops/zendesk-credentials`,
  )
  assert.equal(init.method, undefined)
  assert.deepEqual(init.headers, { Authorization: 'Bearer test-token' })
})

test('credential save posts write-only token payload', async () => {
  const payload = {
    id: 'credential-1',
    account_id: 'account-1',
    email: 'agent@example.com',
    api_token_prefix: 'tok',
    subdomain: 'acme',
    base_url: '',
    label: 'Primary',
    added_at: '2026-05-29T12:00:00+00:00',
    last_used_at: null,
    revoked_at: null,
  }
  const calls = installFetchResponder(payload, 201)

  await saveContentOpsZendeskCredential({
    email: 'agent@example.com',
    api_token: 'secret-token',
    subdomain: 'acme',
    base_url: '',
    label: 'Primary',
  })

  assert.equal(calls.length, 1)
  const [{ url, init }] = calls
  assert.equal(
    url,
    `${API_ORIGIN}/api/v1/content-ops/zendesk-credentials`,
  )
  assert.equal(init.method, 'POST')
  assert.deepEqual(init.headers, {
    'Content-Type': 'application/json',
    Authorization: 'Bearer test-token',
  })
  assert.deepEqual(JSON.parse(init.body), {
    email: 'agent@example.com',
    api_token: 'secret-token',
    subdomain: 'acme',
    base_url: '',
    label: 'Primary',
  })
})

test('credential revoke deletes encoded credential id', async () => {
  const calls = installFetchResponder('', 204)

  await revokeContentOpsZendeskCredential('credential/id')

  assert.equal(calls.length, 1)
  const [{ url, init }] = calls
  assert.equal(
    url,
    `${API_ORIGIN}/api/v1/content-ops/zendesk-credentials/credential%2Fid`,
  )
  assert.equal(init.method, 'DELETE')
  assert.deepEqual(init.headers, { Authorization: 'Bearer test-token' })
})

test('new run page renders Zendesk credential controls', () => {
  assert.ok(newRunSource.includes('function ZendeskCredentialCard'))
  assert.ok(newRunSource.includes('fetchContentOpsZendeskCredentials'))
  assert.ok(newRunSource.includes('saveContentOpsZendeskCredential'))
  assert.ok(newRunSource.includes('revokeContentOpsZendeskCredential'))
  assert.ok(newRunSource.includes('Zendesk macro connection'))
  assert.ok(newRunSource.includes('Save connection'))
  assert.ok(newRunSource.includes('Token prefix'))
})
