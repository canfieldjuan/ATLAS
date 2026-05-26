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
  fetchContentOpsTenantUsageSummary,
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

const { fromWireUsageSummary } = await loadTsModule(
  '../src/domain/contentOps/fromWire.ts',
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

function usageSummaryPayload() {
  return {
    period_days: 7,
    filters: { account_id: 'acct_123', asset_type: 'landing_page' },
    summary: {
      total_cost_usd: 1.2345,
      total_calls: 12,
      failed_calls: 1,
      input_tokens: 100,
      billable_input_tokens: 90,
      output_tokens: 50,
      total_tokens: 150,
      cached_tokens: 10,
      cache_write_tokens: 4,
      cache_hit_calls: 3,
      avg_duration_ms: 320.5,
      latest_call_at: '2026-05-26T12:00:00+00:00',
    },
    by_model: [
      {
        provider: 'openrouter',
        model: 'anthropic/claude-3-5-haiku',
        cost_usd: 1.2,
        calls: 10,
        input_tokens: 80,
        output_tokens: 40,
      },
    ],
    by_asset_type: [
      {
        asset_type: 'landing_page',
        cost_usd: 0.8,
        calls: 7,
        input_tokens: 60,
        output_tokens: 25,
      },
    ],
  }
}

test.beforeEach(() => {
  installBrowserStubs()
})

test('tenant usage summary calls scoped route with query params', async () => {
  const calls = installFetchResponder(usageSummaryPayload())

  await fetchContentOpsTenantUsageSummary({
    days: 14,
    asset_type: 'landing_page',
    run_id: 'run-123',
    request_id: 'req-456',
  })

  assert.equal(calls.length, 1)
  const [{ url, init }] = calls
  const parsed = new URL(url)
  assert.equal(
    `${parsed.origin}${parsed.pathname}`,
    `${API_ORIGIN}/api/v1/content-ops/usage/summary/tenant`,
  )
  assert.equal(parsed.searchParams.get('days'), '14')
  assert.equal(parsed.searchParams.get('asset_type'), 'landing_page')
  assert.equal(parsed.searchParams.get('run_id'), 'run-123')
  assert.equal(parsed.searchParams.get('request_id'), 'req-456')
  assert.deepEqual(init.headers, {
    Authorization: 'Bearer test-token',
  })
})

test('fromWireUsageSummary maps usage payload to domain shape', () => {
  const summary = fromWireUsageSummary(usageSummaryPayload())

  assert.equal(summary.periodDays, 7)
  assert.deepEqual(summary.filters, {
    accountId: 'acct_123',
    assetType: 'landing_page',
    runId: undefined,
    requestId: undefined,
  })
  assert.equal(summary.summary.totalCostUsd, 1.2345)
  assert.equal(summary.summary.totalTokens, 150)
  assert.equal(summary.summary.cacheHitCalls, 3)
  assert.equal(summary.byModel[0].model, 'anthropic/claude-3-5-haiku')
  assert.equal(summary.byModel[0].costUsd, 1.2)
  assert.equal(summary.byAssetType[0].assetType, 'landing_page')
  assert.equal(summary.byAssetType[0].calls, 7)
})
