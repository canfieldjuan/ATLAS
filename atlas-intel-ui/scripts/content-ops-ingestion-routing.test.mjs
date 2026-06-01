import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import ts from 'typescript'

const API_ORIGIN = 'https://api.example.test'

function loadContentOpsApiModule() {
  const source = readFileSync(
    new URL('../src/api/contentOps.ts', import.meta.url),
    'utf8',
  )
    .replace(
      "import { tryRefreshToken } from '../auth/AuthContext'\n",
      'const tryRefreshToken = async () => null\n',
    )
    .replace(
      "import { API_BASE } from './config'\n",
      `const API_BASE = '${API_ORIGIN}'\n`,
    )

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
  importContentOpsIngestion,
  importContentOpsIngestionFile,
  inspectContentOpsIngestion,
  inspectContentOpsIngestionFile,
} = await loadContentOpsApiModule()

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

function diagnosticsPayload() {
  return {
    ok: true,
    mode: 'source_rows',
    source: 'tickets.csv',
    opportunity_count: 2,
    warning_count: 0,
    warning_counts: {},
    missing_field_counts: {},
    source_type_counts: { ticket: 2 },
    samples: [],
    warnings: [],
  }
}

function importPayload() {
  return {
    diagnostics: diagnosticsPayload(),
    import: {
      inserted: 2,
      skipped: 0,
      dry_run: false,
      replace_existing: true,
      target_ids: ['ticket-1', 'ticket-2'],
      warnings: [],
      source: 'tickets.csv',
    },
  }
}

function assertAuthHeader(headers) {
  assert.equal(headers.Authorization, 'Bearer test-token')
}

test.beforeEach(() => {
  installBrowserStubs()
})

test('uploaded file inspect uses multipart file endpoint', async () => {
  const calls = installFetchResponder(diagnosticsPayload())
  const file = new File(['subject,body\nSetup,How do I set this up?\n'], 'tickets.csv', {
    type: 'text/csv',
  })

  await inspectContentOpsIngestionFile({
    file,
    source_rows: true,
    source: 'tickets.csv',
    target_mode: 'vendor_retention',
    file_format: 'csv',
    max_source_text_chars: 900,
    sample_limit: 5,
    include_source_material: true,
    default_fields: { company_name: 'Acme' },
  })

  assert.equal(calls.length, 1)
  const [{ url, init }] = calls
  assert.equal(
    url,
    `${API_ORIGIN}/api/v1/content-ops/ingestion/files/inspect`,
  )
  assert.equal(init.method, 'POST')
  assertAuthHeader(init.headers)
  assert.equal(init.headers['Content-Type'], undefined)
  assert.ok(init.body instanceof FormData)
  assert.equal(init.body.get('file').name, 'tickets.csv')
  assert.equal(init.body.get('source_rows'), 'true')
  assert.equal(init.body.get('source'), 'tickets.csv')
  assert.equal(init.body.get('target_mode'), 'vendor_retention')
  assert.equal(init.body.get('file_format'), 'csv')
  assert.equal(init.body.get('max_source_text_chars'), '900')
  assert.equal(init.body.get('sample_limit'), '5')
  assert.equal(init.body.get('include_source_material'), 'true')
  assert.equal(
    init.body.get('default_fields'),
    JSON.stringify({ company_name: 'Acme' }),
  )
})

test('inline inspect uses deprecated JSON compatibility endpoint', async () => {
  const calls = installFetchResponder(diagnosticsPayload())

  await inspectContentOpsIngestion({
    rows: [{ target_id: 'ticket-1', text: 'How do I set this up?' }],
    source_rows: true,
    source: 'manual-paste',
    target_mode: 'vendor_retention',
    max_source_text_chars: 900,
    sample_limit: 5,
    default_fields: { company_name: 'Acme' },
    include_source_material: true,
  })

  assert.equal(calls.length, 1)
  const [{ url, init }] = calls
  assert.equal(url, `${API_ORIGIN}/api/v1/content-ops/ingestion/inspect`)
  assert.equal(init.method, 'POST')
  assert.deepEqual(init.headers, {
    'Content-Type': 'application/json',
    Authorization: 'Bearer test-token',
  })
  assert.deepEqual(JSON.parse(init.body), {
    rows: [{ target_id: 'ticket-1', text: 'How do I set this up?' }],
    source_rows: true,
    source: 'manual-paste',
    target_mode: 'vendor_retention',
    max_source_text_chars: 900,
    sample_limit: 5,
    default_fields: { company_name: 'Acme' },
    include_source_material: true,
  })
})

test('uploaded file import uses multipart file endpoint', async () => {
  const calls = installFetchResponder(importPayload())
  const file = new File(['subject,body\nBilling,Why was I charged?\n'], 'tickets.csv', {
    type: 'text/csv',
  })

  const outcome = await importContentOpsIngestionFile({
    file,
    source_rows: true,
    source: 'tickets.csv',
    target_mode: 'vendor_retention',
    file_format: 'csv',
    max_source_text_chars: 1200,
    sample_limit: 3,
    include_source_material: true,
    default_fields: { company_name: 'Acme' },
    replace_existing: true,
    dry_run: false,
  })

  assert.equal(outcome.kind, 'success')
  assert.equal(calls.length, 1)
  const [{ url, init }] = calls
  assert.equal(
    url,
    `${API_ORIGIN}/api/v1/content-ops/ingestion/files/import`,
  )
  assert.equal(init.method, 'POST')
  assertAuthHeader(init.headers)
  assert.equal(init.headers['Content-Type'], undefined)
  assert.ok(init.body instanceof FormData)
  assert.equal(init.body.get('file').name, 'tickets.csv')
  assert.equal(init.body.get('include_source_material'), 'true')
  assert.equal(init.body.get('replace_existing'), 'true')
  assert.equal(init.body.get('dry_run'), 'false')
})

test('inline import uses deprecated JSON compatibility endpoint', async () => {
  const calls = installFetchResponder(importPayload())

  const outcome = await importContentOpsIngestion({
    rows: [{ target_id: 'ticket-2', text: 'Why was I charged?' }],
    source_rows: true,
    source: 'manual-paste',
    target_mode: 'vendor_retention',
    max_source_text_chars: 1200,
    sample_limit: 3,
    default_fields: { company_name: 'Acme' },
    include_source_material: true,
    replace_existing: true,
    dry_run: false,
  })

  assert.equal(outcome.kind, 'success')
  assert.equal(calls.length, 1)
  const [{ url, init }] = calls
  assert.equal(url, `${API_ORIGIN}/api/v1/content-ops/ingestion/import`)
  assert.equal(init.method, 'POST')
  assert.deepEqual(init.headers, {
    'Content-Type': 'application/json',
    Authorization: 'Bearer test-token',
  })
  assert.deepEqual(JSON.parse(init.body), {
    rows: [{ target_id: 'ticket-2', text: 'Why was I charged?' }],
    source_rows: true,
    source: 'manual-paste',
    target_mode: 'vendor_retention',
    max_source_text_chars: 1200,
    sample_limit: 3,
    default_fields: { company_name: 'Acme' },
    include_source_material: true,
    replace_existing: true,
    dry_run: false,
  })
})
