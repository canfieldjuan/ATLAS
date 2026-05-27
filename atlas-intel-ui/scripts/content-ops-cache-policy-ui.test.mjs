import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import ts from 'typescript'

async function loadTsModule(path) {
  const source = readFileSync(new URL(path, import.meta.url), 'utf8')
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
  fromWirePreview,
  fromWireRequest,
  toWireRequest,
} = await loadTsModule('../src/domain/contentOps/fromWire.ts')

const newRunSource = readFileSync(
  new URL('../src/pages/ContentOpsNewRun.tsx', import.meta.url),
  'utf8',
)

test('cache policy request field round-trips through domain mapping', () => {
  const domain = fromWireRequest({
    target_mode: 'vendor_retention',
    outputs: ['landing_page'],
    content_ops_cache_policy: 'exact',
    inputs: { target_keyword: 'support tickets' },
  })

  assert.equal(domain.contentOpsCachePolicy, 'exact')

  const wire = toWireRequest(domain)
  assert.equal(wire.content_ops_cache_policy, 'exact')
})

test('cache policy defaults to no explicit request', () => {
  const domain = fromWireRequest({})

  assert.equal(domain.contentOpsCachePolicy, null)
  assert.equal(toWireRequest(domain).content_ops_cache_policy, null)
})

test('preview mapper preserves normalized cache policy', () => {
  const preview = fromWirePreview({
    can_run: true,
    outputs: ['blog_post'],
    estimated_cost_usd: 0.75,
    missing_inputs: [],
    blocked_outputs: [],
    warnings: [],
    normalized_request: {
      outputs: ['blog_post'],
      content_ops_cache_policy: 'no_store',
    },
  })

  assert.equal(preview.normalizedRequest?.contentOpsCachePolicy, 'no_store')
})

test('new run options panel exposes cache policy control', () => {
  assert.ok(newRunSource.includes('Cache policy'))
  assert.ok(newRunSource.includes('contentOpsCachePolicy'))
  assert.ok(newRunSource.includes('<option value="exact">Exact cache</option>'))
  assert.ok(newRunSource.includes('<option value="no_store">No store</option>'))
})
