import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import ts from 'typescript'

const newRunSource = readFileSync(
  new URL('../src/pages/ContentOpsNewRun.tsx', import.meta.url),
  'utf8',
)

async function loadSourceModeModule() {
  const source = readFileSync(
    new URL('../src/pages/contentOpsSourceMode.ts', import.meta.url),
    'utf8',
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
  parseInputsJsonObject,
  sourceModeDraftValue,
  updateSourceModeInputJson,
} = await loadSourceModeModule()

test('new run page exposes review source selection', () => {
  assert.ok(newRunSource.includes('sourceModeDraftValue(parsedInputsForControls)'))
  assert.ok(newRunSource.includes('updateSourceModeInputJson(inputsJson, mode)'))
  assert.ok(newRunSource.includes('Support tickets'))
  assert.ok(newRunSource.includes('Reviews'))
  assert.ok(newRunSource.includes('Competitive'))
})

test('review source mode writes source type into inputs JSON', () => {
  const updated = updateSourceModeInputJson(
    JSON.stringify({
      source_material_type: 'support_ticket',
      source_faq_ids: ['faq-1'],
      target_account: 'Acme',
    }),
    'reviews',
  )

  assert.equal(updated.ok, true)
  const parsed = JSON.parse(updated.value)
  assert.deepEqual(parsed, {
    source_type: 'reviews',
    target_account: 'Acme',
  })
  assert.equal(sourceModeDraftValue(parseInputsJsonObject(updated.value)), 'reviews')
})

test('support-ticket source mode removes explicit source type but preserves FAQ selection', () => {
  const updated = updateSourceModeInputJson(
    JSON.stringify({
      source_type: 'reviews',
      source_faq_ids: ['faq-1'],
      target_account: 'Acme',
    }),
    'support_ticket',
  )

  assert.equal(updated.ok, true)
  const parsed = JSON.parse(updated.value)
  assert.deepEqual(parsed, {
    source_faq_ids: ['faq-1'],
    target_account: 'Acme',
  })
  assert.equal(sourceModeDraftValue(parseInputsJsonObject(updated.value)), 'support_ticket')
})

test('competitive source mode writes source type and removes FAQ selection', () => {
  const updated = updateSourceModeInputJson(
    JSON.stringify({
      source_material_type: 'support_ticket',
      source_faq_ids: ['faq-1'],
      target_account: 'Acme',
    }),
    'competitive',
  )

  assert.equal(updated.ok, true)
  const parsed = JSON.parse(updated.value)
  assert.deepEqual(parsed, {
    source_type: 'competitive',
    target_account: 'Acme',
  })
  assert.equal(
    sourceModeDraftValue(parseInputsJsonObject('{"source_material_type":"displacement"}')),
    'competitive',
  )
  assert.equal(
    sourceModeDraftValue(parseInputsJsonObject('{"source_type":"competitors"}')),
    'competitive',
  )
})

test('non-support source modes disable support-ticket FAQ source selection in the page', () => {
  assert.ok(newRunSource.includes("sourceMode === 'support_ticket'"))
})
