import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import ts from 'typescript'

const newRunSource = readFileSync(
  new URL('../src/pages/ContentOpsNewRun.tsx', import.meta.url),
  'utf8',
)

function loadFromWireModule() {
  const source = readFileSync(
    new URL('../src/domain/contentOps/fromWire.ts', import.meta.url),
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
  fromWireIngestionDiagnostics,
  toWireIngestionImportRequest,
} = await loadFromWireModule()

test('domain mapper preserves full ingestion source material separately from samples', () => {
  const diagnostics = fromWireIngestionDiagnostics({
    ok: true,
    mode: 'source_rows',
    source: 'tickets.csv',
    opportunity_count: 2,
    warning_count: 0,
    warning_counts: {},
    missing_field_counts: {},
    source_type_counts: { support_ticket: 2 },
    samples: [{ target_id: 'ticket-1' }],
    source_material: [
      { target_id: 'ticket-1', text: 'Billing question' },
      { target_id: 'ticket-2', text: 'Setup question' },
    ],
    warnings: [],
  })

  assert.deepEqual(
    diagnostics.samples.map((row) => row.target_id),
    ['ticket-1'],
  )
  assert.deepEqual(
    diagnostics.sourceMaterial.map((row) => row.target_id),
    ['ticket-1', 'ticket-2'],
  )
})

test('domain import request can opt into full source material', () => {
  assert.deepEqual(
    toWireIngestionImportRequest({
      rows: [{ target_id: 'ticket-1' }],
      sourceRows: true,
      source: 'tickets.csv',
      targetMode: 'vendor_retention',
      maxSourceTextChars: 1200,
      sampleLimit: 25,
      defaultFields: { company_name: 'Acme' },
      includeSourceMaterial: true,
      replaceExisting: false,
      dryRun: true,
    }),
    {
      rows: [{ target_id: 'ticket-1' }],
      source_rows: true,
      source: 'tickets.csv',
      target_mode: 'vendor_retention',
      max_source_text_chars: 1200,
      sample_limit: 25,
      default_fields: { company_name: 'Acme' },
      include_source_material: true,
      replace_existing: false,
      dry_run: true,
    },
  )
})

test('new run import handoff applies persisted target ids to the run inputs', () => {
  assert.ok(newRunSource.includes('include_source_material: false'))
  assert.ok(newRunSource.includes('includeSourceMaterial: false'))
  assert.equal(newRunSource.includes('include_source_material: true'), false)
  assert.equal(newRunSource.includes('includeSourceMaterial: true'), false)
  assert.ok(
    newRunSource.includes(
      "const SOURCE_IMPORT_TARGET_IDS_INPUT = 'source_import_target_ids'",
    ),
  )
  assert.ok(newRunSource.includes('function updateSourceImportTargetIdsInputJson'))
  assert.ok(newRunSource.includes('next[SOURCE_IMPORT_TARGET_IDS_INPUT] = values'))
  assert.ok(newRunSource.includes('delete next.source_material'))
  assert.ok(newRunSource.includes('Use targets for run'))
  assert.ok(
    newRunSource.includes(
      'onApplySourceTargetIds(result.targetIds, result.dryRun)',
    ),
  )
  assert.ok(
    newRunSource.includes(
      'disabled={result.dryRun || result.targetIds.length === 0}',
    ),
  )
  assert.ok(
    newRunSource.includes('Dry-run import target IDs are not persisted.'),
  )
  assert.equal(newRunSource.includes('next.source_material = sourceMaterial.map'), false)
  assert.equal(newRunSource.includes('onApplySourceMaterial(sourceMaterial)'), false)
})
