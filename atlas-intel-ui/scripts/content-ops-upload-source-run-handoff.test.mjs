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

test('domain mapper preserves ingestion parse errors', () => {
  const diagnostics = fromWireIngestionDiagnostics({
    ok: false,
    mode: 'source_rows',
    source: 'tickets.csv',
    opportunity_count: 0,
    warning_count: 0,
    warning_counts: {},
    missing_field_counts: {},
    source_type_counts: {},
    samples: [],
    warnings: [],
    parse_error: {
      code: 'csv_parse_error',
      message: 'CSV customer data could not be parsed.',
      how_to_fix: 'Export a valid CSV.',
      location: 'source_row_csv',
      row_index: 3,
      line: 3,
      column: 12,
      encoding: 'utf-8',
      byte: 42,
    },
  })

  assert.deepEqual(diagnostics.parseError, {
    code: 'csv_parse_error',
    message: 'CSV customer data could not be parsed.',
    howToFix: 'Export a valid CSV.',
    location: 'source_row_csv',
    rowIndex: 3,
    line: 3,
    column: 12,
    encoding: 'utf-8',
    byte: 42,
  })
})

test('domain mapper preserves source-row admission guidance', () => {
  const diagnostics = fromWireIngestionDiagnostics({
    ok: false,
    mode: 'source_rows',
    source: 'tickets.csv',
    opportunity_count: 0,
    warning_count: 0,
    warning_counts: {},
    missing_field_counts: {},
    source_type_counts: {},
    samples: [],
    warnings: [],
    source_row_admission: {
      input_format: 'csv',
      raw_source_row_count: 1,
      usable_source_row_count: 0,
      usable_source_ratio: 0,
      mapped_fields: { source_id: ['Ticket ID'] },
      ignored_private_fields: ['Internal Notes'],
      populated_unmapped_fields: ['Conversation Text'],
      field_sample_limit: 12,
      admission_decision: {
        status: 'REJECT',
        reason: 'no_usable_source_rows',
        location: 'source_row_csv',
        message: 'No usable support-ticket text was found in this file.',
        how_to_fix: 'Re-export your tickets with customer text.',
      },
      coverage_warnings: [],
    },
  })

  assert.deepEqual(diagnostics.sourceRowAdmission, {
    inputFormat: 'csv',
    rawSourceRowCount: 1,
    usableSourceRowCount: 0,
    usableSourceRatio: 0,
    mappedFields: { source_id: ['Ticket ID'] },
    ignoredPrivateFields: ['Internal Notes'],
    populatedUnmappedFields: ['Conversation Text'],
    fieldSampleLimit: 12,
    admissionDecision: {
      status: 'REJECT',
      reason: 'no_usable_source_rows',
      location: 'source_row_csv',
      message: 'No usable support-ticket text was found in this file.',
      howToFix: 'Re-export your tickets with customer text.',
    },
    coverageWarnings: [],
  })
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

test('new run page renders parser issues from ingestion diagnostics', () => {
  assert.ok(newRunSource.includes('diagnostics.parseError'))
  assert.ok(newRunSource.includes('state.diagnostics.parseError'))
  assert.ok(newRunSource.includes('function IngestionParseErrorNotice'))
  assert.ok(newRunSource.includes('Parser issue'))
  assert.ok(newRunSource.includes('parseError.howToFix'))
})

test('new run page renders source-row admission guidance', () => {
  assert.ok(newRunSource.includes('sourceRowAdmission?.admissionDecision'))
  assert.ok(newRunSource.includes('function SourceRowAdmissionRejectNotice'))
  assert.ok(newRunSource.includes('Upload guidance'))
  assert.ok(newRunSource.includes('decision.howToFix'))
  assert.ok(
    newRunSource.includes(
      'state.diagnostics.sourceRowAdmission?.admissionDecision?.status',
    ),
  )
})
