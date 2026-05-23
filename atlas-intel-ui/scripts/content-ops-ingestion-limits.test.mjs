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

const { fromWireCatalog } = await loadTsModule(
  '../src/domain/contentOps/fromWire.ts',
)
const {
  contentOpsIngestionFilePreflightError,
  formatContentOpsBytes,
} = await loadTsModule('../src/domain/contentOps/ingestionLimits.ts')

const catalogFixture = JSON.parse(
  readFileSync(
    new URL('../src/api/__fixtures__/contentOps/catalog.json', import.meta.url),
    'utf8',
  ),
)

test('fromWireCatalog maps ingestion limits from wire to domain shape', () => {
  assert.deepEqual(fromWireCatalog(catalogFixture).ingestionLimits, {
    inlineRows: {
      maxRows: 1000,
      deprecated: true,
    },
    fileUpload: {
      maxFileBytes: 25 * 1024 * 1024,
      maxRows: 10000,
      supportedFormats: ['auto', 'json', 'jsonl', 'csv'],
    },
    maxSourceTextChars: 10000,
    maxSampleLimit: 25,
  })
})

test('fromWireCatalog copies ingestion limit supported formats', () => {
  const first = fromWireCatalog(catalogFixture)
  first.ingestionLimits.fileUpload.supportedFormats.push('yaml')

  assert.deepEqual(
    fromWireCatalog(catalogFixture).ingestionLimits.fileUpload.supportedFormats,
    ['auto', 'json', 'jsonl', 'csv'],
  )
})

test('contentOpsIngestionFilePreflightError rejects oversized uploads', () => {
  const limits = fromWireCatalog(catalogFixture).ingestionLimits

  assert.equal(
    contentOpsIngestionFilePreflightError(
      {
        name: 'tickets.csv',
        size: limits.fileUpload.maxFileBytes + 1,
      },
      limits,
    ),
    'tickets.csv is 25.0 MB. Upload files must be 25.0 MB or smaller.',
  )
})

test('contentOpsIngestionFilePreflightError accepts files at the byte cap', () => {
  const limits = fromWireCatalog(catalogFixture).ingestionLimits

  assert.equal(
    contentOpsIngestionFilePreflightError(
      {
        name: 'tickets.csv',
        size: limits.fileUpload.maxFileBytes,
      },
      limits,
    ),
    null,
  )
})

test('contentOpsIngestionFilePreflightError fails open when cap is not finite', () => {
  const limits = fromWireCatalog(catalogFixture).ingestionLimits

  assert.equal(
    contentOpsIngestionFilePreflightError(
      {
        name: 'tickets.csv',
        size: limits.fileUpload.maxFileBytes + 1,
      },
      {
        ...limits,
        fileUpload: {
          ...limits.fileUpload,
          maxFileBytes: Number.NaN,
        },
      },
    ),
    null,
  )
})

test('formatContentOpsBytes formats byte caps for operators', () => {
  assert.equal(formatContentOpsBytes(25 * 1024 * 1024), '25.0 MB')
  assert.equal(formatContentOpsBytes(512), '512 B')
})
