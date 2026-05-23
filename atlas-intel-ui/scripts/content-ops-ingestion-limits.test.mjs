import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import ts from 'typescript'

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
const { fromWireCatalog } = await import(moduleUrl)

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

  assert.deepEqual(fromWireCatalog(catalogFixture).ingestionLimits.fileUpload.supportedFormats, [
    'auto',
    'json',
    'jsonl',
    'csv',
  ])
})
