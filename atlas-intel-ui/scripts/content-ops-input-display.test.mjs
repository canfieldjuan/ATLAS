import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import ts from 'typescript'

const source = readFileSync(
  new URL('../src/domain/contentOps/inputDisplay.ts', import.meta.url),
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
const { inputContractDisplay } = await import(moduleUrl)

test('inputContractDisplay prefers catalog label and placeholder', () => {
  assert.deepEqual(
    inputContractDisplay(
      {
        label: 'Catalog documentation terms',
        placeholder: 'Catalog docs term one\nCatalog docs term two',
      },
      {
        label: 'Fallback documentation terms',
        placeholder: 'Fallback docs term',
      },
    ),
    {
      label: 'Catalog documentation terms',
      placeholder: 'Catalog docs term one\nCatalog docs term two',
    },
  )
})

test('inputContractDisplay falls back when catalog placeholder is absent', () => {
  assert.deepEqual(
    inputContractDisplay(
      {
        label: 'Catalog vocabulary rules',
      },
      {
        label: 'Fallback vocabulary rules',
        placeholder: 'fallback term, canonical term',
      },
    ),
    {
      label: 'Catalog vocabulary rules',
      placeholder: 'fallback term, canonical term',
    },
  )
})
