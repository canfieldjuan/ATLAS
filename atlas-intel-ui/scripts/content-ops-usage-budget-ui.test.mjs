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

test('account usage budget request fields round-trip through domain mapping', () => {
  const domain = fromWireRequest({
    target_mode: 'vendor_retention',
    outputs: ['landing_page'],
    limit: 3,
    max_cost_usd: 1.25,
    account_usage_budget_usd: 5,
    account_usage_budget_days: 14,
    inputs: { target_keyword: 'support tickets' },
  })

  assert.equal(domain.accountUsageBudgetUsd, 5)
  assert.equal(domain.accountUsageBudgetDays, 14)

  const wire = toWireRequest(domain)
  assert.equal(wire.account_usage_budget_usd, 5)
  assert.equal(wire.account_usage_budget_days, 14)
  assert.deepEqual(wire.inputs, { target_keyword: 'support tickets' })
})

test('account usage budget defaults to no cap and seven-day window', () => {
  const domain = fromWireRequest({})

  assert.equal(domain.accountUsageBudgetUsd, null)
  assert.equal(domain.accountUsageBudgetDays, 7)
})

test('preview mapper preserves account usage budget verdict', () => {
  const preview = fromWirePreview({
    can_run: false,
    outputs: ['landing_page'],
    estimated_cost_usd: 0.75,
    missing_inputs: [],
    blocked_outputs: [],
    warnings: ['Projected account usage exceeds account_usage_budget_usd'],
    normalized_request: {
      outputs: ['landing_page'],
      limit: 1,
      max_cost_usd: null,
      account_usage_budget_usd: 1,
      account_usage_budget_days: 7,
    },
    usage_budget: {
      budget_usd: 1,
      period_days: 7,
      current_cost_usd: 0.5,
      estimated_cost_usd: 0.75,
      projected_cost_usd: 1.25,
      exceeded: true,
    },
  })

  assert.equal(preview.usageBudget?.budgetUsd, 1)
  assert.equal(preview.usageBudget?.periodDays, 7)
  assert.equal(preview.usageBudget?.currentCostUsd, 0.5)
  assert.equal(preview.usageBudget?.estimatedCostUsd, 0.75)
  assert.equal(preview.usageBudget?.projectedCostUsd, 1.25)
  assert.equal(preview.usageBudget?.exceeded, true)
  assert.equal(preview.normalizedRequest?.accountUsageBudgetUsd, 1)
  assert.equal(preview.normalizedRequest?.accountUsageBudgetDays, 7)
})
