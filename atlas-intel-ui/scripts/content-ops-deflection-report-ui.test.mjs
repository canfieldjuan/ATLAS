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
  FAQ_DEFLECTION_REPORT_OUTPUT,
  faqDeflectionReportAnswerSteps,
  faqDeflectionReportView,
} = await loadTsModule('../src/domain/contentOps/faqDeflectionReport.ts')

const example = JSON.parse(
  readFileSync(
    new URL('../../docs/frontend/content_ops_faq_deflection_report_example.json', import.meta.url),
    'utf8',
  ),
)

const newRunSource = readFileSync(
  new URL('../src/pages/ContentOpsNewRun.tsx', import.meta.url),
  'utf8',
)

test('deflection report view splits proven and no-proven answers from contract example', () => {
  const report = faqDeflectionReportView(example)

  assert.ok(report)
  assert.equal(report.summary.generated, 2)
  assert.equal(report.provenItems.length, 1)
  assert.equal(report.noProvenItems.length, 1)
  assert.equal(report.provenItems[0].question, 'How do I export attribution reports?')
  assert.equal(report.noProvenItems[0].question, 'How do I enable SSO for my team?')
  assert.equal(report.provenItems[0].answerEvidenceStatus, 'resolution_evidence')
  assert.equal(report.noProvenItems[0].answerEvidenceStatus, 'draft_needs_review')
  assert.ok(
    !report.provenItems.some((item) => item.question.includes('SSO')),
    'no-proven SSO gap must not render as a proven drafted answer',
  )
})

test('deflection report UI suppresses unproven answer steps at render boundary', () => {
  const report = faqDeflectionReportView(example)
  assert.ok(report)
  const [unproven] = report.noProvenItems
  const [proven] = report.provenItems

  assert.ok(unproven.steps.length > 0, 'fixture carries raw review-placeholder steps')
  assert.deepEqual(faqDeflectionReportAnswerSteps(unproven, 'unproven'), [])
  assert.deepEqual(faqDeflectionReportAnswerSteps(unproven, 'proven'), [])
  assert.deepEqual(
    faqDeflectionReportAnswerSteps(proven, 'proven'),
    proven.steps,
  )
})

test('new run page renders faq deflection report execute results distinctly', () => {
  assert.equal(FAQ_DEFLECTION_REPORT_OUTPUT, 'faq_deflection_report')
  assert.ok(newRunSource.includes('FAQDeflectionReportExecutionSummary'))
  assert.ok(newRunSource.includes('output === FAQ_DEFLECTION_REPORT_OUTPUT'))
  assert.ok(newRunSource.includes('faqDeflectionReportAnswerSteps(item, tone)'))
  assert.ok(newRunSource.includes('Drafted Answers (proven solutions)'))
  assert.ok(newRunSource.includes('No Proven Answer Yet'))
  assert.ok(newRunSource.includes('FAQ_RESOLUTION_EVIDENCE_STATUS'))
})
