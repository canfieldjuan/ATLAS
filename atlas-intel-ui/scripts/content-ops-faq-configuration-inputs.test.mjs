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
  FAQ_DEFLECTION_REPORT_CONFIGURATION_OUTPUT,
  FAQ_MARKDOWN_OUTPUT,
  FAQ_REPRESENTATIVE_TAXONOMY_TERMS_INPUT,
  faqConfigurationControlsVisible,
  faqConfigurationInputsSelected,
  faqIntentRulesDraftValue,
  faqIntentRulesFromDraft,
} = await loadTsModule('../src/domain/contentOps/faqConfigurationInputs.ts')
const { FAQ_DEFLECTION_REPORT_OUTPUT } = await loadTsModule(
  '../src/domain/contentOps/faqDeflectionReport.ts',
)

const newRunSource = readFileSync(
  new URL('../src/pages/ContentOpsNewRun.tsx', import.meta.url),
  'utf8',
)

test('FAQ configuration controls apply to FAQ markdown and deflection report outputs', () => {
  assert.equal(FAQ_MARKDOWN_OUTPUT, 'faq_markdown')
  assert.equal(
    FAQ_DEFLECTION_REPORT_CONFIGURATION_OUTPUT,
    FAQ_DEFLECTION_REPORT_OUTPUT,
  )
  assert.equal(faqConfigurationInputsSelected([FAQ_MARKDOWN_OUTPUT]), true)
  assert.equal(
    faqConfigurationInputsSelected([FAQ_DEFLECTION_REPORT_OUTPUT]),
    true,
  )
  assert.equal(faqConfigurationInputsSelected(['landing_page']), false)
  assert.equal(faqConfigurationInputsSelected([]), false)
})

test('new run page uses the shared FAQ configuration output predicate', () => {
  assert.ok(
    newRunSource.includes(
      'const faqConfigurationOutputSelected = faqConfigurationInputsSelected(',
    ),
  )
  assert.ok(
    newRunSource.includes(
      'const faqConfigurationControlsVisibleForCatalog =',
    ),
  )
  assert.ok(newRunSource.includes('{faqConfigurationControlsVisibleForCatalog && ('))
  assert.ok(newRunSource.includes('{faqIntentRulesContract && ('))
  assert.ok(newRunSource.includes('{faqDocumentationTermsContract && ('))
  assert.ok(
    newRunSource.includes('{faqRepresentativeTaxonomyTermsContract && ('),
  )
  assert.ok(newRunSource.includes('{faqVocabularyGapRulesContract && ('))
  assert.ok(!newRunSource.includes('faqMarkdownOutputSelected'))
  assert.ok(newRunSource.includes('FAQ vocabulary-gap inputs'))
  assert.ok(newRunSource.includes('FAQ_INTENT_RULES_INPUT'))
  assert.ok(newRunSource.includes('FAQ_REPRESENTATIVE_TAXONOMY_TERMS_INPUT'))
  assert.ok(
    newRunSource.includes('handleFaqRepresentativeTaxonomyTermsChange'),
  )
  assert.ok(newRunSource.includes('handleFaqIntentRulesChange'))
})

test('FAQ configuration controls require selected output and advertised catalog contract', () => {
  assert.equal(
    faqConfigurationControlsVisible([FAQ_DEFLECTION_REPORT_OUTPUT], {
      intentRules: { key: 'faq_intent_rules' },
    }),
    true,
  )
  assert.equal(
    faqConfigurationControlsVisible([FAQ_DEFLECTION_REPORT_OUTPUT], {
      representativeTaxonomyTerms: {
        key: FAQ_REPRESENTATIVE_TAXONOMY_TERMS_INPUT,
      },
    }),
    true,
  )
  assert.equal(
    faqConfigurationControlsVisible([FAQ_DEFLECTION_REPORT_OUTPUT], {}),
    false,
  )
  assert.equal(
    faqConfigurationControlsVisible(['landing_page'], {
      intentRules: { key: 'faq_intent_rules' },
      documentationTerms: { key: 'faq_documentation_terms' },
      representativeTaxonomyTerms: {
        key: FAQ_REPRESENTATIVE_TAXONOMY_TERMS_INPUT,
      },
      vocabularyGapRules: { key: 'faq_vocabulary_gap_rules' },
    }),
    false,
  )
})

test('FAQ intent-rule draft helpers preserve line-based rules', () => {
  assert.deepEqual(
    faqIntentRulesFromDraft(
      'data freshness=warehouse sync,connector lag\n\nDATA FRESHNESS=warehouse sync,connector lag\naccess setup=invite link',
    ),
    [
      'data freshness=warehouse sync,connector lag',
      'access setup=invite link',
    ],
  )
  assert.equal(
    faqIntentRulesDraftValue([
      {
        topic: 'data freshness',
        keywords: ['warehouse sync', 'connector lag'],
      },
      'access setup=invite link',
    ]),
    'data freshness=warehouse sync,connector lag\naccess setup=invite link',
  )
})
