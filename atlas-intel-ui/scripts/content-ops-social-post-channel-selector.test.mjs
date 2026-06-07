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
  fromWireRequest,
  toWireRequest,
} = await loadTsModule('../src/domain/contentOps/fromWire.ts')

const {
  DEFAULT_SOCIAL_POST_CHANNELS,
  SOCIAL_POST_CHANNEL_OPTIONS,
  SOCIAL_POST_OUTPUT,
  normalizeSocialPostChannels,
  requestWithSocialPostChannels,
} = await loadTsModule('../src/domain/contentOps/socialPostChannels.ts')

const newRunSource = readFileSync(
  new URL('../src/pages/ContentOpsNewRun.tsx', import.meta.url),
  'utf8',
)
const helperSource = readFileSync(
  new URL('../src/domain/contentOps/socialPostChannels.ts', import.meta.url),
  'utf8',
)

test('social post channel ids match the backend contract order', () => {
  assert.equal(SOCIAL_POST_OUTPUT, 'social_post')
  assert.deepEqual(
    SOCIAL_POST_CHANNEL_OPTIONS.map((option) => option.id),
    ['linkedin', 'x', 'facebook', 'instagram', 'threads'],
  )
  assert.deepEqual(DEFAULT_SOCIAL_POST_CHANNELS, ['linkedin'])
})

test('normalizes selected social post channels with LinkedIn default fallback', () => {
  assert.deepEqual(normalizeSocialPostChannels([]), ['linkedin'])
  assert.deepEqual(normalizeSocialPostChannels(null), ['linkedin'])
  assert.deepEqual(
    normalizeSocialPostChannels(['linkedin', 'x', 'linkedin', 'invalid']),
    ['linkedin', 'x'],
  )
})

test('social post output submits selected channels through inputs.social_channels', () => {
  const request = fromWireRequest({
    outputs: ['social_post'],
    inputs: {
      topic: 'retention',
      channels: ['email'],
      social_post_channels: ['instagram'],
    },
  })

  const wire = toWireRequest(
    requestWithSocialPostChannels(request, ['linkedin', 'x', 'facebook']),
  )

  assert.deepEqual(wire.inputs.social_channels, [
    'linkedin',
    'x',
    'facebook',
  ])
  assert.deepEqual(wire.inputs.channels, ['email'])
  assert.equal(wire.inputs.social_post_channels, undefined)
})

test('non-social outputs omit social channel inputs', () => {
  const request = fromWireRequest({
    outputs: ['blog_post'],
    inputs: {
      social_channels: ['x'],
      social_post_channels: ['threads'],
    },
  })

  const wire = toWireRequest(requestWithSocialPostChannels(request, ['x']))

  assert.equal(wire.inputs.social_channels, undefined)
  assert.equal(wire.inputs.social_post_channels, undefined)
})

test('New Run UI gates the selector and uses the same submit request mapper', () => {
  assert.match(newRunSource, /socialPostOutputSelected &&/)
  assert.match(newRunSource, /SOCIAL_POST_CHANNEL_OPTIONS\.map/)
  assert.match(newRunSource, /handleSocialPostChannelChange/)
  assert.match(newRunSource, /requestWithSocialPostChannels\(/)
  assert.match(newRunSource, /toWireRequest\(parsed\.domainRequest\)/)
  assert.match(helperSource, /social_channels/)
  assert.doesNotMatch(helperSource, /inputs\.channels/)
})
