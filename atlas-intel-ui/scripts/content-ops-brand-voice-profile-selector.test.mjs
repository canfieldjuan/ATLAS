import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import ts from 'typescript'

const API_ORIGIN = 'https://api.example.test'

async function loadTsModule(path, replacements = []) {
  let source = readFileSync(new URL(path, import.meta.url), 'utf8')
  for (const [needle, replacement] of replacements) {
    source = source.replace(needle, replacement)
  }
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
  createContentOpsBrandVoiceProfile,
  deleteContentOpsBrandVoiceProfile,
  fetchContentOpsBrandVoiceSampleUrl,
  fetchContentOpsBrandVoiceProfiles,
  updateContentOpsBrandVoiceProfile,
} = await loadTsModule('../src/api/contentOps.ts', [
  [
    "import { tryRefreshToken } from '../auth/AuthContext'\n",
    'const tryRefreshToken = async () => null\n',
  ],
  [
    "import { API_BASE } from './config'\n",
    `const API_BASE = '${API_ORIGIN}'\n`,
  ],
])

const {
  fromWireRequest,
  toWireRequest,
} = await loadTsModule('../src/domain/contentOps/fromWire.ts')

const {
  applyBrandVoiceProfileEditorPatch,
  blankBrandVoiceProfileEditorState,
  brandVoiceProfileEditorRequest,
  canSaveBrandVoiceProfileEditor,
  deriveBrandVoiceProfileEditorPatch,
} = await loadTsModule('../src/domain/contentOps/brandVoiceProfileEditor.ts')

const newRunSource = readFileSync(
  new URL('../src/pages/ContentOpsNewRun.tsx', import.meta.url),
  'utf8',
)

function installBrowserStubs() {
  Object.defineProperty(globalThis, 'localStorage', {
    configurable: true,
    value: {
      getItem(key) {
        return key === 'atlas_token' ? 'test-token' : null
      },
      removeItem() {},
    },
  })
  Object.defineProperty(globalThis, 'window', {
    configurable: true,
    value: { location: { href: '' } },
  })
}

function installFetchResponder(payload, status = 200) {
  const calls = []
  globalThis.fetch = async (url, init = {}) => {
    calls.push({ url: String(url), init })
    const body = status === 204 ? null : JSON.stringify(payload)
    return new Response(body, {
      status,
      headers: { 'content-type': 'application/json' },
    })
  }
  return calls
}

function profilePayload(overrides = {}) {
  return {
    id: 'profile-1',
    account_id: 'account-1',
    name: 'Acme editorial',
    descriptors: ['plainspoken'],
    exemplars: ['Write like this.'],
    banned_terms: ['synergy'],
    preferred_pov: 'second_person',
    reading_level: 'plain',
    metadata: { source: 'operator' },
    created_at: '2026-06-04T00:00:00+00:00',
    updated_at: '2026-06-04T00:00:00+00:00',
    archived_at: null,
    ...overrides,
  }
}

test.beforeEach(() => {
  installBrowserStubs()
})

test('brand voice profile list calls tenant route', async () => {
  const payload = [profilePayload()]
  const calls = installFetchResponder(payload)

  const result = await fetchContentOpsBrandVoiceProfiles()

  assert.deepEqual(result, payload)
  assert.equal(calls.length, 1)
  const [{ url, init }] = calls
  assert.equal(
    url,
    `${API_ORIGIN}/api/v1/content-ops/brand-voice-profiles`,
  )
  assert.equal(init.method, undefined)
  assert.deepEqual(init.headers, { Authorization: 'Bearer test-token' })
})

test('brand voice profile create posts bounded profile payload', async () => {
  const calls = installFetchResponder(profilePayload(), 201)
  const body = {
    name: 'Acme editorial',
    descriptors: ['plainspoken'],
    exemplars: ['Write like this.'],
    banned_terms: ['synergy'],
    preferred_pov: 'second_person',
    reading_level: 'plain',
    metadata: { source: 'operator' },
  }

  await createContentOpsBrandVoiceProfile(body)

  assert.equal(calls.length, 1)
  const [{ url, init }] = calls
  assert.equal(
    url,
    `${API_ORIGIN}/api/v1/content-ops/brand-voice-profiles`,
  )
  assert.equal(init.method, 'POST')
  assert.deepEqual(init.headers, {
    'Content-Type': 'application/json',
    Authorization: 'Bearer test-token',
  })
  assert.deepEqual(JSON.parse(init.body), body)
})

test('brand voice profile update request shape is encoded and authenticated', async () => {
  const calls = installFetchResponder(profilePayload())
  const body = {
    name: 'Acme editorial',
    descriptors: ['plainspoken'],
  }

  await updateContentOpsBrandVoiceProfile('profile/id', body)

  assert.equal(calls.length, 1)
  const [{ url, init }] = calls
  assert.equal(
    url,
    `${API_ORIGIN}/api/v1/content-ops/brand-voice-profiles/profile%2Fid`,
  )
  assert.equal(init.method, 'PUT')
  assert.deepEqual(init.headers, {
    'Content-Type': 'application/json',
    Authorization: 'Bearer test-token',
  })
  assert.deepEqual(JSON.parse(init.body), body)
})

test('brand voice profile delete archives encoded profile id', async () => {
  const calls = installFetchResponder('', 204)

  await deleteContentOpsBrandVoiceProfile('profile/id')

  assert.equal(calls.length, 1)
  const [{ url, init }] = calls
  assert.equal(
    url,
    `${API_ORIGIN}/api/v1/content-ops/brand-voice-profiles/profile%2Fid`,
  )
  assert.equal(init.method, 'DELETE')
  assert.deepEqual(init.headers, { Authorization: 'Bearer test-token' })
})

test('brand voice sample URL fetch posts authenticated URL payload', async () => {
  const payload = {
    url: 'https://example.test/about',
    title: 'Acme About',
    text: 'Launch secure workflows faster.',
    source_character_count: 31,
  }
  const calls = installFetchResponder(payload)

  const result = await fetchContentOpsBrandVoiceSampleUrl({
    url: 'https://example.test/about',
  })

  assert.deepEqual(result, payload)
  assert.equal(calls.length, 1)
  const [{ url, init }] = calls
  assert.equal(
    url,
    `${API_ORIGIN}/api/v1/content-ops/brand-voice-profiles/sample-url`,
  )
  assert.equal(init.method, 'POST')
  assert.deepEqual(init.headers, {
    'Content-Type': 'application/json',
    Authorization: 'Bearer test-token',
  })
  assert.deepEqual(JSON.parse(init.body), {
    url: 'https://example.test/about',
  })
})

test('brand voice profile id round-trips through domain request mapping', () => {
  const domain = fromWireRequest({
    target_mode: 'vendor_retention',
    outputs: ['landing_page'],
    brand_voice_profile_id: 'profile-1',
    inputs: { target_keyword: 'support tickets' },
  })

  assert.equal(domain.brandVoiceProfileId, 'profile-1')

  const wire = toWireRequest(domain)
  assert.equal(wire.brand_voice_profile_id, 'profile-1')
  assert.deepEqual(wire.inputs, { target_keyword: 'support tickets' })
})

test('brand voice editor request trims and bounds textarea fields', () => {
  const editor = {
    ...blankBrandVoiceProfileEditorState(),
    name: '  Acme editorial  ',
    descriptorsText: Array.from({ length: 10 }, (_, index) => ` descriptor ${index} `).join('\n'),
    exemplarsText: ' First example. \n\n Second example. \n Third example. \n Fourth example. ',
    bannedTermsText: ' synergy \n\n leverage ',
    preferredPov: ' second_person ',
    readingLevel: ' plain ',
  }

  assert.equal(canSaveBrandVoiceProfileEditor(editor), true)
  const body = brandVoiceProfileEditorRequest(editor)

  assert.equal(body.name, 'Acme editorial')
  assert.deepEqual(body.descriptors, [
    'descriptor 0',
    'descriptor 1',
    'descriptor 2',
    'descriptor 3',
    'descriptor 4',
    'descriptor 5',
    'descriptor 6',
    'descriptor 7',
  ])
  assert.deepEqual(body.exemplars, [
    'First example.',
    'Second example.',
    'Third example.',
  ])
  assert.deepEqual(body.banned_terms, ['synergy', 'leverage'])
  assert.equal(body.preferred_pov, 'second_person')
  assert.equal(body.reading_level, 'plain')
  assert.equal(
    canSaveBrandVoiceProfileEditor({
      ...blankBrandVoiceProfileEditorState(),
      name: 'Acme editorial',
    }),
    false,
  )
})

test('brand voice sample import derives editable profile fields', () => {
  const patch = deriveBrandVoiceProfileEditorPatch(
    "You can launch secure workflows faster. Your team gets clean automation without waiting on a long integration project. We're here when your operators need help!",
    { fallbackName: 'acme-homepage.txt' },
  )

  assert.equal(patch.name, 'acme homepage')
  assert.equal(patch.preferredPov, 'second_person')
  assert.equal(patch.readingLevel, 'plain')
  assert.ok(patch.descriptorsText.includes('concise'))
  assert.ok(patch.descriptorsText.includes('customer-focused'))
  assert.ok(patch.descriptorsText.includes('conversational'))
  assert.ok(patch.descriptorsText.includes('technical'))
  assert.ok(patch.exemplarsText.includes('secure workflows'))
  assert.deepEqual(patch.metadata, {
    source: 'content_ops_ui_sample',
    sample_character_count: 160,
  })
})

test('brand voice sample patch preserves manually entered fields', () => {
  const editor = {
    ...blankBrandVoiceProfileEditorState(),
    name: 'Manual profile',
    descriptorsText: '',
    exemplarsText: 'Existing example.',
  }
  const patch = deriveBrandVoiceProfileEditorPatch(
    'Your support team can resolve data questions quickly.',
    { fallbackName: 'sample.txt' },
  )

  const merged = applyBrandVoiceProfileEditorPatch(editor, patch)

  assert.equal(merged.name, 'Manual profile')
  assert.ok(merged.descriptorsText.includes('customer-focused'))
  assert.equal(merged.exemplarsText, 'Existing example.')
  assert.equal(merged.preferredPov, 'second_person')
})

test('new run page renders brand voice profile selector wiring', () => {
  assert.ok(newRunSource.includes('function BrandVoiceProfileSelector'))
  assert.ok(newRunSource.includes('fetchContentOpsBrandVoiceProfiles'))
  assert.ok(newRunSource.includes('fetchContentOpsBrandVoiceSampleUrl'))
  assert.ok(newRunSource.includes('createContentOpsBrandVoiceProfile'))
  assert.ok(newRunSource.includes('updateContentOpsBrandVoiceProfile'))
  assert.ok(newRunSource.includes('deleteContentOpsBrandVoiceProfile'))
  assert.ok(newRunSource.includes('brandVoiceProfileId'))
  assert.ok(newRunSource.includes('Saved profile'))
  assert.ok(newRunSource.includes('No saved brand voice'))
  assert.ok(newRunSource.includes('New brand voice'))
  assert.ok(newRunSource.includes('Edit brand voice'))
  assert.ok(newRunSource.includes('Archive'))
  assert.ok(newRunSource.includes('Sample import'))
  assert.ok(newRunSource.includes('type="file"'))
  assert.ok(newRunSource.includes('Fetch URL'))
  assert.ok(newRunSource.includes('brandVoiceSampleFallbackName'))
  assert.ok(newRunSource.includes('deriveBrandVoiceProfileEditorPatch'))
  assert.ok(newRunSource.includes('disabled={loading || mutating}'))
  assert.ok(newRunSource.includes('selectedProfileIdRef.current === archiveProfileId'))
})
