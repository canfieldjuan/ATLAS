import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

const appSource = readFileSync(new URL('../src/App.tsx', import.meta.url), 'utf8')
const sidebarSource = readFileSync(
  new URL('../src/components/Sidebar.tsx', import.meta.url),
  'utf8',
)
const pageSource = readFileSync(
  new URL('../src/pages/ContentOpsBrandVoiceSettings.tsx', import.meta.url),
  'utf8',
)
const managerSource = readFileSync(
  new URL('../src/components/contentOps/BrandVoiceProfileManager.tsx', import.meta.url),
  'utf8',
)

test('brand voice settings page is registered as protected Content Ops route', () => {
  assert.ok(
    appSource.includes(
      "const ContentOpsBrandVoiceSettings = lazy(",
    ),
  )
  assert.ok(
    appSource.includes(
      '<Route path="/content-ops/brand-voice" element={renderLazyRoute(ContentOpsBrandVoiceSettings)} />',
    ),
  )
})

test('B2B sidebar exposes brand voice settings navigation', () => {
  const matches = sidebarSource.match(/to: '\/content-ops\/brand-voice'/g) ?? []
  assert.equal(matches.length, 2)
  assert.ok(sidebarSource.includes("label: 'Brand Voice'"))
  assert.ok(sidebarSource.includes('icon: Palette'))
})

test('settings page lists tenant brand voice profiles with real API and states', () => {
  assert.ok(pageSource.includes('fetchContentOpsBrandVoiceProfiles()'))
  assert.ok(pageSource.includes('useApiData(() => fetchContentOpsBrandVoiceProfiles(), [])'))
  assert.ok(pageSource.includes('if ((loading || refreshing) && !profiles)'))
  assert.ok(pageSource.includes('Loading brand voice profiles...'))
  assert.ok(pageSource.includes('BrandVoiceProfileManager'))
  assert.ok(pageSource.includes('selectedProfileId'))
  assert.ok(pageSource.includes('onChange={setSelectedProfileId}'))
  assert.ok(pageSource.includes('No saved brand voice profiles'))
  assert.ok(pageSource.includes('Refresh failed:'))
  assert.ok(pageSource.includes('profile.descriptors'))
  assert.ok(pageSource.includes('profile.exemplars[0]'))
})

test('settings page uses shared manager instead of routing management to New Run', () => {
  const newRunLinks = pageSource.match(/to="\/content-ops\/new"/g) ?? []
  assert.equal(newRunLinks.length, 0)
  assert.ok(pageSource.includes("selected ? 'Selected' : 'Manage'"))
  assert.ok(managerSource.includes('New brand voice'))
  assert.ok(managerSource.includes('Edit brand voice'))
  assert.ok(managerSource.includes('Archive'))
  assert.ok(managerSource.includes('fetchContentOpsBrandVoiceSampleUrl'))
  assert.equal(pageSource.includes('Use in run'), false)
})
