import { describe, expect, it } from 'vitest'
import {
  buildCurrentRedirectTarget,
  buildLoginRedirectPath,
  buildSignupRedirectPath,
  inferRedirectProduct,
  normalizeRedirectTarget,
} from './redirects'

describe('auth redirects', () => {
  it('preserves protected paths with query and hash', () => {
    expect(
      buildCurrentRedirectTarget({
        pathname: '/vendors/Zendesk',
        search: '?tab=history',
        hash: '#signals',
      }),
    ).toBe('/vendors/Zendesk?tab=history#signals')
  })

  it('falls back to watchlists for unsafe or public targets', () => {
    expect(normalizeRedirectTarget('')).toBe('/watchlists')
    expect(normalizeRedirectTarget('/')).toBe('/watchlists')
    expect(normalizeRedirectTarget('https://example.com')).toBe('/watchlists')
    expect(normalizeRedirectTarget('//example.com/path')).toBe('/watchlists')
    expect(normalizeRedirectTarget('/login?redirect_to=%2Freports')).toBe('/watchlists')
    expect(normalizeRedirectTarget('/landing')).toBe('/watchlists')
  })

  it('builds a login path with encoded redirect_to', () => {
    expect(buildLoginRedirectPath('/reports?vendor=Zendesk')).toBe(
      '/login?redirect_to=%2Freports%3Fvendor%3DZendesk&product=b2b_retention',
    )
    expect(buildSignupRedirectPath('/reports?vendor=Zendesk')).toBe(
      '/signup?redirect_to=%2Freports%3Fvendor%3DZendesk&product=b2b_retention',
    )
  })

  it('infers challenger product for challenger workflows', () => {
    expect(inferRedirectProduct('/challengers?search=Zendesk')).toBe('b2b_challenger')
    expect(buildLoginRedirectPath('/vendor-targets?search=Zendesk')).toBe(
      '/login?redirect_to=%2Fvendor-targets%3Fsearch%3DZendesk&product=b2b_challenger',
    )
  })
})
