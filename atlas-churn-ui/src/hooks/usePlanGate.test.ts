import { renderHook } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { usePlanGate } from './usePlanGate'

const useAuth = vi.hoisted(() => vi.fn())

vi.mock('../auth/AuthContext', () => ({
  useAuth,
}))

describe('usePlanGate', () => {
  beforeEach(() => {
    useAuth.mockReset()
  })

  it('maps growth plans to the expected access flags and vendor limits', () => {
    useAuth.mockReturnValue({
      user: {
        plan: 'b2b_growth',
      },
    })

    const { result } = renderHook(() => usePlanGate())

    expect(result.current.plan).toBe('b2b_growth')
    expect(result.current.isTrial).toBe(false)
    expect(result.current.vendorLimit).toBe(25)
    expect(result.current.canAccessCampaigns).toBe(true)
    expect(result.current.canAccessReports).toBe(true)
    expect(result.current.canAccessApi).toBe(false)
  })

  it('falls back to trial limits for missing or unknown plans', () => {
    useAuth.mockReturnValue({
      user: {
        plan: 'custom_enterprise',
      },
    })

    const { result } = renderHook(() => usePlanGate())

    expect(result.current.plan).toBe('custom_enterprise')
    expect(result.current.vendorLimit).toBe(1)
    expect(result.current.canAccessCampaigns).toBe(false)
    expect(result.current.canAccessReports).toBe(false)
    expect(result.current.canAccessApi).toBe(false)
    expect(result.current.isTrial).toBe(false)
  })
})
