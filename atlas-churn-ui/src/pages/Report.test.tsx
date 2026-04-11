import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import Report from './Report'

const auth = vi.hoisted(() => ({
  useAuth: vi.fn(),
}))
const router = vi.hoisted(() => ({
  navigate: vi.fn(),
}))

vi.mock('../auth/AuthContext', () => auth)
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom')
  return {
    ...actual,
    useNavigate: () => router.navigate,
  }
})
vi.mock('../components/SeoHead', () => ({
  default: function SeoHead() {
    return null
  },
}))
vi.mock('../components/report-renderers/StructuredReportData', () => ({
  StructuredReportData: function StructuredReportData() {
    return <div>Structured Report</div>
  },
}))
vi.mock('../components/report-renderers/SpecializedReportData', () => ({
  SpecializedReportData: function SpecializedReportData() {
    return <div>Specialized Report</div>
  },
}))

describe('Report', () => {
  const signup = vi.fn()
  const login = vi.fn()
  const originalFetch = global.fetch

  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
    signup.mockResolvedValue(undefined)
    login.mockResolvedValue(undefined)
    router.navigate.mockReset()
    auth.useAuth.mockReturnValue({
      user: null,
      signup,
      login,
    })
  })

  afterEach(() => {
    global.fetch = originalFetch
  })

  it('exposes product workflow links on the gated report page', () => {
    render(
      <MemoryRouter initialEntries={['/report?vendor=Zendesk&ref=test-token']}>
        <Routes>
          <Route path="/report" element={<Report />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(screen.getByRole('link', { name: 'Sign in to Watchlists' })).toHaveAttribute(
      'href',
      '/login?redirect_to=%2Fwatchlists&product=b2b_retention',
    )
    expect(screen.getByRole('link', { name: 'Sign in to Challengers' })).toHaveAttribute(
      'href',
      '/login?redirect_to=%2Fchallengers&product=b2b_challenger',
    )
    expect(screen.getByRole('link', { name: 'Start Vendor Retention' })).toHaveAttribute(
      'href',
      '/signup?product=b2b_retention&redirect_to=%2Fwatchlists',
    )
    expect(screen.getByRole('link', { name: 'Start Challenger Lead Gen' })).toHaveAttribute(
      'href',
      '/signup?product=b2b_challenger&redirect_to=%2Fchallengers',
    )
  })

  it('redirects checkout-created accounts into watchlists', async () => {
    const user = userEvent.setup()
    const timeoutSpy = vi.spyOn(global, 'setTimeout')
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ email: 'juan@example.com', tier: 'pro' }),
    } as Response)

    render(
      <MemoryRouter initialEntries={['/report?vendor=Zendesk&checkout=success&session_id=session-123']}>
        <Routes>
          <Route path="/report" element={<Report />} />
        </Routes>
      </MemoryRouter>,
    )

    await screen.findByDisplayValue('juan@example.com')
    await user.type(screen.getByPlaceholderText('Jane Smith'), 'Juan Canfield')
    await user.type(screen.getByPlaceholderText('Acme Inc.'), 'Atlas Labs')
    await user.type(screen.getByPlaceholderText('Min. 8 characters'), 'password123')
    await user.click(screen.getByRole('button', { name: 'Create Account' }))

    await waitFor(() => {
      expect(signup).toHaveBeenCalledWith(
        'juan@example.com',
        'password123',
        'Juan Canfield',
        'Atlas Labs',
        'b2b_retention',
      )
    })

    const redirectCall = timeoutSpy.mock.calls.find(([, delay]) => delay === 2000)
    const redirectTimer = redirectCall?.[0]
    expect(typeof redirectTimer).toBe('function')
    if (typeof redirectTimer === 'function') {
      redirectTimer()
    }
    expect(router.navigate).toHaveBeenCalledWith('/watchlists')
    timeoutSpy.mockRestore()
  })
})
