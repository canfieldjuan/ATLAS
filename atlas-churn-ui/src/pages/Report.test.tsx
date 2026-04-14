import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes, RouterProvider, createMemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import Report from './Report'

const auth = vi.hoisted(() => ({
  useAuth: vi.fn(),
}))
const router = vi.hoisted(() => ({
  navigate: vi.fn(),
}))
const specializedRenderer = vi.hoisted(() => ({
  lastProps: null as any,
}))
const structuredRenderer = vi.hoisted(() => ({
  lastProps: null as any,
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
  StructuredReportData: function StructuredReportData(props: any) {
    structuredRenderer.lastProps = props
    return <div>Structured Report</div>
  },
}))
vi.mock('../components/report-renderers/SpecializedReportData', () => ({
  SpecializedReportData: function SpecializedReportData(props: any) {
    specializedRenderer.lastProps = props
    return <div>Specialized Report</div>
  },
}))

function buildStructuredPublicReport() {
  return {
    vendor_name: 'Zendesk',
    briefing: {},
    intelligence_reports: [
      {
        report_type: 'exploratory_overview',
        executive_summary: 'Structured summary',
        data: {
          data_as_of_date: '2026-04-08',
          window_days: 45,
          key_insights: [
            { label: 'Pricing friction', summary: 'Pricing created churn risk' },
          ],
          key_insights_reference_ids: {
            witness_ids: ['w1'],
          },
        },
        report_date: '2026-04-10',
      },
    ],
    product_profile: null,
  }
}

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
    specializedRenderer.lastProps = null
    structuredRenderer.lastProps = null
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


  it('passes the vendor context into structured report renderers on direct report load', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => buildStructuredPublicReport(),
    } as Response)

    render(
      <MemoryRouter initialEntries={['/report?vendor=Zendesk&ref=test-token&mode=view']}>
        <Routes>
          <Route path="/report" element={<Report />} />
        </Routes>
      </MemoryRouter>,
    )

    await screen.findByText('Structured Report')
    await waitFor(() => {
      expect(structuredRenderer.lastProps?.vendorName).toBe('Zendesk')
      expect(structuredRenderer.lastProps?.backTo).toBe('/report?vendor=Zendesk&ref=test-token&mode=view')
      expect(structuredRenderer.lastProps?.asOfDate).toBe('2026-04-08')
      expect(structuredRenderer.lastProps?.windowDays).toBe(45)
    })
  })

  it('rehydrates the direct report view on same-route vendor changes and ignores stale loads', async () => {
    let resolveFirst: ((value: Response) => void) | undefined
    let resolveSecond: ((value: Response) => void) | undefined

    global.fetch = vi.fn()
      .mockImplementationOnce(() => new Promise((resolve) => {
        resolveFirst = resolve as (value: Response) => void
      }))
      .mockImplementationOnce(() => new Promise((resolve) => {
        resolveSecond = resolve as (value: Response) => void
      }))

    const appRouter = createMemoryRouter(
      [{ path: '/report', element: <Report /> }],
      { initialEntries: ['/report?vendor=Zendesk&ref=token-1&mode=view'] },
    )

    render(<RouterProvider router={appRouter} />)

    expect(screen.getByText('Loading Zendesk intelligence report...')).toBeInTheDocument()

    await act(async () => {
      await appRouter.navigate('/report?vendor=Intercom&ref=token-2&mode=view')
    })

    expect(screen.getByText('Loading Intercom intelligence report...')).toBeInTheDocument()

    const resolveLatestReport = resolveSecond
    if (!resolveLatestReport) {
      throw new Error('Expected latest report fetch resolver')
    }

    await act(async () => {
      resolveLatestReport({
        ok: true,
        json: async () => ({
          ...buildStructuredPublicReport(),
          vendor_name: 'Intercom',
        }),
      } as Response)
      await Promise.resolve()
      await Promise.resolve()
    })

    await waitFor(() => {
      expect(structuredRenderer.lastProps?.vendorName).toBe('Intercom')
    })
    expect(structuredRenderer.lastProps?.backTo).toBe('/report?vendor=Intercom&ref=token-2&mode=view')
    expect(screen.getByRole('heading', { name: 'Intercom' })).toBeInTheDocument()

    const resolveStaleReport = resolveFirst
    if (!resolveStaleReport) {
      throw new Error('Expected stale report fetch resolver')
    }

    await act(async () => {
      resolveStaleReport({
        ok: true,
        json: async () => buildStructuredPublicReport(),
      } as Response)
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(structuredRenderer.lastProps?.vendorName).toBe('Intercom')
    expect(screen.getByRole('heading', { name: 'Intercom' })).toBeInTheDocument()
  })

  it('passes the vendor context into specialized report renderers on direct report load', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        vendor_name: 'Zendesk',
        briefing: {},
        intelligence_reports: [
          {
            report_type: 'accounts_in_motion',
            executive_summary: 'Specialized summary',
            data: {
              data_as_of_date: '2026-04-08',
              window_days: 45,
              reference_ids: {
                witness_ids: ['witness-1'],
              },
              accounts: [
                {
                  company: 'Acme Corp',
                  opportunity_score: 72,
                },
              ],
            },
            report_date: '2026-04-10',
          },
        ],
        product_profile: null,
      }),
    } as Response)

    render(
      <MemoryRouter initialEntries={['/report?vendor=Zendesk&ref=test-token&mode=view']}>
        <Routes>
          <Route path="/report" element={<Report />} />
        </Routes>
      </MemoryRouter>,
    )

    await screen.findByText('Specialized Report')
    await waitFor(() => {
      expect(specializedRenderer.lastProps?.vendorName).toBe('Zendesk')
      expect(specializedRenderer.lastProps?.backTo).toBe('/report?vendor=Zendesk&ref=test-token&mode=view')
      expect(specializedRenderer.lastProps?.asOfDate).toBe('2026-04-08')
      expect(specializedRenderer.lastProps?.windowDays).toBe(45)
    })
    expect(specializedRenderer.lastProps?.reportType).toBe('accounts_in_motion')
  })

  it('shows account-pressure disclaimer and confidence tier on the public report page', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        vendor_name: 'Zendesk',
        briefing: {
          churn_pressure_score: 72,
          category: 'Helpdesk',
          named_account_count: 3,
          account_pressure_disclaimer: 'Early account signal only.',
          account_actionability_tier: 'low',
        },
        intelligence_reports: [],
        product_profile: null,
      }),
    } as Response)

    render(
      <MemoryRouter initialEntries={['/report?vendor=Zendesk&ref=test-token&mode=view']}>
        <Routes>
          <Route path="/report" element={<Report />} />
        </Routes>
      </MemoryRouter>,
    )

    await screen.findByRole('heading', { name: 'Zendesk' })
    expect(screen.getByText('Accounts Showing Friction')).toBeInTheDocument()
    expect(screen.getByText('Early account signal only.')).toBeInTheDocument()
    expect(screen.getByText('Confidence tier: low')).toBeInTheDocument()
  })

  it('shows checkout API errors inline in the pricing modal', async () => {
    const user = userEvent.setup()
    const alertSpy = vi.spyOn(window, 'alert').mockImplementation(() => {})
    global.fetch = vi.fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => buildStructuredPublicReport(),
      } as Response)
      .mockResolvedValueOnce({
        ok: false,
        status: 503,
        json: async () => ({ detail: 'Checkout temporarily unavailable' }),
      } as Response)

    render(
      <MemoryRouter initialEntries={['/report?vendor=Zendesk&ref=test-token&mode=view']}>
        <Routes>
          <Route path="/report" element={<Report />} />
        </Routes>
      </MemoryRouter>,
    )

    await screen.findByText('Structured Report')
    await user.click(screen.getAllByRole('button', { name: 'Get Weekly Intelligence' })[0])
    await user.click(screen.getAllByRole('button', { name: 'Get Started' })[0])

    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent('Checkout temporarily unavailable')
    })
    expect(alertSpy).not.toHaveBeenCalled()
    alertSpy.mockRestore()
  })

  it('shows malformed checkout responses inline in the pricing modal', async () => {
    const user = userEvent.setup()
    const alertSpy = vi.spyOn(window, 'alert').mockImplementation(() => {})
    global.fetch = vi.fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => buildStructuredPublicReport(),
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({}),
      } as Response)

    render(
      <MemoryRouter initialEntries={['/report?vendor=Zendesk&ref=test-token&mode=view']}>
        <Routes>
          <Route path="/report" element={<Report />} />
        </Routes>
      </MemoryRouter>,
    )

    await screen.findByText('Structured Report')
    await user.click(screen.getAllByRole('button', { name: 'Get Weekly Intelligence' })[0])
    await user.click(screen.getAllByRole('button', { name: 'Get Started' })[1])

    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent('Checkout session unavailable -- please try again.')
    })
    expect(alertSpy).not.toHaveBeenCalled()
    alertSpy.mockRestore()
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

  it('ignores stale checkout session fetches after the session route changes', async () => {
    let resolveFirst: ((value: Response) => void) | undefined
    let resolveSecond: ((value: Response) => void) | undefined

    global.fetch = vi.fn()
      .mockImplementationOnce(() => new Promise((resolve) => {
        resolveFirst = resolve as (value: Response) => void
      }))
      .mockImplementationOnce(() => new Promise((resolve) => {
        resolveSecond = resolve as (value: Response) => void
      }))

    const appRouter = createMemoryRouter(
      [{ path: '/report', element: <Report /> }],
      { initialEntries: ['/report?vendor=Zendesk&checkout=success&session_id=session-1'] },
    )

    render(<RouterProvider router={appRouter} />)

    await act(async () => {
      await appRouter.navigate('/report?vendor=Zendesk&checkout=success&session_id=session-2')
    })

    const resolveCurrentSession = resolveSecond
    if (!resolveCurrentSession) {
      throw new Error('Expected current checkout session resolver')
    }

    await act(async () => {
      resolveCurrentSession({
        ok: true,
        json: async () => ({ email: 'current@example.com', tier: 'standard' }),
      } as Response)
      await Promise.resolve()
      await Promise.resolve()
    })

    await waitFor(() => {
      expect(screen.getByDisplayValue('current@example.com')).toBeInTheDocument()
    })
    expect(screen.getByText("Your Zendesk Standard subscription is confirmed.")).toBeInTheDocument()

    const resolveStaleSession = resolveFirst
    if (!resolveStaleSession) {
      throw new Error('Expected stale checkout session resolver')
    }

    await act(async () => {
      resolveStaleSession({
        ok: true,
        json: async () => ({ email: 'stale@example.com', tier: 'pro' }),
      } as Response)
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(screen.getByDisplayValue('current@example.com')).toBeInTheDocument()
    expect(screen.queryByDisplayValue('stale@example.com')).not.toBeInTheDocument()
    expect(screen.getByText("Your Zendesk Standard subscription is confirmed.")).toBeInTheDocument()
  })

  it('clears checkout redirect timers when the session route changes', async () => {
    vi.useFakeTimers()
    global.fetch = vi.fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ email: 'first@example.com', tier: 'pro' }),
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ email: 'second@example.com', tier: 'standard' }),
      } as Response)

    const appRouter = createMemoryRouter(
      [{ path: '/report', element: <Report /> }],
      { initialEntries: ['/report?vendor=Zendesk&checkout=success&session_id=session-1'] },
    )

    render(<RouterProvider router={appRouter} />)

    await act(async () => {
      await Promise.resolve()
      await Promise.resolve()
    })
    expect(screen.getByDisplayValue('first@example.com')).toBeInTheDocument()
    fireEvent.change(screen.getByPlaceholderText('Jane Smith'), {
      target: { value: 'Juan Canfield' },
    })
    fireEvent.change(screen.getByPlaceholderText('Acme Inc.'), {
      target: { value: 'Atlas Labs' },
    })
    fireEvent.change(screen.getByPlaceholderText('Min. 8 characters'), {
      target: { value: 'password123' },
    })
    fireEvent.click(screen.getByRole('button', { name: 'Create Account' }))

    await act(async () => {
      await Promise.resolve()
      await Promise.resolve()
    })
    expect(screen.getByText('Account ready. Redirecting to Watchlists...')).toBeInTheDocument()

    await act(async () => {
      await appRouter.navigate('/report?vendor=Zendesk&checkout=success&session_id=session-2')
    })

    await act(async () => {
      await Promise.resolve()
      await Promise.resolve()
    })
    expect(screen.getByDisplayValue('second@example.com')).toBeInTheDocument()
    expect(screen.queryByText('Account ready. Redirecting to Watchlists...')).not.toBeInTheDocument()

    await act(async () => {
      vi.advanceTimersByTime(2000)
    })

    expect(router.navigate).not.toHaveBeenCalled()
    vi.useRealTimers()
  })
})
