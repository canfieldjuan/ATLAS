import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'

const auth = vi.hoisted(() => ({
  useAuth: vi.fn(),
}))

vi.mock('./auth/AuthContext', () => auth)

vi.mock('./components/Layout', () => ({
  default: function LayoutMock({ children }: { children: React.ReactNode }) {
    return <div data-testid="layout-shell">{children}</div>
  },
}))

vi.mock('./pages/Watchlists', () => ({
  default: function WatchlistsMock() {
    return <div>Watchlists Page</div>
  },
}))

vi.mock('./pages/Landing', () => ({
  default: function LandingMock() {
    return <div>Landing Page</div>
  },
}))

vi.mock('./pages/IncidentAlerts', async () => {
  const { useLocation } = await vi.importActual<typeof import('react-router-dom')>('react-router-dom')
  return {
    default: function IncidentAlertsMock() {
      const location = useLocation()
      return <div>{`Incident Alerts ${location.pathname}`}</div>
    },
  }
})

describe('App routing', () => {
  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
    auth.useAuth.mockReturnValue({
      user: {
        user_id: 'user-1',
        email: 'juan@example.com',
        full_name: 'Juan',
        role: 'owner',
        account_id: 'acct-1',
        account_name: 'Atlas',
        plan: 'b2b_pro',
        plan_status: 'active',
        asin_limit: 0,
        trial_ends_at: null,
        product: 'b2b_retention',
        vendor_limit: -1,
      },
      loading: false,
    })
  })

  it('redirects the root route into the watchlists-first product flow', async () => {
    render(
      <MemoryRouter initialEntries={['/']}>
        <App />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Watchlists Page')).toBeInTheDocument()
    expect(screen.getByTestId('layout-shell')).toBeInTheDocument()
  })

  it('keeps landing public and outside the protected layout shell', async () => {
    render(
      <MemoryRouter initialEntries={['/landing']}>
        <App />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Landing Page')).toBeInTheDocument()
    expect(screen.queryByTestId('layout-shell')).not.toBeInTheDocument()
  })

  it('redirects the legacy alerts path into the canonical alerts route', async () => {
    render(
      <MemoryRouter initialEntries={['/alerts-api']}>
        <App />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Incident Alerts /alerts')).toBeInTheDocument()
    expect(screen.getByTestId('layout-shell')).toBeInTheDocument()
  })
})
