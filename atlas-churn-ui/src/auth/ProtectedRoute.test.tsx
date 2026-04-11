import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import ProtectedRoute from './ProtectedRoute'

const auth = vi.hoisted(() => ({
  useAuth: vi.fn(),
}))

vi.mock('./AuthContext', () => auth)

function LocationEcho() {
  const location = useLocation()
  return <div>{`${location.pathname}${location.search}${location.hash}`}</div>
}

describe('ProtectedRoute', () => {
  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('redirects unauthenticated users to login with redirect_to', async () => {
    auth.useAuth.mockReturnValue({
      user: null,
      loading: false,
    })

    render(
      <MemoryRouter initialEntries={['/challengers?search=Zendesk#signals']}>
        <Routes>
          <Route
            path="/challengers"
            element={(
              <ProtectedRoute>
                <div>Secure Challenger</div>
              </ProtectedRoute>
            )}
          />
          <Route path="/login" element={<LocationEcho />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(
      await screen.findByText('/login?redirect_to=%2Fchallengers%3Fsearch%3DZendesk%23signals&product=b2b_challenger'),
    ).toBeInTheDocument()
  })

  it('renders children for authenticated users', () => {
    auth.useAuth.mockReturnValue({
      user: { user_id: 'user-1' },
      loading: false,
    })

    render(
      <MemoryRouter initialEntries={['/watchlists']}>
        <Routes>
          <Route
            path="/watchlists"
            element={(
              <ProtectedRoute>
                <div>Watchlists Workspace</div>
              </ProtectedRoute>
            )}
          />
        </Routes>
      </MemoryRouter>,
    )

    expect(screen.getByText('Watchlists Workspace')).toBeInTheDocument()
  })
})
