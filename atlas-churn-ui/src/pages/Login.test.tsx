import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import Login from './Login'

const auth = vi.hoisted(() => ({
  useAuth: vi.fn(),
}))

vi.mock('../auth/AuthContext', () => auth)

describe('Login', () => {
  const login = vi.fn()

  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
    login.mockResolvedValue(undefined)
    auth.useAuth.mockReturnValue({
      user: null,
      login,
    })
  })

  it('navigates to redirect_to after sign in', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/login?redirect_to=%2Fwatchlists']}>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/watchlists" element={<div>Watchlists Destination</div>} />
        </Routes>
      </MemoryRouter>,
    )

    await user.type(screen.getByPlaceholderText('you@company.com'), 'juan@example.com')
    await user.type(screen.getByPlaceholderText('********'), 'password123')
    await user.click(screen.getByRole('button', { name: 'Sign in' }))

    await waitFor(() => {
      expect(login).toHaveBeenCalledWith('juan@example.com', 'password123')
    })
    expect(await screen.findByText('Watchlists Destination')).toBeInTheDocument()
  })

  it('preserves redirect_to and product on account-creation and recovery links', () => {
    render(
      <MemoryRouter initialEntries={['/login?redirect_to=%2Fchallengers&product=b2b_challenger']}>
        <Login />
      </MemoryRouter>,
    )

    expect(screen.getByRole('link', { name: 'Create one' })).toHaveAttribute(
      'href',
      '/signup?redirect_to=%2Fchallengers&product=b2b_challenger',
    )
    expect(screen.getByRole('link', { name: 'Forgot password?' })).toHaveAttribute(
      'href',
      '/forgot-password?redirect_to=%2Fchallengers&product=b2b_challenger',
    )
  })

  it('normalizes invalid redirect_to on account-creation and recovery links', () => {
    render(
      <MemoryRouter initialEntries={['/login?redirect_to=https%3A%2F%2Fevil.example&product=b2b_challenger']}>
        <Login />
      </MemoryRouter>,
    )

    expect(screen.getByRole('link', { name: 'Create one' })).toHaveAttribute(
      'href',
      '/signup?redirect_to=%2Fwatchlists&product=b2b_challenger',
    )
    expect(screen.getByRole('link', { name: 'Forgot password?' })).toHaveAttribute(
      'href',
      '/forgot-password?redirect_to=%2Fwatchlists&product=b2b_challenger',
    )
  })
})
