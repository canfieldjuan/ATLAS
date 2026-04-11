import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import Signup from './Signup'

const auth = vi.hoisted(() => ({
  useAuth: vi.fn(),
}))

vi.mock('../auth/AuthContext', () => auth)

function OnboardingDestination() {
  const location = useLocation()
  return <div>{`${location.pathname}${location.search}`}</div>
}

describe('Signup', () => {
  const signup = vi.fn()

  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
    signup.mockResolvedValue(undefined)
    auth.useAuth.mockReturnValue({
      user: null,
      signup,
    })
  })

  it('navigates to onboarding with back_to after signup', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/signup?redirect_to=%2Fpredictor&product=b2b_challenger']}>
        <Routes>
          <Route path="/signup" element={<Signup />} />
          <Route path="/onboarding" element={<OnboardingDestination />} />
        </Routes>
      </MemoryRouter>,
    )

    await user.type(screen.getByPlaceholderText('Jane Smith'), 'Juan Canfield')
    await user.type(screen.getByPlaceholderText('Acme Inc.'), 'Atlas Labs')
    await user.type(screen.getByPlaceholderText('you@company.com'), 'juan@example.com')
    await user.type(screen.getByPlaceholderText('Min. 8 characters'), 'password123')
    await user.click(screen.getByRole('button', { name: 'Start Free Trial' }))

    await waitFor(() => {
      expect(signup).toHaveBeenCalledWith(
        'juan@example.com',
        'password123',
        'Juan Canfield',
        'Atlas Labs',
        'b2b_challenger',
      )
    })
    expect(await screen.findByText('/onboarding?back_to=%2Fpredictor')).toBeInTheDocument()
  })

  it('preserves redirect_to on the sign in link', () => {
    render(
      <MemoryRouter initialEntries={['/signup?redirect_to=%2Fwatchlists&product=b2b_challenger']}>
        <Signup />
      </MemoryRouter>,
    )

    expect(screen.getByRole('link', { name: 'Sign in' })).toHaveAttribute(
      'href',
      '/login?redirect_to=%2Fwatchlists&product=b2b_challenger',
    )
  })
})
