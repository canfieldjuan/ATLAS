import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import Account from './Account'

const auth = vi.hoisted(() => ({
  useAuth: vi.fn(),
}))

vi.mock('../auth/AuthContext', () => auth)

describe('Account', () => {
  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
    auth.useAuth.mockReturnValue({
      user: {
        full_name: 'Juan Canfield',
        email: 'juan@example.com',
        account_name: 'Atlas Labs',
        plan: 'b2b_growth',
        plan_status: 'active',
        vendor_limit: 25,
        product: 'b2b_growth',
        trial_ends_at: null,
      },
      logout: vi.fn(),
    })
  })

  it('links account users back into the product surfaces', () => {
    render(
      <MemoryRouter initialEntries={['/account']}>
        <Account />
      </MemoryRouter>,
    )

    expect(screen.getByRole('link', { name: 'Open Watchlists' })).toHaveAttribute('href', '/watchlists')
    expect(screen.getByRole('link', { name: 'Browse Vendors' })).toHaveAttribute('href', '/vendors')
    expect(screen.getByRole('link', { name: 'Manage Tracked Vendors' })).toHaveAttribute(
      'href',
      '/onboarding?back_to=%2Faccount',
    )
  })
})
