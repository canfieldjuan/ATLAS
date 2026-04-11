import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import Onboarding from './Onboarding'

const auth = vi.hoisted(() => ({
  useAuth: vi.fn(),
}))

const api = vi.hoisted(() => ({
  searchAvailableVendors: vi.fn(),
  addTrackedVendor: vi.fn(),
  removeTrackedVendor: vi.fn(),
  listTrackedVendors: vi.fn(),
}))

vi.mock('../auth/AuthContext', () => auth)
vi.mock('../api/client', () => api)

describe('Onboarding', () => {
  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
    auth.useAuth.mockReturnValue({
      user: {
        vendor_limit: 5,
        product: 'b2b_growth',
      },
      refreshUser: vi.fn().mockResolvedValue(undefined),
    })
    api.searchAvailableVendors.mockResolvedValue({ vendors: [] })
    api.addTrackedVendor.mockResolvedValue({})
    api.removeTrackedVendor.mockResolvedValue({})
    api.listTrackedVendors.mockResolvedValue({ vendors: [] })
  })

  it('hydrates the search query from the URL', async () => {
    render(
      <MemoryRouter initialEntries={['/onboarding?q=Zendesk']}>
        <Onboarding />
      </MemoryRouter>,
    )

    expect(await screen.findByDisplayValue('Zendesk')).toBeInTheDocument()
  })

  it('links tracked vendors into the vendor workspace with onboarding back_to', async () => {
    api.listTrackedVendors.mockResolvedValue({
      vendors: [{ vendor_name: 'Zendesk' }],
    })

    render(
      <MemoryRouter initialEntries={['/onboarding?q=Zendesk']}>
        <Onboarding />
      </MemoryRouter>,
    )

    expect(await screen.findByRole('link', { name: 'Zendesk' })).toHaveAttribute(
      'href',
      '/vendors/Zendesk?back_to=%2Fonboarding%3Fq%3DZendesk',
    )
    expect(screen.getByRole('button', { name: 'Continue to watchlists' })).toBeInTheDocument()
  })
})
