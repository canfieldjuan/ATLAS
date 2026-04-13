import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, RouterProvider, Routes, createMemoryRouter } from 'react-router-dom'
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

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

describe('Onboarding', () => {
  const refreshUser = vi.fn()

  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
    refreshUser.mockResolvedValue(undefined)
    auth.useAuth.mockReturnValue({
      user: {
        vendor_limit: 5,
        product: 'b2b_growth',
      },
      refreshUser,
    })
    api.searchAvailableVendors.mockResolvedValue({ vendors: [] })
    api.addTrackedVendor.mockResolvedValue({})
    api.removeTrackedVendor.mockResolvedValue({})
    api.listTrackedVendors.mockResolvedValue({ vendors: [] })
  })

  it('hydrates the search query from the URL and loads matching results', async () => {
    api.searchAvailableVendors.mockResolvedValue({
      vendors: [{ vendor_name: 'Zendesk', product_category: 'Helpdesk', total_reviews: 120, avg_urgency: 6.2 }],
    })

    render(
      <MemoryRouter initialEntries={['/onboarding?q=Zendesk']}>
        <Onboarding />
      </MemoryRouter>,
    )

    expect(await screen.findByDisplayValue('Zendesk')).toBeInTheDocument()
    await waitFor(() => {
      expect(api.searchAvailableVendors).toHaveBeenCalledWith('Zendesk')
    })
    expect(await screen.findByText('Zendesk')).toBeInTheDocument()
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

  it('preserves back_to when continuing', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/onboarding?back_to=%2Faccount&q=Zendesk']}>
        <Routes>
          <Route path="/onboarding" element={<Onboarding />} />
          <Route path="/account" element={<div>Account Destination</div>} />
        </Routes>
      </MemoryRouter>,
    )

    await user.click(await screen.findByRole('button', { name: 'Skip for now' }))

    expect(refreshUser).toHaveBeenCalledTimes(1)
    expect(await screen.findByText('Account Destination')).toBeInTheDocument()
  })

  it('rehydrates the query from same-route URL changes', async () => {
    api.searchAvailableVendors
      .mockResolvedValueOnce({ vendors: [{ vendor_name: 'Zendesk', product_category: 'Helpdesk', total_reviews: 120, avg_urgency: 6.2 }] })
      .mockResolvedValueOnce({ vendors: [{ vendor_name: 'HubSpot', product_category: 'CRM', total_reviews: 220, avg_urgency: 5.8 }] })

    const router = createMemoryRouter([{ path: '/onboarding', element: <Onboarding /> }], {
      initialEntries: ['/onboarding?q=Zendesk'],
    })

    render(<RouterProvider router={router} />)

    expect(await screen.findByDisplayValue('Zendesk')).toBeInTheDocument()
    expect(await screen.findByText('Zendesk')).toBeInTheDocument()

    await router.navigate('/onboarding?q=HubSpot')

    await waitFor(() => {
      expect(screen.getByDisplayValue('HubSpot')).toBeInTheDocument()
      expect(api.searchAvailableVendors).toHaveBeenLastCalledWith('HubSpot')
    })
    expect(await screen.findByText('HubSpot')).toBeInTheDocument()
    router.dispose()
  })

  it('ignores stale search responses after same-route query changes', async () => {
    const zendesk = deferred<{ vendors: Array<{ vendor_name: string; product_category: string; total_reviews: number; avg_urgency: number }> }>()
    const hubspot = deferred<{ vendors: Array<{ vendor_name: string; product_category: string; total_reviews: number; avg_urgency: number }> }>()

    api.searchAvailableVendors.mockImplementation((query: string) => {
      if (query === 'Zendesk') return zendesk.promise
      if (query === 'HubSpot') return hubspot.promise
      return Promise.resolve({ vendors: [] })
    })

    const router = createMemoryRouter([{ path: '/onboarding', element: <Onboarding /> }], {
      initialEntries: ['/onboarding?q=Zendesk'],
    })

    render(<RouterProvider router={router} />)

    expect(await screen.findByDisplayValue('Zendesk')).toBeInTheDocument()
    await waitFor(() => {
      expect(api.searchAvailableVendors).toHaveBeenCalledWith('Zendesk')
    })

    await router.navigate('/onboarding?q=HubSpot')

    await waitFor(() => {
      expect(screen.getByDisplayValue('HubSpot')).toBeInTheDocument()
      expect(api.searchAvailableVendors).toHaveBeenLastCalledWith('HubSpot')
    })

    zendesk.resolve({ vendors: [{ vendor_name: 'Zendesk', product_category: 'Helpdesk', total_reviews: 120, avg_urgency: 6.2 }] })
    await waitFor(() => {
      expect(screen.queryByText('Zendesk')).not.toBeInTheDocument()
    })

    hubspot.resolve({ vendors: [{ vendor_name: 'HubSpot', product_category: 'CRM', total_reviews: 220, avg_urgency: 5.8 }] })
    expect(await screen.findByText('HubSpot')).toBeInTheDocument()
    router.dispose()
  })
})
