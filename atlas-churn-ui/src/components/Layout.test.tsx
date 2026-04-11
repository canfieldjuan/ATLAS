import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import Layout from './Layout'

const auth = vi.hoisted(() => ({
  useAuth: vi.fn(),
}))

const planGate = vi.hoisted(() => ({
  usePlanGate: vi.fn(),
}))

vi.mock('../auth/AuthContext', () => auth)
vi.mock('../hooks/usePlanGate', () => planGate)

describe('Layout', () => {
  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
    auth.useAuth.mockReturnValue({
      user: { email: 'juan@example.com' },
      logout: vi.fn(),
    })
    planGate.usePlanGate.mockReturnValue({
      canAccessCampaigns: true,
      canAccessReports: true,
    })
  })

  it('opens and closes the mobile navigation from the header controls', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/watchlists']}>
        <Layout>
          <div>Primary Workspace</div>
        </Layout>
      </MemoryRouter>,
    )

    expect(screen.getByText('Primary Workspace')).toBeInTheDocument()
    expect(screen.queryByTestId('sidebar-backdrop')).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Open navigation' }))
    expect(screen.getByTestId('sidebar-backdrop')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Close navigation' }))
    await waitFor(() => {
      expect(screen.queryByTestId('sidebar-backdrop')).not.toBeInTheDocument()
    })
  })

  it('closes the mobile navigation when the backdrop is clicked', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/watchlists']}>
        <Layout>
          <div>Primary Workspace</div>
        </Layout>
      </MemoryRouter>,
    )

    await user.click(screen.getByRole('button', { name: 'Open navigation' }))
    expect(screen.getByTestId('sidebar-backdrop')).toBeInTheDocument()

    await user.click(screen.getByTestId('sidebar-backdrop'))
    await waitFor(() => {
      expect(screen.queryByTestId('sidebar-backdrop')).not.toBeInTheDocument()
    })
  })
})
