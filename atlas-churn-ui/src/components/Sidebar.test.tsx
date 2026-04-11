import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import Sidebar from './Sidebar'

const auth = vi.hoisted(() => ({
  useAuth: vi.fn(),
}))

const planGate = vi.hoisted(() => ({
  usePlanGate: vi.fn(),
}))

vi.mock('../auth/AuthContext', () => auth)
vi.mock('../hooks/usePlanGate', () => planGate)

describe('Sidebar', () => {
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

  it('prioritizes the primary product workflows ahead of supporting and operations links', () => {
    const { container } = render(
      <MemoryRouter initialEntries={['/watchlists']}>
        <Sidebar open onClose={() => {}} />
      </MemoryRouter>,
    )

    expect(screen.getByText('Product')).toBeInTheDocument()
    expect(screen.getByText('Supporting')).toBeInTheDocument()
    expect(screen.getAllByText('Operations').length).toBeGreaterThan(0)

    const navLabels = Array.from(container.querySelectorAll('nav a')).map((link) =>
      (link.textContent ?? '').replace(/\s+/g, ' ').trim(),
    )

    expect(navLabels.slice(0, 6)).toEqual([
      'Watchlists',
      'Vendors',
      'Evidence',
      'Opportunities',
      'Challengers',
      'Targets',
    ])
    expect(navLabels.indexOf('Reports')).toBeGreaterThan(navLabels.indexOf('Targets'))
    expect(navLabels.indexOf('Operations')).toBeGreaterThan(navLabels.indexOf('Blog'))
  })

  it('keeps gated supporting links visible in the secondary section', () => {
    planGate.usePlanGate.mockReturnValue({
      canAccessCampaigns: true,
      canAccessReports: false,
    })

    const { container } = render(
      <MemoryRouter initialEntries={['/watchlists']}>
        <Sidebar open onClose={() => {}} />
      </MemoryRouter>,
    )

    const navLabels = Array.from(container.querySelectorAll('nav a')).map((link) =>
      (link.textContent ?? '').replace(/\s+/g, ' ').trim(),
    )

    expect(navLabels).toContain('Reports')
    expect(navLabels.indexOf('Reports')).toBeGreaterThan(navLabels.indexOf('Reviews'))
    expect(navLabels.indexOf('Reports')).toBeLessThan(navLabels.indexOf('Alerts API'))
  })
})
