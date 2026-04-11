import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import Blog from './Blog'

const blogModule = vi.hoisted(() => ({
  loadAllPosts: vi.fn(),
}))

vi.mock('../content/blog', () => blogModule)
vi.mock('../components/AtlasRobotScene', () => ({
  default: function AtlasRobotScene() {
    return <div data-testid="robot-scene">Atlas Robot Scene</div>
  },
}))

describe('Blog', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    blogModule.loadAllPosts.mockResolvedValue([])
  })

  it('routes blog readers into the primary product workflows', async () => {
    render(
      <MemoryRouter>
        <Blog />
      </MemoryRouter>,
    )

    expect(await screen.findByRole('link', { name: 'Start Vendor Retention' })).toHaveAttribute(
      'href',
      '/signup?product=b2b_retention&redirect_to=%2Fwatchlists',
    )
    expect(screen.getByRole('link', { name: 'Start Challenger Lead Gen' })).toHaveAttribute(
      'href',
      '/signup?product=b2b_challenger&redirect_to=%2Fchallengers',
    )
    expect(screen.getByRole('link', { name: 'Sign in to Watchlists' })).toHaveAttribute(
      'href',
      '/login?redirect_to=%2Fwatchlists&product=b2b_retention',
    )
    expect(screen.getByRole('link', { name: 'Sign in to Challengers' })).toHaveAttribute(
      'href',
      '/login?redirect_to=%2Fchallengers&product=b2b_challenger',
    )
    expect(screen.getByRole('link', { name: 'Read Methodology' })).toHaveAttribute('href', '/methodology')
  })
})
