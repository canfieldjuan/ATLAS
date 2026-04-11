import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'
import Landing from './Landing'

vi.mock('../components/AtlasHeroScene', () => ({
  default: function AtlasHeroScene() {
    return <div data-testid="hero-scene">Atlas Hero</div>
  },
}))

describe('Landing', () => {
  it('routes the public shell into the watchlists-first product flow', () => {
    render(
      <MemoryRouter>
        <Landing />
      </MemoryRouter>,
    )

    const signInLinks = screen.getAllByRole('link', { name: 'Sign in' })
    expect(signInLinks).toHaveLength(2)
    signInLinks.forEach(link => {
      expect(link).toHaveAttribute('href', '/login?redirect_to=%2Fwatchlists&product=b2b_retention')
    })

    const signupLinks = screen.getAllByRole('link', { name: 'Sign up' })
    expect(signupLinks).toHaveLength(1)
    expect(signupLinks[0]).toHaveAttribute(
      'href',
      '/signup?product=b2b_retention&redirect_to=%2Fwatchlists',
    )

    const methodologyLinks = screen.getAllByRole('link', { name: 'Methodology' })
    expect(methodologyLinks).toHaveLength(2)
    methodologyLinks.forEach(link => {
      expect(link).toHaveAttribute('href', '/methodology')
    })
  })

  it('exposes product-specific entry paths for retention and challenger workflows', () => {
    render(
      <MemoryRouter>
        <Landing />
      </MemoryRouter>,
    )

    const retentionTrialLinks = screen.getAllByRole('link', { name: 'Start Vendor Retention Trial' })
    expect(retentionTrialLinks.length).toBeGreaterThan(0)
    retentionTrialLinks.forEach(link => {
      expect(link).toHaveAttribute('href', '/signup?product=b2b_retention&redirect_to=%2Fwatchlists')
    })
    const challengerTrialLinks = screen.getAllByRole('link', { name: 'Start Challenger Trial' })
    expect(challengerTrialLinks.length).toBeGreaterThan(0)
    challengerTrialLinks.forEach(link => {
      expect(link).toHaveAttribute('href', '/signup?product=b2b_challenger&redirect_to=%2Fchallengers')
    })
    const retentionWorkflowLinks = screen.getAllByRole('link', { name: 'Start Vendor Retention' })
    expect(retentionWorkflowLinks.length).toBeGreaterThan(0)
    retentionWorkflowLinks.forEach(link => {
      expect(link).toHaveAttribute('href', '/signup?product=b2b_retention&redirect_to=%2Fwatchlists')
    })

    const challengerWorkflowLinks = screen.getAllByRole('link', { name: 'Start Challenger Lead Gen' })
    expect(challengerWorkflowLinks.length).toBeGreaterThan(0)
    challengerWorkflowLinks.forEach(link => {
      expect(link).toHaveAttribute('href', '/signup?product=b2b_challenger&redirect_to=%2Fchallengers')
    })
    const watchlistSignInLinks = screen.getAllByRole('link', { name: 'Sign in to Watchlists' })
    expect(watchlistSignInLinks.length).toBeGreaterThan(0)
    watchlistSignInLinks.forEach(link => {
      expect(link).toHaveAttribute('href', '/login?redirect_to=%2Fwatchlists&product=b2b_retention')
    })

    const challengerSignInLinks = screen.getAllByRole('link', { name: 'Sign in to Challengers' })
    expect(challengerSignInLinks.length).toBeGreaterThan(0)
    challengerSignInLinks.forEach(link => {
      expect(link).toHaveAttribute('href', '/login?redirect_to=%2Fchallengers&product=b2b_challenger')
    })
  })
})
