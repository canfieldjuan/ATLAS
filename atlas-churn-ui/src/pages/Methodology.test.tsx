import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import Methodology from './Methodology'

describe('Methodology', () => {
  it('links methodology readers into the product workflows', () => {
    render(
      <MemoryRouter>
        <Methodology />
      </MemoryRouter>,
    )

    expect(screen.getByRole('link', { name: 'Open Watchlists Workflow' })).toHaveAttribute(
      'href',
      '/signup?product=b2b_retention&redirect_to=%2Fwatchlists',
    )
    expect(screen.getByRole('link', { name: 'Sign in to Watchlists' })).toHaveAttribute(
      'href',
      '/login?product=b2b_retention&redirect_to=%2Fwatchlists',
    )
    expect(screen.getByRole('link', { name: 'Open Challenger Workflow' })).toHaveAttribute(
      'href',
      '/signup?product=b2b_challenger&redirect_to=%2Fchallengers',
    )
    expect(screen.getByRole('link', { name: 'Sign in to Challengers' })).toHaveAttribute(
      'href',
      '/login?product=b2b_challenger&redirect_to=%2Fchallengers',
    )
    expect(screen.getByRole('link', { name: 'Browse Blog Analysis' })).toHaveAttribute('href', '/blog')
  })
})
