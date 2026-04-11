import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import PublicLayout from './PublicLayout'

describe('PublicLayout', () => {
  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('keeps the default public shell pointed at the watchlists-first auth flow', () => {
    render(
      <MemoryRouter initialEntries={['/landing']}>
        <PublicLayout>
          <div>Landing Body</div>
        </PublicLayout>
      </MemoryRouter>,
    )

    expect(screen.getByText('Landing Body')).toBeInTheDocument()
    expect(screen.getAllByRole('link', { name: 'Sign in' })[0]).toHaveAttribute(
      'href',
      '/login?redirect_to=%2Fwatchlists&product=b2b_retention',
    )
    expect(screen.getByRole('link', { name: 'Start Free Trial' })).toHaveAttribute(
      'href',
      '/signup?redirect_to=%2Fwatchlists&product=b2b_retention',
    )
  })

  it('uses the report variant CTA without rendering the public auth links', async () => {
    const user = userEvent.setup()
    const onCtaClick = vi.fn()

    render(
      <MemoryRouter initialEntries={['/report']}>
        <PublicLayout variant="report" onCtaClick={onCtaClick}>
          <div>Report Body</div>
        </PublicLayout>
      </MemoryRouter>,
    )

    expect(screen.getByText('Report Body')).toBeInTheDocument()
    expect(screen.queryByRole('link', { name: 'Sign in' })).not.toBeInTheDocument()
    expect(screen.queryByRole('link', { name: 'Start Free Trial' })).not.toBeInTheDocument()

    const ctaButtons = screen.getAllByRole('button', { name: 'Get Weekly Intelligence' })
    expect(ctaButtons).toHaveLength(2)

    await user.click(ctaButtons[0])
    expect(onCtaClick).toHaveBeenCalledTimes(1)
  })
})
