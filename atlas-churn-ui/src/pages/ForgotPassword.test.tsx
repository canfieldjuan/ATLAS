import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it } from 'vitest'
import ForgotPassword from './ForgotPassword'

describe('ForgotPassword', () => {
  beforeEach(() => {
    cleanup()
  })

  it('preserves redirect_to and product on the sign-in link', () => {
    render(
      <MemoryRouter initialEntries={['/forgot-password?redirect_to=%2Fchallengers&product=b2b_challenger']}>
        <ForgotPassword />
      </MemoryRouter>,
    )

    expect(screen.getByRole('link', { name: /back to sign in/i })).toHaveAttribute(
      'href',
      '/login?redirect_to=%2Fchallengers&product=b2b_challenger',
    )
  })

  it('normalizes invalid redirect_to on the sign-in link', () => {
    render(
      <MemoryRouter initialEntries={['/forgot-password?redirect_to=https%3A%2F%2Fevil.example&product=b2b_retention']}>
        <ForgotPassword />
      </MemoryRouter>,
    )

    expect(screen.getByRole('link', { name: /back to sign in/i })).toHaveAttribute(
      'href',
      '/login?redirect_to=%2Fwatchlists&product=b2b_retention',
    )
  })
})
