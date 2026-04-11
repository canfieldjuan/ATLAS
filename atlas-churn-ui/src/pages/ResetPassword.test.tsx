import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import ResetPassword from './ResetPassword'

describe('ResetPassword', () => {
  it('preserves redirect_to and product on the sign-in link', () => {
    render(
      <MemoryRouter initialEntries={['/reset-password?token=test-token&redirect_to=%2Fwatchlists&product=b2b_retention']}>
        <ResetPassword />
      </MemoryRouter>,
    )

    expect(screen.getByRole('link', { name: /back to sign in/i })).toHaveAttribute(
      'href',
      '/login?redirect_to=%2Fwatchlists&product=b2b_retention',
    )
  })
})
