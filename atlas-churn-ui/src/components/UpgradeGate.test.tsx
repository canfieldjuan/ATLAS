import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it } from 'vitest'
import UpgradeGate from './UpgradeGate'

describe('UpgradeGate', () => {
  beforeEach(() => {
    cleanup()
  })

  it('renders children when access is allowed', () => {
    render(
      <MemoryRouter>
        <UpgradeGate allowed feature="Intelligence Library" requiredPlan="Starter">
          <div>Unlocked content</div>
        </UpgradeGate>
      </MemoryRouter>,
    )

    expect(screen.getByText('Unlocked content')).toBeInTheDocument()
  })

  it('renders the upgrade prompt when access is blocked', () => {
    render(
      <MemoryRouter>
        <UpgradeGate allowed={false} feature="Intelligence Library" requiredPlan="Starter">
          <div>Unlocked content</div>
        </UpgradeGate>
      </MemoryRouter>,
    )

    expect(
      screen.getByRole('heading', { name: 'Intelligence Library requires Starter' }),
    ).toBeInTheDocument()
    expect(screen.queryByText('Unlocked content')).not.toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Upgrade Plan' })).toHaveAttribute('href', '/account')
  })
})
