import { cleanup, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'
import AtlasRobotLogo from './AtlasRobotLogo'

describe('AtlasRobotLogo', () => {
  beforeEach(() => {
    cleanup()
  })

  it('renders the robot svg with the default accessible label', () => {
    render(<AtlasRobotLogo />)

    const logo = screen.getByLabelText('Atlas Robot')
    expect(logo.tagName).toBe('svg')
    expect(logo).toHaveClass('h-7', 'w-7')
  })

  it('accepts a custom class name for shell contexts', () => {
    render(<AtlasRobotLogo className="h-10 w-10 text-cyan-400" />)

    expect(screen.getByLabelText('Atlas Robot')).toHaveClass('h-10', 'w-10', 'text-cyan-400')
  })
})
