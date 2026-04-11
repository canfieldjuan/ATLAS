import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import ErrorBoundary, { PageError } from './ErrorBoundary'

describe('PageError', () => {
  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('renders the error message and retry action', async () => {
    const user = userEvent.setup()
    const onRetry = vi.fn()

    render(<PageError error={new Error('API timed out')} onRetry={onRetry} />)

    expect(screen.getByText('Something went wrong')).toBeInTheDocument()
    expect(screen.getByText('API timed out')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Try Again' }))
    expect(onRetry).toHaveBeenCalledTimes(1)
  })
})

describe('ErrorBoundary', () => {
  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('catches a render error and recovers after retry', async () => {
    const user = userEvent.setup()
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {})
    let shouldThrow = true

    function BuggyChild() {
      if (shouldThrow) {
        throw new Error('Exploded during render')
      }
      return <div>Recovered content</div>
    }

    render(
      <ErrorBoundary>
        <BuggyChild />
      </ErrorBoundary>,
    )

    expect(await screen.findByText('Exploded during render')).toBeInTheDocument()

    shouldThrow = false
    await user.click(screen.getByRole('button', { name: 'Try Again' }))
    expect(screen.getByText('Recovered content')).toBeInTheDocument()

    consoleError.mockRestore()
  })
})
