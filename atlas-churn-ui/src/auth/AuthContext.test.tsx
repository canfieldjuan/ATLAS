import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import AuthProvider, { tryRefreshToken, useAuth } from './AuthContext'

function jsonResponse(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 200 ? 'OK' : 'Error',
    json: vi.fn().mockResolvedValue(body),
  } as unknown as Response
}

function AuthConsumer() {
  const auth = useAuth()

  return (
    <div>
      <div>{auth.loading ? 'loading' : 'ready'}</div>
      <div>{auth.user?.email ?? 'anon'}</div>
      <div>{auth.token ?? 'no-token'}</div>
      <button onClick={() => void auth.login('owner@atlas.test', 'secret')}>login</button>
      <button onClick={() => auth.logout()}>logout</button>
      <button onClick={() => void auth.refreshUser()}>refresh</button>
    </div>
  )
}

describe('AuthContext', () => {
  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
    localStorage.clear()
    globalThis.fetch = vi.fn()
  })

  afterEach(() => {
    cleanup()
    localStorage.clear()
  })

  it('logs in, hydrates the user, and logs out cleanly', async () => {
    const user = userEvent.setup()
    vi.mocked(fetch)
      .mockResolvedValueOnce(
        jsonResponse(200, {
          access_token: 'access-1',
          refresh_token: 'refresh-1',
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse(200, {
          user_id: 'user-1',
          email: 'owner@atlas.test',
          full_name: 'Atlas Owner',
          role: 'admin',
          account_id: 'acct-1',
          account_name: 'Atlas',
          plan: 'pro',
          plan_status: 'active',
          asin_limit: 0,
          trial_ends_at: null,
          product: 'b2b_retention',
          vendor_limit: 100,
        }),
      )

    render(
      <AuthProvider>
        <AuthConsumer />
      </AuthProvider>,
    )

    expect(await screen.findByText('ready')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'login' }))

    expect(await screen.findByText('owner@atlas.test')).toBeInTheDocument()
    expect(screen.getByText('access-1')).toBeInTheDocument()
    expect(localStorage.getItem('atlas_token')).toBe('access-1')
    expect(localStorage.getItem('atlas_refresh_token')).toBe('refresh-1')

    await user.click(screen.getByRole('button', { name: 'logout' }))
    expect(screen.getByText('anon')).toBeInTheDocument()
    expect(screen.getByText('no-token')).toBeInTheDocument()
    expect(localStorage.getItem('atlas_token')).toBeNull()
    expect(localStorage.getItem('atlas_refresh_token')).toBeNull()
  })

  it('refreshes the access token when the stored token fails /me', async () => {
    localStorage.setItem('atlas_token', 'expired-token')
    localStorage.setItem('atlas_refresh_token', 'refresh-1')

    vi.mocked(fetch)
      .mockResolvedValueOnce(jsonResponse(401, { detail: 'expired' }))
      .mockResolvedValueOnce(
        jsonResponse(200, {
          access_token: 'access-2',
          refresh_token: 'refresh-2',
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse(200, {
          user_id: 'user-2',
          email: 'renewed@atlas.test',
          full_name: 'Renewed User',
          role: 'admin',
          account_id: 'acct-2',
          account_name: 'Atlas',
          plan: 'pro',
          plan_status: 'active',
          asin_limit: 0,
          trial_ends_at: null,
          product: 'b2b_retention',
          vendor_limit: 100,
        }),
      )

    render(
      <AuthProvider>
        <AuthConsumer />
      </AuthProvider>,
    )

    expect(await screen.findByText('renewed@atlas.test')).toBeInTheDocument()
    expect(screen.getByText('access-2')).toBeInTheDocument()
    expect(localStorage.getItem('atlas_token')).toBe('access-2')
    expect(localStorage.getItem('atlas_refresh_token')).toBe('refresh-2')
  })

  it('deduplicates concurrent refresh token requests', async () => {
    localStorage.setItem('atlas_refresh_token', 'refresh-shared')

    let resolveFetch: ((value: Response) => void) | null = null
    vi.mocked(fetch).mockImplementation(
      () =>
        new Promise<Response>((resolve) => {
          resolveFetch = resolve
        }),
    )

    const pendingOne = tryRefreshToken()
    const pendingTwo = tryRefreshToken()

    expect(fetch).toHaveBeenCalledTimes(1)
    const finishFetch = resolveFetch as ((value: Response) => void) | null
    if (typeof finishFetch !== 'function') {
      throw new Error('refresh fetch was not captured')
    }
    finishFetch(
      jsonResponse(200, {
        access_token: 'access-shared',
        refresh_token: 'refresh-next',
      }),
    )

    await expect(Promise.all([pendingOne, pendingTwo])).resolves.toEqual([
      'access-shared',
      'access-shared',
    ])
    await waitFor(() => {
      expect(localStorage.getItem('atlas_token')).toBe('access-shared')
      expect(localStorage.getItem('atlas_refresh_token')).toBe('refresh-next')
    })
  })

  it('does not retry /api fallback when refresh returns 404', async () => {
    localStorage.setItem('atlas_refresh_token', 'refresh-missing')
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse(404, { detail: 'missing route' }))

    await expect(tryRefreshToken()).resolves.toBeNull()

    expect(fetch).toHaveBeenCalledTimes(1)
    expect(String(vi.mocked(fetch).mock.calls[0]?.[0] ?? '')).toContain('/api/v1/auth/refresh')
  })
})
