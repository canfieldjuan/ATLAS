"use client";

/* eslint-disable react-refresh/only-export-components, react-hooks/set-state-in-effect */
import { createContext, useContext, useState, useEffect, useCallback, useRef, type ReactNode } from 'react'

const TOKEN_KEY = 'atlas_token'
const REFRESH_KEY = 'atlas_refresh_token'

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || ''
const BASE = `${API_BASE}/api/v1`

async function fetchWithApiFallback(input: string, opts?: RequestInit): Promise<Response> {
  const res = await fetch(input, opts)
  if (res.status !== 404) return res
  if (!input.includes('/api/v1/')) return res
  const fallbackInput = input.replace('/api/v1/', '/api/')
  return fetch(fallbackInput, opts)
}

export interface User {
  user_id: string
  email: string
  full_name: string | null
  role: string
  account_id: string
  account_name: string
  plan: string
  plan_status: string
  asin_limit: number
  trial_ends_at: string | null
  product: string       // consumer | b2b_retention | b2b_challenger
  vendor_limit: number
}

interface AuthState {
  user: User | null
  token: string | null
  loading: boolean
  login: (email: string, password: string) => Promise<void>
  signup: (email: string, password: string, fullName: string, accountName: string, product?: string) => Promise<void>
  logout: () => void
  refreshUser: () => Promise<void>
}

const AuthContext = createContext<AuthState | null>(null)

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used inside <AuthProvider>')
  return ctx
}

async function apiFetch<T>(path: string, opts?: RequestInit): Promise<T> {
  const { headers: extraHeaders, ...restOpts } = opts ?? {}
  const res = await fetchWithApiFallback(`${BASE}${path}`, {
    ...restOpts,
    headers: { 'Content-Type': 'application/json', ...(extraHeaders as Record<string, string>) },
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(body.detail || `${res.status} ${res.statusText}`)
  }
  return res.json()
}

// ---------------------------------------------------------------------------
// Refresh token logic (shared across client.ts and AuthContext)
// ---------------------------------------------------------------------------

let refreshPromise: Promise<string | null> | null = null

/**
 * Attempt to exchange the stored refresh token for a new access token.
 * Deduplicates concurrent calls -- only one refresh request is in-flight at a time.
 * Returns the new access token on success, or null on failure (caller should logout).
 */
export async function tryRefreshToken(): Promise<string | null> {
  // Deduplicate: if a refresh is already in-flight, piggyback on it
  if (refreshPromise) return refreshPromise

  refreshPromise = (async () => {
    const refreshToken = typeof window !== 'undefined' ? (typeof window !== 'undefined' ? localStorage.getItem(REFRESH_KEY) : null) : null
    if (!refreshToken) return null

    try {
      const res = await fetchWithApiFallback(`${BASE}/auth/refresh`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: refreshToken }),
      })
      if (!res.ok) return null

      const data: { access_token: string; refresh_token: string } = await res.json()
      typeof window !== 'undefined' && localStorage.setItem(TOKEN_KEY, data.access_token)
      typeof window !== 'undefined' && localStorage.setItem(REFRESH_KEY, data.refresh_token)
      return data.access_token
    } catch {
      return null
    } finally {
      refreshPromise = null
    }
  })()

  return refreshPromise
}

export default function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [token, setToken] = useState<string | null>(() => (typeof window !== 'undefined' ? localStorage.getItem(TOKEN_KEY) : null))
  const [loading, setLoading] = useState(true)
  const tokenRef = useRef(token)

  useEffect(() => {
    tokenRef.current = token
  }, [token])

  const saveTokens = useCallback((access: string, refresh: string) => {
    typeof window !== 'undefined' && localStorage.setItem(TOKEN_KEY, access)
    typeof window !== 'undefined' && localStorage.setItem(REFRESH_KEY, refresh)
    setToken(access)
  }, [])

  const clearTokens = useCallback(() => {
    typeof window !== 'undefined' && localStorage.removeItem(TOKEN_KEY)
    typeof window !== 'undefined' && localStorage.removeItem(REFRESH_KEY)
    setToken(null)
    setUser(null)
  }, [])

  const fetchMe = useCallback(async (t: string) => {
    try {
      const u = await apiFetch<User>('/auth/me', {
        headers: { Authorization: `Bearer ${t}` },
      })
      setUser(u)
    } catch {
      // Access token failed -- try refresh before giving up
      const newToken = await tryRefreshToken()
      if (newToken) {
        setToken(newToken)
        try {
          const u = await apiFetch<User>('/auth/me', {
            headers: { Authorization: `Bearer ${newToken}` },
          })
          setUser(u)
          return
        } catch {
          // Refresh succeeded but /me still failed -- give up
        }
      }
      clearTokens()
    }
  }, [clearTokens])

  useEffect(() => {
    if (token) {
      fetchMe(token).finally(() => setLoading(false))
    } else {
      setLoading(false)
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const login = useCallback(async (email: string, password: string) => {
    const res = await apiFetch<{ access_token: string; refresh_token: string }>('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    })
    saveTokens(res.access_token, res.refresh_token)
    await fetchMe(res.access_token)
  }, [saveTokens, fetchMe])

  const signup = useCallback(async (email: string, password: string, fullName: string, accountName: string, product?: string) => {
    const res = await apiFetch<{ access_token: string; refresh_token: string }>('/auth/register', {
      method: 'POST',
      body: JSON.stringify({ email, password, full_name: fullName, account_name: accountName, product: product || 'consumer' }),
    })
    saveTokens(res.access_token, res.refresh_token)
    await fetchMe(res.access_token)
  }, [saveTokens, fetchMe])

  const logout = useCallback(() => {
    clearTokens()
  }, [clearTokens])

  const refreshUser = useCallback(async () => {
    const t = tokenRef.current
    if (t) await fetchMe(t)
  }, [fetchMe])

  return (
    <AuthContext.Provider value={{ user, token, loading, login, signup, logout, refreshUser }}>
      {children}
    </AuthContext.Provider>
  )
}
