import { createContext, useContext, useState, useEffect, useCallback, type ReactNode } from 'react'

const TOKEN_KEY = 'atlas_token'
const REFRESH_KEY = 'atlas_refresh_token'
const BASE = '/api/v1'

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
}

interface AuthState {
  user: User | null
  token: string | null
  loading: boolean
  login: (email: string, password: string) => Promise<void>
  signup: (email: string, password: string, fullName: string, accountName: string) => Promise<void>
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
  const res = await fetch(`${BASE}${path}`, {
    ...restOpts,
    headers: { 'Content-Type': 'application/json', ...(extraHeaders as Record<string, string>) },
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(body.detail || `${res.status} ${res.statusText}`)
  }
  return res.json()
}

export default function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [token, setToken] = useState<string | null>(() => localStorage.getItem(TOKEN_KEY))
  const [loading, setLoading] = useState(true)

  const saveTokens = (access: string, refresh: string) => {
    localStorage.setItem(TOKEN_KEY, access)
    localStorage.setItem(REFRESH_KEY, refresh)
    setToken(access)
  }

  const clearTokens = () => {
    localStorage.removeItem(TOKEN_KEY)
    localStorage.removeItem(REFRESH_KEY)
    setToken(null)
    setUser(null)
  }

  const fetchMe = useCallback(async (t: string) => {
    try {
      const u = await apiFetch<User>('/auth/me', {
        headers: { Authorization: `Bearer ${t}` },
      })
      setUser(u)
    } catch {
      clearTokens()
    }
  }, [])

  useEffect(() => {
    if (token) {
      fetchMe(token).finally(() => setLoading(false))
    } else {
      setLoading(false)
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const login = async (email: string, password: string) => {
    const res = await apiFetch<{ access_token: string; refresh_token: string }>('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    })
    saveTokens(res.access_token, res.refresh_token)
    await fetchMe(res.access_token)
  }

  const signup = async (email: string, password: string, fullName: string, accountName: string) => {
    const res = await apiFetch<{ access_token: string; refresh_token: string }>('/auth/register', {
      method: 'POST',
      body: JSON.stringify({ email, password, full_name: fullName, account_name: accountName }),
    })
    saveTokens(res.access_token, res.refresh_token)
    await fetchMe(res.access_token)
  }

  const logout = () => {
    clearTokens()
  }

  const refreshUser = async () => {
    if (token) await fetchMe(token)
  }

  return (
    <AuthContext.Provider value={{ user, token, loading, login, signup, logout, refreshUser }}>
      {children}
    </AuthContext.Provider>
  )
}
