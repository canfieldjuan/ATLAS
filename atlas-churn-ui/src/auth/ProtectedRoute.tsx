import { Navigate, useLocation } from 'react-router-dom'
import { useAuth } from './AuthContext'
import { buildCurrentRedirectTarget, buildLoginRedirectPath } from './redirects'

export default function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth()
  const location = useLocation()

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin h-8 w-8 border-2 border-cyan-400 border-t-transparent rounded-full" />
      </div>
    )
  }

  if (!user) {
    const redirectTo = buildCurrentRedirectTarget(location)
    return <Navigate to={buildLoginRedirectPath(redirectTo)} replace />
  }

  return <>{children}</>
}
