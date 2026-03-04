import { Routes, Route, Navigate, useLocation } from 'react-router-dom'
import Layout from './components/Layout'
import ErrorBoundary from './components/ErrorBoundary'
import ProtectedRoute from './auth/ProtectedRoute'
import Dashboard from './pages/Dashboard'
import Vendors from './pages/Vendors'
import VendorDetail from './pages/VendorDetail'
import Reviews from './pages/Reviews'
import ReviewDetail from './pages/ReviewDetail'
import Reports from './pages/Reports'
import ReportDetail from './pages/ReportDetail'
import Leads from './pages/Leads'
import VendorTargets from './pages/VendorTargets'
import Challengers from './pages/Challengers'
import Affiliates from './pages/Affiliates'
import Landing from './pages/Landing'
import Login from './pages/Login'
import Signup from './pages/Signup'
import Onboarding from './pages/Onboarding'
import Account from './pages/Account'
import Blog from './pages/Blog'
import BlogPost from './pages/BlogPost'

const PUBLIC_PATHS = ['/blog', '/landing', '/login', '/signup']

export default function App() {
  const location = useLocation()
  const isPublicRoute = PUBLIC_PATHS.some(p => location.pathname.startsWith(p))

  if (isPublicRoute) {
    return (
      <ErrorBoundary key={location.pathname}>
        <Routes>
          <Route path="/landing" element={<Landing />} />
          <Route path="/blog" element={<Blog />} />
          <Route path="/blog/:slug" element={<BlogPost />} />
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
        </Routes>
      </ErrorBoundary>
    )
  }

  // Onboarding is protected but doesn't use the sidebar layout
  if (location.pathname === '/onboarding') {
    return (
      <ProtectedRoute>
        <ErrorBoundary key={location.pathname}>
          <Onboarding />
        </ErrorBoundary>
      </ProtectedRoute>
    )
  }

  return (
    <ProtectedRoute>
      <Layout>
        <ErrorBoundary key={location.pathname}>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/vendors" element={<Vendors />} />
            <Route path="/vendors/:name" element={<VendorDetail />} />
            <Route path="/reviews" element={<Reviews />} />
            <Route path="/reviews/:id" element={<ReviewDetail />} />
            <Route path="/reports" element={<Reports />} />
            <Route path="/reports/:id" element={<ReportDetail />} />
            <Route path="/leads" element={<Leads />} />
            <Route path="/vendor-targets" element={<VendorTargets />} />
            <Route path="/challengers" element={<Challengers />} />
            <Route path="/affiliates" element={<Affiliates />} />
            <Route path="/account" element={<Account />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </ErrorBoundary>
      </Layout>
    </ProtectedRoute>
  )
}
