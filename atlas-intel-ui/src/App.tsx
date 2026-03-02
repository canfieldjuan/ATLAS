import { Routes, Route } from 'react-router-dom'
import AuthProvider from './auth/AuthContext'
import ProtectedRoute from './auth/ProtectedRoute'
import Layout from './components/Layout'
import ErrorBoundary from './components/ErrorBoundary'
import Dashboard from './pages/Dashboard'
import Brands from './pages/Brands'
import BrandDetail from './pages/BrandDetail'
import BrandCompare from './pages/BrandCompare'
import Flows from './pages/Flows'
import Features from './pages/Features'
import Safety from './pages/Safety'
import Reviews from './pages/Reviews'
import ReviewDetail from './pages/ReviewDetail'
import Login from './pages/Login'
import Signup from './pages/Signup'
import Onboarding from './pages/Onboarding'
import Account from './pages/Account'

export default function App() {
  return (
    <AuthProvider>
      <ErrorBoundary>
        <Routes>
          {/* Public routes */}
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />

          {/* Protected routes */}
          <Route
            path="/*"
            element={
              <ProtectedRoute>
                <Layout>
                  <Routes>
                    <Route path="/" element={<Dashboard />} />
                    <Route path="/brands" element={<Brands />} />
                    <Route path="/brands/:name" element={<BrandDetail />} />
                    <Route path="/compare" element={<BrandCompare />} />
                    <Route path="/flows" element={<Flows />} />
                    <Route path="/features" element={<Features />} />
                    <Route path="/safety" element={<Safety />} />
                    <Route path="/reviews" element={<Reviews />} />
                    <Route path="/reviews/:id" element={<ReviewDetail />} />
                    <Route path="/onboarding" element={<Onboarding />} />
                    <Route path="/account" element={<Account />} />
                  </Routes>
                </Layout>
              </ProtectedRoute>
            }
          />
        </Routes>
      </ErrorBoundary>
    </AuthProvider>
  )
}
