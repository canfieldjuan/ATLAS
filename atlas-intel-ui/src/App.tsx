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

// B2B pages
import B2BDashboard from './pages/b2b/B2BDashboard'
import B2BOnboarding from './pages/b2b/B2BOnboarding'
import ChurnSignals from './pages/b2b/ChurnSignals'
import VendorDetail from './pages/b2b/VendorDetail'
import LeadPipeline from './pages/b2b/LeadPipeline'
import LeadDetail from './pages/b2b/LeadDetail'
import CompetitorDisplacement from './pages/b2b/CompetitorDisplacement'
import B2BReports from './pages/b2b/B2BReports'
import B2BReviews from './pages/b2b/B2BReviews'
import B2BCampaigns from './pages/b2b/B2BCampaigns'

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
                    {/* Consumer routes */}
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

                    {/* B2B routes */}
                    <Route path="/b2b" element={<B2BDashboard />} />
                    <Route path="/b2b/onboarding" element={<B2BOnboarding />} />
                    <Route path="/b2b/signals" element={<ChurnSignals />} />
                    <Route path="/b2b/signals/:vendorName" element={<VendorDetail />} />
                    <Route path="/b2b/leads" element={<LeadPipeline />} />
                    <Route path="/b2b/leads/:company" element={<LeadDetail />} />
                    <Route path="/b2b/displacement" element={<CompetitorDisplacement />} />
                    <Route path="/b2b/reports" element={<B2BReports />} />
                    <Route path="/b2b/reviews" element={<B2BReviews />} />
                    <Route path="/b2b/campaigns" element={<B2BCampaigns />} />
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
