import { lazy, Suspense, type ComponentType } from 'react'
import { Routes, Route } from 'react-router-dom'
import AuthProvider from './auth/AuthContext'
import ProtectedRoute from './auth/ProtectedRoute'
import Layout from './components/Layout'
import ErrorBoundary from './components/ErrorBoundary'

const Dashboard = lazy(() => import('./pages/Dashboard'))
const Brands = lazy(() => import('./pages/Brands'))
const BrandDetail = lazy(() => import('./pages/BrandDetail'))
const BrandCompare = lazy(() => import('./pages/BrandCompare'))
const Flows = lazy(() => import('./pages/Flows'))
const Features = lazy(() => import('./pages/Features'))
const Safety = lazy(() => import('./pages/Safety'))
const Reviews = lazy(() => import('./pages/Reviews'))
const ReviewDetail = lazy(() => import('./pages/ReviewDetail'))
const Landing = lazy(() => import('./pages/Landing'))
const Blog = lazy(() => import('./pages/Blog'))
const BlogPost = lazy(() => import('./pages/BlogPost'))
const Login = lazy(() => import('./pages/Login'))
const Signup = lazy(() => import('./pages/Signup'))
const Onboarding = lazy(() => import('./pages/Onboarding'))
const Account = lazy(() => import('./pages/Account'))
const B2BDashboard = lazy(() => import('./pages/b2b/B2BDashboard'))
const B2BOnboarding = lazy(() => import('./pages/b2b/B2BOnboarding'))
const ChurnSignals = lazy(() => import('./pages/b2b/ChurnSignals'))
const VendorDetail = lazy(() => import('./pages/b2b/VendorDetail'))
const LeadPipeline = lazy(() => import('./pages/b2b/LeadPipeline'))
const LeadDetail = lazy(() => import('./pages/b2b/LeadDetail'))
const CompetitorDisplacement = lazy(() => import('./pages/b2b/CompetitorDisplacement'))
const B2BReports = lazy(() => import('./pages/b2b/B2BReports'))
const B2BReviews = lazy(() => import('./pages/b2b/B2BReviews'))
const B2BCampaigns = lazy(() => import('./pages/b2b/B2BCampaigns'))

function renderLazyRoute(Component: ComponentType) {
  return (
    <Suspense fallback={null}>
      <Component />
    </Suspense>
  )
}

export default function App() {
  return (
    <AuthProvider>
      <ErrorBoundary>
        <Routes>
          {/* Public routes */}
          <Route path="/landing" element={renderLazyRoute(Landing)} />
          <Route path="/blog" element={renderLazyRoute(Blog)} />
          <Route path="/blog/:slug" element={renderLazyRoute(BlogPost)} />
          <Route path="/login" element={renderLazyRoute(Login)} />
          <Route path="/signup" element={renderLazyRoute(Signup)} />

          {/* Protected routes */}
          <Route
            path="/*"
            element={
              <ProtectedRoute>
                <Layout>
                  <Routes>
                    {/* Consumer routes */}
                    <Route path="/" element={renderLazyRoute(Dashboard)} />
                    <Route path="/brands" element={renderLazyRoute(Brands)} />
                    <Route path="/brands/:name" element={renderLazyRoute(BrandDetail)} />
                    <Route path="/compare" element={renderLazyRoute(BrandCompare)} />
                    <Route path="/flows" element={renderLazyRoute(Flows)} />
                    <Route path="/features" element={renderLazyRoute(Features)} />
                    <Route path="/safety" element={renderLazyRoute(Safety)} />
                    <Route path="/reviews" element={renderLazyRoute(Reviews)} />
                    <Route path="/reviews/:id" element={renderLazyRoute(ReviewDetail)} />
                    <Route path="/onboarding" element={renderLazyRoute(Onboarding)} />
                    <Route path="/account" element={renderLazyRoute(Account)} />

                    {/* B2B routes */}
                    <Route path="/b2b" element={renderLazyRoute(B2BDashboard)} />
                    <Route path="/b2b/onboarding" element={renderLazyRoute(B2BOnboarding)} />
                    <Route path="/b2b/signals" element={renderLazyRoute(ChurnSignals)} />
                    <Route path="/b2b/signals/:vendorName" element={renderLazyRoute(VendorDetail)} />
                    <Route path="/b2b/leads" element={renderLazyRoute(LeadPipeline)} />
                    <Route path="/b2b/leads/:company" element={renderLazyRoute(LeadDetail)} />
                    <Route path="/b2b/displacement" element={renderLazyRoute(CompetitorDisplacement)} />
                    <Route path="/b2b/reports" element={renderLazyRoute(B2BReports)} />
                    <Route path="/b2b/reviews" element={renderLazyRoute(B2BReviews)} />
                    <Route path="/b2b/campaigns" element={renderLazyRoute(B2BCampaigns)} />
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
