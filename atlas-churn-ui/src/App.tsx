import { lazy, Suspense, type ComponentType } from 'react'
import { Routes, Route, Navigate, useLocation } from 'react-router-dom'
import Layout from './components/Layout'
import ErrorBoundary from './components/ErrorBoundary'
import ProtectedRoute from './auth/ProtectedRoute'

const Dashboard = lazy(() => import('./pages/Dashboard'))
const Vendors = lazy(() => import('./pages/Vendors'))
const VendorDetail = lazy(() => import('./pages/VendorDetail'))
const Reviews = lazy(() => import('./pages/Reviews'))
const ReviewDetail = lazy(() => import('./pages/ReviewDetail'))
const Reports = lazy(() => import('./pages/Reports'))
const ReportDetail = lazy(() => import('./pages/ReportDetail'))
const Leads = lazy(() => import('./pages/Leads'))
const VendorTargets = lazy(() => import('./pages/VendorTargets'))
const Challengers = lazy(() => import('./pages/Challengers'))
const Affiliates = lazy(() => import('./pages/Affiliates'))
const Landing = lazy(() => import('./pages/Landing'))
const Login = lazy(() => import('./pages/Login'))
const Signup = lazy(() => import('./pages/Signup'))
const ForgotPassword = lazy(() => import('./pages/ForgotPassword'))
const ResetPassword = lazy(() => import('./pages/ResetPassword'))
const Onboarding = lazy(() => import('./pages/Onboarding'))
const Account = lazy(() => import('./pages/Account'))
const Blog = lazy(() => import('./pages/Blog'))
const BlogPost = lazy(() => import('./pages/BlogPost'))
const Methodology = lazy(() => import('./pages/Methodology'))
const BlogReview = lazy(() => import('./pages/BlogReview'))
const CampaignReview = lazy(() => import('./pages/CampaignReview'))
const BriefingReview = lazy(() => import('./pages/BriefingReview'))
const Prospects = lazy(() => import('./pages/Prospects'))
const Report = lazy(() => import('./pages/Report'))

const PUBLIC_PATHS = ['/blog', '/landing', '/login', '/signup', '/forgot-password', '/reset-password', '/methodology', '/report']

function isPublicPath(pathname: string): boolean {
  return PUBLIC_PATHS.some(p => pathname === p || pathname.startsWith(p + '/'))
}

function renderLazyRoute(Component: ComponentType) {
  return (
    <Suspense fallback={null}>
      <Component />
    </Suspense>
  )
}

export default function App() {
  const location = useLocation()
  const isPublicRoute = isPublicPath(location.pathname)

  if (isPublicRoute) {
    return (
      <ErrorBoundary key={location.pathname}>
        <Routes>
          <Route path="/landing" element={renderLazyRoute(Landing)} />
          <Route path="/blog" element={renderLazyRoute(Blog)} />
          <Route path="/blog/:slug" element={renderLazyRoute(BlogPost)} />
          <Route path="/methodology" element={renderLazyRoute(Methodology)} />
          <Route path="/login" element={renderLazyRoute(Login)} />
          <Route path="/signup" element={renderLazyRoute(Signup)} />
          <Route path="/forgot-password" element={renderLazyRoute(ForgotPassword)} />
          <Route path="/reset-password" element={renderLazyRoute(ResetPassword)} />
          <Route path="/report" element={renderLazyRoute(Report)} />
        </Routes>
      </ErrorBoundary>
    )
  }

  // Onboarding is protected but doesn't use the sidebar layout
  if (location.pathname === '/onboarding') {
    return (
      <ProtectedRoute>
        <ErrorBoundary key={location.pathname}>
          {renderLazyRoute(Onboarding)}
        </ErrorBoundary>
      </ProtectedRoute>
    )
  }

  return (
    <ProtectedRoute>
      <Layout>
        <ErrorBoundary key={location.pathname}>
          <Routes>
            <Route path="/" element={renderLazyRoute(Dashboard)} />
            <Route path="/vendors" element={renderLazyRoute(Vendors)} />
            <Route path="/vendors/:name" element={renderLazyRoute(VendorDetail)} />
            <Route path="/reviews" element={renderLazyRoute(Reviews)} />
            <Route path="/reviews/:id" element={renderLazyRoute(ReviewDetail)} />
            <Route path="/reports" element={renderLazyRoute(Reports)} />
            <Route path="/reports/:id" element={renderLazyRoute(ReportDetail)} />
            <Route path="/leads" element={renderLazyRoute(Leads)} />
            <Route path="/vendor-targets" element={renderLazyRoute(VendorTargets)} />
            <Route path="/challengers" element={renderLazyRoute(Challengers)} />
            <Route path="/affiliates" element={renderLazyRoute(Affiliates)} />
            <Route path="/blog-review" element={renderLazyRoute(BlogReview)} />
            <Route path="/campaign-review" element={renderLazyRoute(CampaignReview)} />
            <Route path="/briefing-review" element={renderLazyRoute(BriefingReview)} />
            <Route path="/prospects" element={renderLazyRoute(Prospects)} />
            <Route path="/account" element={renderLazyRoute(Account)} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </ErrorBoundary>
      </Layout>
    </ProtectedRoute>
  )
}
