import { Routes, Route, Navigate, useLocation } from 'react-router-dom'
import Layout from './components/Layout'
import ErrorBoundary from './components/ErrorBoundary'
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
import Blog from './pages/Blog'
import BlogPost from './pages/BlogPost'

const BLOG_PATHS = ['/blog']

export default function App() {
  const location = useLocation()
  const isBlogRoute = BLOG_PATHS.some(p => location.pathname.startsWith(p))

  if (isBlogRoute) {
    return (
      <ErrorBoundary key={location.pathname}>
        <Routes>
          <Route path="/blog" element={<Blog />} />
          <Route path="/blog/:slug" element={<BlogPost />} />
        </Routes>
      </ErrorBoundary>
    )
  }

  return (
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
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </ErrorBoundary>
    </Layout>
  )
}
