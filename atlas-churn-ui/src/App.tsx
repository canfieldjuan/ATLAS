import { Routes, Route, useLocation } from 'react-router-dom'
import Layout from './components/Layout'
import ErrorBoundary from './components/ErrorBoundary'
import Dashboard from './pages/Dashboard'
import Vendors from './pages/Vendors'
import VendorDetail from './pages/VendorDetail'
import Reviews from './pages/Reviews'
import ReviewDetail from './pages/ReviewDetail'
import Reports from './pages/Reports'
import ReportDetail from './pages/ReportDetail'

export default function App() {
  const location = useLocation()

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
        </Routes>
      </ErrorBoundary>
    </Layout>
  )
}
