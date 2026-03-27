/**
 * API base URL.
 *
 * - Development: empty string (Vite proxy handles /api -> localhost:8000)
 * - Production:  set VITE_API_BASE to the backend origin, e.g.
 *                "https://atlas-brain.example.com"
 */
export const API_BASE = process.env.NEXT_PUBLIC_API_BASE || ''
