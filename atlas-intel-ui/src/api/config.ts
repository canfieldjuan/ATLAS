/**
 * API base URL.
 *
 * - Development: empty string (Vite proxy handles /api -> localhost:8000)
 * - Production:  set VITE_API_BASE to the backend origin, e.g.
 *                "https://atlas-brain.example.com"
 */
export const API_BASE = import.meta.env.VITE_API_BASE || ''
