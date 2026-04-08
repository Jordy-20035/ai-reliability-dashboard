import axios from 'axios'

/**
 * In dev, default to same-origin + Vite proxy (see vite.config.ts) so CORS is not an issue.
 * In production build, default to explicit API host unless VITE_API_BASE_URL is set.
 */
const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ??
  (import.meta.env.DEV ? '' : 'http://127.0.0.1:8000')

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000,
  headers: {
    'Content-Type': 'application/json',
  },
})

