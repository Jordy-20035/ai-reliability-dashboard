import axios from 'axios'

/** Human-readable message for failed API calls (axios or unknown). */
export function getErrorMessage(err: unknown): string {
  if (axios.isAxiosError(err)) {
    const detail = err.response?.data
    if (detail && typeof detail === 'object' && 'detail' in detail) {
      const d = (detail as { detail: unknown }).detail
      if (typeof d === 'string') return d
      if (Array.isArray(d)) return JSON.stringify(d)
    }
    return err.message || String(err.response?.status ?? 'request failed')
  }
  if (err instanceof Error) return err.message
  return String(err)
}
