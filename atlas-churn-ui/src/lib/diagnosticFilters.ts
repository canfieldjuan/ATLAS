export const DIAGNOSTIC_DAY_OPTIONS = [7, 14, 30, 60, 90] as const
export const DIAGNOSTIC_TOP_N_OPTIONS = [5, 10, 20] as const

export const DEFAULT_DIAGNOSTIC_DAYS = 14
export const DEFAULT_DIAGNOSTICS_TOP_N = 10
export const DEFAULT_TRENDS_TOP_N = 5

export function coerceDiagnosticParam(
  raw: string | null,
  allowed: readonly number[],
  fallback: number,
): number {
  const parsed = Number(raw)
  if (!Number.isFinite(parsed)) {
    return fallback
  }
  return allowed.some((value) => value === parsed) ? parsed : fallback
}
