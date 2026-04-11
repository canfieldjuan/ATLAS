const DEFAULT_REDIRECT_TARGET = '/watchlists'
const DEFAULT_PRODUCT = 'b2b_retention'
const CHALLENGER_PRODUCT = 'b2b_challenger'
const DISALLOWED_REDIRECT_PREFIXES = [
  '/landing',
  '/login',
  '/signup',
  '/forgot-password',
  '/reset-password',
]
const CHALLENGER_PATH_PREFIXES = [
  '/challengers',
  '/vendor-targets',
  '/prospects',
]

function isDisallowedRedirect(target: string): boolean {
  const path = target.split(/[?#]/, 1)[0]
  return DISALLOWED_REDIRECT_PREFIXES.some(prefix => path === prefix || path.startsWith(`${prefix}/`))
}

export function inferRedirectProduct(target?: string | null): string {
  const redirectTarget = normalizeRedirectTarget(target)
  const path = redirectTarget.split(/[?#]/, 1)[0]
  if (CHALLENGER_PATH_PREFIXES.some(prefix => path === prefix || path.startsWith(`${prefix}/`))) {
    return CHALLENGER_PRODUCT
  }
  return DEFAULT_PRODUCT
}

export function normalizeRedirectTarget(target?: string | null): string {
  const trimmed = target?.trim() || ''
  if (!trimmed || trimmed === '/') return DEFAULT_REDIRECT_TARGET
  if (!trimmed.startsWith('/') || trimmed.startsWith('//')) return DEFAULT_REDIRECT_TARGET
  if (isDisallowedRedirect(trimmed)) return DEFAULT_REDIRECT_TARGET
  return trimmed
}

export function buildLoginRedirectPath(target?: string | null, product?: string | null): string {
  const redirectTarget = normalizeRedirectTarget(target)
  const params = new URLSearchParams()
  params.set('redirect_to', redirectTarget)
  params.set('product', product?.trim() || inferRedirectProduct(redirectTarget))
  return `/login?${params.toString()}`
}

export function buildSignupRedirectPath(target?: string | null, product?: string | null): string {
  const redirectTarget = normalizeRedirectTarget(target)
  const resolvedProduct = product?.trim() || inferRedirectProduct(redirectTarget)
  const params = new URLSearchParams()
  params.set('product', resolvedProduct)
  params.set('redirect_to', redirectTarget)
  return `/signup?${params.toString()}`
}

export function buildCurrentRedirectTarget(locationLike: Pick<Location, 'pathname' | 'search' | 'hash'>): string {
  return normalizeRedirectTarget(`${locationLike.pathname}${locationLike.search}${locationLike.hash}`)
}
