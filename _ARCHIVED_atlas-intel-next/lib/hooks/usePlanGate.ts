import { useAuth } from '@/lib/auth/AuthContext'

const B2B_VENDOR_LIMITS: Record<string, number> = {
  b2b_trial: 1,
  b2b_starter: 5,
  b2b_growth: 25,
  b2b_pro: -1, // unlimited
}

export function usePlanGate() {
  const { user } = useAuth()
  const plan = user?.plan ?? 'b2b_trial'

  return {
    canAccessCampaigns: ['b2b_growth', 'b2b_pro'].includes(plan),
    canAccessReports: ['b2b_starter', 'b2b_growth', 'b2b_pro'].includes(plan),
    canAccessApi: plan === 'b2b_pro',
    vendorLimit: B2B_VENDOR_LIMITS[plan] ?? 1,
    plan,
    isTrial: plan === 'b2b_trial',
  }
}
