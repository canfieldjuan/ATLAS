import { useAuth } from '../auth/AuthContext'
import { CreditCard, AlertTriangle, LogOut } from 'lucide-react'

export default function Account() {
  const { user, logout } = useAuth()

  const isPastDue = user?.plan_status === 'past_due'
  const isTrialExpired = user?.plan_status === 'canceled' || (
    user?.trial_ends_at && new Date(user.trial_ends_at) < new Date()
  )

  return (
    <div className="max-w-xl mx-auto py-12 px-4">
      <h1 className="text-2xl font-bold text-white mb-6">Account</h1>

      <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-6 space-y-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-cyan-900/30 flex items-center justify-center">
            <span className="text-cyan-400 font-bold text-sm">
              {user?.full_name?.[0]?.toUpperCase() ?? user?.email?.[0]?.toUpperCase() ?? '?'}
            </span>
          </div>
          <div>
            <div className="text-white font-medium">{user?.full_name ?? 'User'}</div>
            <div className="text-sm text-slate-400">{user?.email}</div>
          </div>
        </div>

        <div className="border-t border-slate-700/50 pt-4 space-y-3">
          <div className="flex justify-between text-sm">
            <span className="text-slate-400">Company</span>
            <span className="text-white">{user?.account_name ?? '--'}</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-slate-400">Plan</span>
            <span className="text-white capitalize">{user?.plan?.replace(/_/g, ' ') ?? '--'}</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-slate-400">Status</span>
            <span className={
              isPastDue ? 'text-red-400 font-medium' :
              isTrialExpired ? 'text-amber-400 font-medium' :
              'text-green-400 font-medium'
            }>
              {isPastDue ? 'Past Due' : isTrialExpired ? 'Trial Expired' : user?.plan_status?.replace(/_/g, ' ') ?? '--'}
            </span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-slate-400">Vendor Limit</span>
            <span className="text-white">{user?.vendor_limit ?? '--'}</span>
          </div>
          {user?.trial_ends_at && (
            <div className="flex justify-between text-sm">
              <span className="text-slate-400">Trial Ends</span>
              <span className="text-white">{new Date(user.trial_ends_at).toLocaleDateString()}</span>
            </div>
          )}
        </div>

        {(isPastDue || isTrialExpired) && (
          <div className="flex items-start gap-2 text-sm text-amber-400 bg-amber-900/20 border border-amber-800/50 rounded-lg px-3 py-2">
            <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
            <div>
              {isPastDue
                ? 'Your payment is past due. Please update your billing information to continue using the platform.'
                : 'Your free trial has ended. Upgrade to a paid plan to continue accessing your churn intelligence.'}
            </div>
          </div>
        )}

        <div className="border-t border-slate-700/50 pt-4 flex items-center gap-3">
          <button
            disabled
            className="flex-1 inline-flex items-center justify-center gap-2 py-2.5 bg-cyan-600/50 rounded-lg text-white/50 font-medium cursor-not-allowed"
          >
            <CreditCard className="h-4 w-4" />
            Manage Billing (Coming Soon)
          </button>
          <button
            onClick={logout}
            className="inline-flex items-center gap-2 px-4 py-2.5 border border-slate-600 hover:border-slate-500 rounded-lg text-slate-300 text-sm transition-colors"
          >
            <LogOut className="h-4 w-4" />
            Sign out
          </button>
        </div>
      </div>
    </div>
  )
}
