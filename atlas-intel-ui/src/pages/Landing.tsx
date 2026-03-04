import React from 'react'
import { Link } from 'react-router-dom'
import { AlertTriangle, ArrowRightLeft, Shield, Mail, Check } from 'lucide-react'
import PublicLayout from '../components/PublicLayout'

const AtlasHeroScene = React.lazy(() => import('../components/AtlasHeroScene'))

const FEATURES = [
  {
    icon: AlertTriangle,
    title: 'Complaint Tracking',
    desc: 'Surface recurring product complaints before they become returns. Spot defect patterns across your catalog.',
  },
  {
    icon: ArrowRightLeft,
    title: 'Competitive Flows',
    desc: 'See where customers switch and why. Track competitor mentions and migration patterns in real time.',
  },
  {
    icon: Shield,
    title: 'Safety Signals',
    desc: 'Detect safety-related reviews early. Get alerts when injury, recall, or hazard language appears.',
  },
  {
    icon: Mail,
    title: 'Weekly Digest',
    desc: 'Automated email reports with new signals, trend shifts, and actionable insights for your tracked ASINs.',
  },
]

const PLANS = [
  {
    name: 'Starter',
    price: 49,
    features: ['5 tracked ASINs', 'Complaint analytics', 'Safety signals', 'Weekly email digest'],
    cta: 'Start Free Trial',
    href: '/signup?product=consumer',
    highlight: false,
  },
  {
    name: 'Growth',
    price: 149,
    features: ['25 tracked ASINs', 'Everything in Starter', 'Brand comparison', 'API access'],
    cta: 'Start Free Trial',
    href: '/signup?product=consumer',
    highlight: true,
  },
  {
    name: 'Pro',
    price: 399,
    features: ['100 tracked ASINs', 'Everything in Growth', 'Priority support', 'Custom alerts'],
    cta: 'Start Free Trial',
    href: '/signup?product=consumer',
    highlight: false,
  },
]

export default function Landing() {
  return (
    <PublicLayout>
      {/* Hero */}
      <section className="mx-auto px-6 pt-8 pb-24 text-center">
        <React.Suspense fallback={<div className="h-[300px]" />}>
          <AtlasHeroScene title="ATLAS" tagline="INTELLIGENCE PLATFORM" />
        </React.Suspense>
        <h1 className="mt-8 text-4xl sm:text-5xl font-bold leading-tight">
          Amazon review intelligence
          <br />
          <span className="text-cyan-400">before your competitors see it</span>
        </h1>
        <p className="mt-6 text-lg text-slate-400 max-w-2xl mx-auto">
          Track competitor complaints, safety signals, and customer migration patterns across your product category. Automated alerts so you never miss a trend.
        </p>
        <div className="mt-10 flex items-center justify-center gap-4">
          <Link
            to="/signup?product=consumer"
            className="px-6 py-3 bg-cyan-600 hover:bg-cyan-500 rounded-lg text-white font-semibold transition-colors"
          >
            Start Free Trial
          </Link>
          <a href="#pricing" className="px-6 py-3 border border-slate-600 hover:border-slate-500 rounded-lg text-slate-300 font-medium transition-colors">
            View Pricing
          </a>
        </div>
        <p className="mt-4 text-sm text-slate-500">14-day free trial. No credit card required.</p>
      </section>

      {/* Features */}
      <section className="max-w-6xl mx-auto px-6 pb-24">
        <h2 className="text-2xl font-bold text-center mb-12">Everything you need to monitor your market</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {FEATURES.map(f => {
            const Icon = f.icon
            return (
              <div
                key={f.title}
                className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-6"
              >
                <Icon className="h-8 w-8 text-cyan-400 mb-4" />
                <h3 className="text-lg font-semibold mb-2">{f.title}</h3>
                <p className="text-sm text-slate-400 leading-relaxed">{f.desc}</p>
              </div>
            )
          })}
        </div>
      </section>

      {/* Pricing */}
      <section id="pricing" className="max-w-5xl mx-auto px-6 pb-24">
        <h2 className="text-2xl font-bold text-center mb-4">Simple, transparent pricing</h2>
        <p className="text-center text-slate-400 mb-12">Start free, upgrade when you need more ASINs or features.</p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {PLANS.map(plan => (
            <div
              key={plan.name}
              className={`bg-slate-800/60 border rounded-xl p-6 flex flex-col ${
                plan.highlight ? 'border-cyan-500/50 ring-1 ring-cyan-500/20' : 'border-slate-700/50'
              }`}
            >
              {plan.highlight && (
                <span className="text-xs font-semibold text-cyan-400 uppercase tracking-wide mb-2">Most Popular</span>
              )}
              <h3 className="text-xl font-bold">{plan.name}</h3>
              <div className="mt-3 mb-6">
                <span className="text-4xl font-bold">${plan.price}</span>
                <span className="text-slate-400 text-sm">/mo</span>
              </div>
              <ul className="space-y-3 mb-8 flex-1">
                {plan.features.map(feat => (
                  <li key={feat} className="flex items-start gap-2 text-sm text-slate-300">
                    <Check className="h-4 w-4 text-cyan-400 mt-0.5 shrink-0" />
                    {feat}
                  </li>
                ))}
              </ul>
              <Link
                to={plan.href}
                className={`block text-center py-2.5 rounded-lg font-medium transition-colors ${
                  plan.highlight
                    ? 'bg-cyan-600 hover:bg-cyan-500 text-white'
                    : 'bg-slate-700 hover:bg-slate-600 text-slate-200'
                }`}
              >
                {plan.cta}
              </Link>
            </div>
          ))}
        </div>
      </section>
    </PublicLayout>
  )
}
