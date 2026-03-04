import React, { useEffect } from 'react'
import { Link } from 'react-router-dom'
import { AlertTriangle, TrendingDown, Users, BarChart3, Zap, Bell, Check } from 'lucide-react'
import PublicLayout from '../components/PublicLayout'

const AtlasHeroScene = React.lazy(() => import('../components/AtlasHeroScene'))

const FEATURES = [
  {
    icon: TrendingDown,
    title: 'Churn Signal Detection',
    desc: 'Surface vendor churn signals from real enterprise reviews. Know which accounts are at risk before they leave.',
  },
  {
    icon: AlertTriangle,
    title: 'Competitive Displacement',
    desc: 'Track which vendors are losing customers and where those customers are going. Identify displacement patterns in real time.',
  },
  {
    icon: Users,
    title: 'High-Intent Leads',
    desc: 'Find companies actively evaluating alternatives to their current vendor. Scored by urgency, pain, and authority signals.',
  },
  {
    icon: Bell,
    title: 'Real-Time Alerts',
    desc: 'Get notified when high-urgency signals appear for vendors or categories you track. Never miss a window.',
  },
]

const STATS = [
  { value: '15+', label: 'Review sources tracked' },
  { value: '500+', label: 'SaaS vendors monitored' },
  { value: '50K+', label: 'Reviews analyzed' },
  { value: '24h', label: 'Signal freshness' },
]

const PLANS = [
  {
    name: 'Starter',
    price: 299,
    features: [
      '5 tracked vendors',
      'Churn signal feed',
      'Weekly intelligence digest',
      'CSV export',
    ],
    cta: 'Start Free Trial',
    href: '/signup',
    highlight: false,
  },
  {
    name: 'Growth',
    price: 999,
    features: [
      '25 tracked vendors',
      'Everything in Starter',
      'High-intent lead scoring',
      'Competitive displacement maps',
      'API access',
    ],
    cta: 'Start Free Trial',
    href: '/signup',
    highlight: true,
  },
  {
    name: 'Enterprise',
    price: 2499,
    features: [
      'Unlimited vendors',
      'Everything in Growth',
      'CRM integration',
      'Custom alert rules',
      'Dedicated support',
    ],
    cta: 'Contact Sales',
    href: '/signup',
    highlight: false,
  },
]

const HOW_IT_WORKS = [
  {
    icon: BarChart3,
    step: '1',
    title: 'We monitor review sites',
    desc: 'Our pipeline scans 15+ B2B review sources daily -- G2, Capterra, TrustRadius, Reddit, and more.',
  },
  {
    icon: Zap,
    step: '2',
    title: 'AI extracts churn signals',
    desc: 'Each review is enriched with urgency scoring, pain classification, competitor mentions, and migration intent.',
  },
  {
    icon: TrendingDown,
    step: '3',
    title: 'You act on intelligence',
    desc: 'Get scored leads, vendor risk reports, and real-time alerts. Close deals while competitors are still reading reviews.',
  },
]

export default function Landing() {
  useEffect(() => { document.title = 'Churn Signals -- B2B Churn Intelligence' }, [])

  return (
    <PublicLayout>
      {/* Hero */}
      <section className="mx-auto px-6 pt-8 pb-20 text-center">
        <React.Suspense fallback={<div className="h-[300px]" />}>
          <AtlasHeroScene title="CHURN" tagline="SIGNALS INTELLIGENCE" />
        </React.Suspense>
        <h1 className="mt-8 text-4xl sm:text-5xl font-bold leading-tight">
          Know which vendors are
          <br />
          <span className="text-cyan-400">losing customers before they do</span>
        </h1>
        <p className="mt-6 text-lg text-slate-400 max-w-2xl mx-auto">
          Real-time churn intelligence from 15+ B2B review sources. Identify at-risk accounts, score high-intent leads, and close displacement deals faster.
        </p>
        <div className="mt-10 flex items-center justify-center gap-4">
          <Link
            to="/signup"
            className="px-6 py-3 bg-cyan-600 hover:bg-cyan-500 rounded-lg text-white font-semibold transition-colors"
          >
            Start Free Trial
          </Link>
          <a
            href="#pricing"
            className="px-6 py-3 border border-slate-600 hover:border-slate-500 rounded-lg text-slate-300 font-medium transition-colors"
          >
            View Pricing
          </a>
        </div>
        <p className="mt-4 text-sm text-slate-500">14-day free trial. No credit card required.</p>
      </section>

      {/* Social proof stats */}
      <section className="max-w-4xl mx-auto px-6 pb-20">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          {STATS.map(s => (
            <div key={s.label} className="text-center">
              <div className="text-3xl font-bold text-cyan-400">{s.value}</div>
              <div className="mt-1 text-sm text-slate-400">{s.label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Features */}
      <section className="max-w-6xl mx-auto px-6 pb-24">
        <h2 className="text-2xl font-bold text-center mb-12">Intelligence your competitors don't have</h2>
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

      {/* How it works */}
      <section className="max-w-4xl mx-auto px-6 pb-24">
        <h2 className="text-2xl font-bold text-center mb-12">How it works</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {HOW_IT_WORKS.map(step => {
            const Icon = step.icon
            return (
              <div key={step.step} className="text-center">
                <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-cyan-500/10 border border-cyan-500/20 mb-4">
                  <Icon className="h-6 w-6 text-cyan-400" />
                </div>
                <h3 className="text-lg font-semibold mb-2">{step.title}</h3>
                <p className="text-sm text-slate-400 leading-relaxed">{step.desc}</p>
              </div>
            )
          })}
        </div>
      </section>

      {/* Pricing */}
      <section id="pricing" className="max-w-5xl mx-auto px-6 pb-24">
        <h2 className="text-2xl font-bold text-center mb-4">Simple, transparent pricing</h2>
        <p className="text-center text-slate-400 mb-12">Start free, upgrade when you need more vendors or features.</p>
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
