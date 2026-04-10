import { useState } from 'react'
import { Download, Bell, BellOff, Check, Copy } from 'lucide-react'
import { clsx } from 'clsx'
import { downloadReportPdf } from '../api/client'

interface ReportActionBarProps {
  reportId: string
  onSubscribe: () => void
  hasSubscription?: boolean
  subscriptionState?: 'none' | 'active' | 'paused'
  hasPdfExport?: boolean
  artifactState?: 'ready' | 'processing' | 'failed' | 'unknown' | null
  artifactLabel?: string | null
  linkUrl?: string
}

function getPdfButtonPresentation(
  hasPdfExport: boolean,
  artifactState?: 'ready' | 'processing' | 'failed' | 'unknown' | null,
  artifactLabel?: string | null,
) {
  if (hasPdfExport) {
    return {
      label: 'PDF',
      title: 'Download PDF',
    }
  }

  if (artifactState === 'processing') {
    return {
      label: 'Generating',
      title: `PDF unavailable while report is ${artifactLabel?.toLowerCase() || 'processing'}`,
    }
  }

  if (artifactState === 'failed') {
    return {
      label: 'PDF Failed',
      title: `PDF unavailable while report is ${artifactLabel?.toLowerCase() || 'failed'}`,
    }
  }

  return {
    label: 'Unavailable',
    title: 'PDF unavailable until the report is ready',
  }
}

function getSubscriptionButtonPresentation(
  subscriptionState?: 'none' | 'active' | 'paused',
  hasSubscription?: boolean,
) {
  const resolvedState = subscriptionState ?? (hasSubscription ? 'active' : 'none')

  if (resolvedState === 'active') {
    return {
      label: 'Manage Subscription',
      title: 'Manage or pause subscription',
      className: 'bg-cyan-900/30 border-cyan-700/50 text-cyan-300 hover:bg-cyan-900/50',
      icon: BellOff,
    }
  }

  if (resolvedState === 'paused') {
    return {
      label: 'Resume Subscription',
      title: 'Resume or edit subscription',
      className: 'bg-amber-500/10 border-amber-500/25 text-amber-300 hover:bg-amber-500/20',
      icon: Bell,
    }
  }

  return {
    label: 'Subscribe',
    title: 'Subscribe to updates',
    className: 'bg-slate-800 border-slate-700 text-slate-300 hover:text-white hover:bg-slate-700',
    icon: Bell,
  }
}

export default function ReportActionBar({
  reportId, onSubscribe, hasSubscription, subscriptionState, hasPdfExport = true, artifactState, artifactLabel, linkUrl,
}: ReportActionBarProps) {
  const [copied, setCopied] = useState(false)
  const pdfButton = getPdfButtonPresentation(hasPdfExport, artifactState, artifactLabel)
  const subscriptionButton = getSubscriptionButtonPresentation(subscriptionState, hasSubscription)
  const SubscriptionIcon = subscriptionButton.icon

  const handleCopyLink = () => {
    const target = linkUrl || `/reports/${reportId}`
    const url = target.startsWith('http')
      ? target
      : `${window.location.origin}${target}`
    navigator.clipboard.writeText(url).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  return (
    <div className="flex items-center gap-2">
      {/* PDF Export */}
      <button
        onClick={() => {
          if (hasPdfExport) downloadReportPdf(reportId)
        }}
        disabled={!hasPdfExport}
        className={clsx(
          'inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border transition-colors',
          hasPdfExport
            ? 'bg-slate-800 border-slate-700 text-slate-300 hover:text-white hover:bg-slate-700'
            : 'bg-slate-900 border-slate-800 text-slate-500 cursor-not-allowed',
        )}
        title={pdfButton.title}
      >
        <Download className="w-3.5 h-3.5" />
        {pdfButton.label}
      </button>

      {/* Copy Link */}
      <button
        onClick={handleCopyLink}
        className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-slate-800 border border-slate-700 text-slate-300 hover:text-white hover:bg-slate-700 transition-colors"
        title="Copy link"
      >
        {copied ? <Check className="w-3.5 h-3.5 text-green-400" /> : <Copy className="w-3.5 h-3.5" />}
        {copied ? 'Copied' : 'Link'}
      </button>

      {/* Subscribe */}
      <button
        onClick={onSubscribe}
        className={clsx(
          'inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border transition-colors',
          subscriptionButton.className,
        )}
        title={subscriptionButton.title}
      >
        <SubscriptionIcon className="w-3.5 h-3.5" />
        {subscriptionButton.label}
      </button>
    </div>
  )
}
