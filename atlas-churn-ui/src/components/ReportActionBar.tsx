import { useState } from 'react'
import { Download, Bell, BellOff, Check, Copy } from 'lucide-react'
import { clsx } from 'clsx'
import { downloadReportPdf } from '../api/client'

interface ReportActionBarProps {
  reportId: string
  onSubscribe: () => void
  hasSubscription?: boolean
}

export default function ReportActionBar({
  reportId, onSubscribe, hasSubscription,
}: ReportActionBarProps) {
  const [copied, setCopied] = useState(false)

  const handleCopyLink = () => {
    const url = `${window.location.origin}/reports/${reportId}`
    navigator.clipboard.writeText(url).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  return (
    <div className="flex items-center gap-2">
      {/* PDF Export */}
      <button
        onClick={() => downloadReportPdf(reportId)}
        className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-slate-800 border border-slate-700 text-slate-300 hover:text-white hover:bg-slate-700 transition-colors"
        title="Download PDF"
      >
        <Download className="w-3.5 h-3.5" />
        PDF
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
          hasSubscription
            ? 'bg-cyan-900/30 border-cyan-700/50 text-cyan-300 hover:bg-cyan-900/50'
            : 'bg-slate-800 border-slate-700 text-slate-300 hover:text-white hover:bg-slate-700',
        )}
        title={hasSubscription ? 'Manage subscription' : 'Subscribe to updates'}
      >
        {hasSubscription ? <BellOff className="w-3.5 h-3.5" /> : <Bell className="w-3.5 h-3.5" />}
        {hasSubscription ? 'Subscribed' : 'Subscribe'}
      </button>
    </div>
  )
}
