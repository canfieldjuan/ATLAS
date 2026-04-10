import { useState, useEffect, useCallback, useRef, type ReactNode } from 'react'
import {
  X, ExternalLink, Quote, User, Building2, Calendar,
  Star, Tag, Fingerprint, FileText, ChevronRight, Loader2,
  Pin, Flag, EyeOff, RotateCcw,
} from 'lucide-react'
import { Link } from 'react-router-dom'
import { clsx } from 'clsx'
import { fetchWitness, fetchAnnotations, setAnnotation, removeAnnotations } from '../api/client'
import type { EvidenceWitnessDetail, EvidenceAnnotation } from '../api/client'

// Inject keyframes once at module load (not per-render)
if (typeof document !== 'undefined' && !document.getElementById('evidence-drawer-keyframes')) {
  const style = document.createElement('style')
  style.id = 'evidence-drawer-keyframes'
  style.textContent = `
    @keyframes slideInRight {
      from { transform: translateX(100%); opacity: 0.8; }
      to { transform: translateX(0); opacity: 1; }
    }
  `
  document.head.appendChild(style)
}

const SOURCE_COLORS: Record<string, string> = {
  g2: 'bg-orange-500/20 text-orange-300 border-orange-500/30',
  capterra: 'bg-blue-500/20 text-blue-300 border-blue-500/30',
  trustradius: 'bg-violet-500/20 text-violet-300 border-violet-500/30',
  reddit: 'bg-red-500/20 text-red-300 border-red-500/30',
  gartner: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
  peerspot: 'bg-teal-500/20 text-teal-300 border-teal-500/30',
  trustpilot: 'bg-green-500/20 text-green-300 border-green-500/30',
  getapp: 'bg-sky-500/20 text-sky-300 border-sky-500/30',
  hackernews: 'bg-amber-500/20 text-amber-300 border-amber-500/30',
  twitter: 'bg-cyan-500/20 text-cyan-300 border-cyan-500/30',
  youtube: 'bg-rose-500/20 text-rose-300 border-rose-500/30',
  stackoverflow: 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30',
  producthunt: 'bg-pink-500/20 text-pink-300 border-pink-500/30',
  github: 'bg-slate-400/20 text-slate-300 border-slate-400/30',
  quora: 'bg-red-400/20 text-red-300 border-red-400/30',
  rss: 'bg-amber-400/20 text-amber-300 border-amber-400/30',
}

const SIGNAL_COLORS: Record<string, string> = {
  pricing_backlash: 'bg-red-900/40 text-red-300',
  complaint: 'bg-orange-900/40 text-orange-300',
  feature_gap: 'bg-amber-900/40 text-amber-300',
  competitor_pressure: 'bg-violet-900/40 text-violet-300',
  recommendation: 'bg-emerald-900/40 text-emerald-300',
  positive_anchor: 'bg-green-900/40 text-green-300',
  event: 'bg-cyan-900/40 text-cyan-300',
}

function SourceBadge({ source }: { source: string }) {
  const colors = SOURCE_COLORS[source.toLowerCase()] || 'bg-slate-600/30 text-slate-300 border-slate-500/30'
  return (
    <span className={clsx('inline-flex items-center px-2 py-0.5 rounded text-xs font-medium border', colors)}>
      {source}
    </span>
  )
}

function highlightExcerpt(fullText: string, excerptText: string): ReactNode {
  if (!fullText || !excerptText) return <>{fullText || ''}</>

  const lower = fullText.toLowerCase()
  const excerptLower = excerptText.toLowerCase().slice(0, 120)
  const idx = lower.indexOf(excerptLower)

  if (idx === -1) return <>{fullText}</>

  const before = fullText.slice(0, idx)
  const match = fullText.slice(idx, idx + excerptText.length)
  const after = fullText.slice(idx + excerptText.length)

  return (
    <>
      {before}
      <mark className="bg-cyan-500/20 text-cyan-200 rounded px-0.5">{match}</mark>
      {after}
    </>
  )
}

interface EvidenceDrawerProps {
  vendorName: string
  witnessId: string | null
  asOfDate?: string | null
  windowDays?: number
  open: boolean
  onClose: () => void
}

export default function EvidenceDrawer({
  vendorName,
  witnessId,
  asOfDate,
  windowDays,
  open,
  onClose,
}: EvidenceDrawerProps) {
  const [witness, setWitness] = useState<EvidenceWitnessDetail | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [annotation, setAnnotationState] = useState<EvidenceAnnotation | null>(null)
  const [annotating, setAnnotating] = useState(false)
  const panelRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open || !witnessId || !vendorName) return
    setLoading(true)
    setError('')
    setAnnotationState(null)
    Promise.all([
      fetchWitness(witnessId, vendorName, {
        as_of_date: asOfDate || undefined,
        window_days: windowDays,
      }),
      fetchAnnotations({ vendor_name: vendorName }).catch(() => ({ annotations: [] })),
    ])
      .then(([witnessRes, annotRes]) => {
        setWitness(witnessRes.witness)
        const match = annotRes.annotations.find((a: EvidenceAnnotation) => a.witness_id === witnessId)
        setAnnotationState(match || null)
      })
      .catch(err => setError(err instanceof Error ? err.message : 'Failed to load witness'))
      .finally(() => setLoading(false))
  }, [open, witnessId, vendorName, asOfDate, windowDays])

  async function handleAnnotate(type: 'pin' | 'flag' | 'suppress') {
    if (!witnessId || !vendorName) return
    setAnnotating(true)
    try {
      const result = await setAnnotation({
        witness_id: witnessId,
        vendor_name: vendorName,
        annotation_type: type,
      })
      setAnnotationState(result)
    } catch {
      // keep current state
    } finally {
      setAnnotating(false)
    }
  }

  async function handleRemoveAnnotation() {
    if (!witnessId) return
    setAnnotating(true)
    try {
      await removeAnnotations({ witness_ids: [witnessId] })
      setAnnotationState(null)
    } catch {
      // keep current state
    } finally {
      setAnnotating(false)
    }
  }

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (e.key === 'Escape') onClose()
  }, [onClose])

  useEffect(() => {
    if (open) document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [open, handleKeyDown])

  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />

      {/* Panel */}
      <div
        ref={panelRef}
        className="relative w-full max-w-2xl bg-slate-900 border-l border-slate-700/50 shadow-2xl overflow-y-auto"
        style={{ animation: 'slideInRight 0.2s ease-out' }}
      >
        {/* Header */}
        <div className="sticky top-0 z-10 bg-slate-900/95 backdrop-blur-sm border-b border-slate-700/50 px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Fingerprint className="w-5 h-5 text-cyan-400" />
              <h2 className="text-lg font-semibold text-white">Witness Detail</h2>
              {annotation && (
                <span className={clsx(
                  'inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium',
                  annotation.annotation_type === 'pin' && 'bg-amber-500/20 text-amber-300',
                  annotation.annotation_type === 'flag' && 'bg-red-500/20 text-red-300',
                  annotation.annotation_type === 'suppress' && 'bg-slate-500/20 text-slate-400',
                )}>
                  {annotation.annotation_type === 'pin' && <Pin className="h-3 w-3" />}
                  {annotation.annotation_type === 'flag' && <Flag className="h-3 w-3" />}
                  {annotation.annotation_type === 'suppress' && <EyeOff className="h-3 w-3" />}
                  {annotation.annotation_type}
                </span>
              )}
            </div>
            <button onClick={onClose} className="p-1 rounded hover:bg-slate-700/50 text-slate-400 hover:text-white transition-colors">
              <X className="w-5 h-5" />
            </button>
          </div>
          {witness && (
            <div className="mt-3 flex items-center gap-2">
              {annotation ? (
                <button
                  onClick={handleRemoveAnnotation}
                  disabled={annotating}
                  className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium bg-slate-700/50 text-slate-300 hover:bg-slate-600/50 disabled:opacity-50 transition-colors"
                >
                  <RotateCcw className="h-3 w-3" />
                  Remove {annotation.annotation_type}
                </button>
              ) : (
                <>
                  <button
                    onClick={() => handleAnnotate('pin')}
                    disabled={annotating}
                    className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium bg-amber-500/10 text-amber-300 hover:bg-amber-500/20 disabled:opacity-50 transition-colors"
                    title="Pin: prioritize in campaign generation"
                  >
                    <Pin className="h-3 w-3" />
                    Pin
                  </button>
                  <button
                    onClick={() => handleAnnotate('flag')}
                    disabled={annotating}
                    className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium bg-red-500/10 text-red-300 hover:bg-red-500/20 disabled:opacity-50 transition-colors"
                    title="Flag: mark for quality review"
                  >
                    <Flag className="h-3 w-3" />
                    Flag
                  </button>
                  <button
                    onClick={() => handleAnnotate('suppress')}
                    disabled={annotating}
                    className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium bg-slate-500/10 text-slate-400 hover:bg-slate-500/20 disabled:opacity-50 transition-colors"
                    title="Suppress: exclude from campaign generation"
                  >
                    <EyeOff className="h-3 w-3" />
                    Suppress
                  </button>
                </>
              )}
            </div>
          )}
        </div>

        {loading && (
          <div className="flex items-center justify-center py-20">
            <Loader2 className="w-6 h-6 text-cyan-400 animate-spin" />
          </div>
        )}

        {error && (
          <div className="mx-6 mt-6 p-4 bg-red-900/20 border border-red-800/50 rounded-lg text-red-300 text-sm">{error}</div>
        )}

        {witness && !loading && (
          <div className="p-6 space-y-6">
            {/* Excerpt card */}
            <div className="bg-slate-800/60 rounded-xl border border-cyan-500/20 p-5">
              <div className="flex items-start gap-3 mb-3">
                <Quote className="w-4 h-4 text-cyan-400 mt-1 shrink-0" />
                <p className="text-slate-200 text-sm leading-relaxed italic">
                  &ldquo;{witness.excerpt_text}&rdquo;
                </p>
              </div>
              <div className="flex items-center gap-3 mt-3 flex-wrap">
                <SourceBadge source={witness.source || 'unknown'} />
                {witness.pain_category && (
                  <span className="text-xs px-2 py-0.5 rounded bg-red-900/30 text-red-300 border border-red-800/40">
                    {witness.pain_category}
                  </span>
                )}
                {witness.competitor && (
                  <span className="text-xs px-2 py-0.5 rounded bg-violet-900/30 text-violet-300 border border-violet-800/40">
                    vs {witness.competitor}
                  </span>
                )}
                {witness.salience_score != null && (
                  <span className="text-xs text-slate-500 font-mono">
                    salience: {witness.salience_score.toFixed(2)}
                  </span>
                )}
              </div>
            </div>

            {/* Reviewer context */}
            <div className="grid grid-cols-2 gap-3">
              {witness.reviewer_company && (
                <div className="bg-slate-800/40 rounded-lg p-3 border border-slate-700/30">
                  <div className="flex items-center gap-2 text-slate-500 text-xs mb-1">
                    <Building2 className="w-3 h-3" /> Company
                  </div>
                  <span className="text-sm text-slate-200">{witness.reviewer_company}</span>
                </div>
              )}
              {witness.reviewer_title && (
                <div className="bg-slate-800/40 rounded-lg p-3 border border-slate-700/30">
                  <div className="flex items-center gap-2 text-slate-500 text-xs mb-1">
                    <User className="w-3 h-3" /> Title
                  </div>
                  <span className="text-sm text-slate-200">{witness.reviewer_title}</span>
                </div>
              )}
              {witness.reviewed_at && (
                <div className="bg-slate-800/40 rounded-lg p-3 border border-slate-700/30">
                  <div className="flex items-center gap-2 text-slate-500 text-xs mb-1">
                    <Calendar className="w-3 h-3" /> Reviewed
                  </div>
                  <span className="text-sm text-slate-200">
                    {new Date(witness.reviewed_at).toLocaleDateString()}
                  </span>
                </div>
              )}
              {witness.rating != null && (
                <div className="bg-slate-800/40 rounded-lg p-3 border border-slate-700/30">
                  <div className="flex items-center gap-2 text-slate-500 text-xs mb-1">
                    <Star className="w-3 h-3" /> Rating
                  </div>
                  <span className="text-sm text-slate-200">{witness.rating}/5</span>
                </div>
              )}
            </div>

            {/* Signal tags */}
            {witness.signal_tags && witness.signal_tags.length > 0 && (
              <div>
                <h3 className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-2">
                  <Tag className="w-3 h-3" /> Signal Tags
                </h3>
                <div className="flex flex-wrap gap-1.5">
                  {witness.signal_tags.map((tag, i) => (
                    <span key={i} className={clsx(
                      'text-xs px-2 py-0.5 rounded',
                      SIGNAL_COLORS[tag] || 'bg-slate-700/50 text-slate-300',
                    )}>
                      {tag.replace(/_/g, ' ')}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Witness metadata */}
            <div className="bg-slate-800/30 rounded-lg p-3 border border-slate-700/20">
              <h3 className="text-xs font-medium text-slate-500 uppercase tracking-wider mb-2 flex items-center gap-2">
                <Fingerprint className="w-3 h-3" /> Witness Metadata
              </h3>
              <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                <span className="text-slate-500">ID</span>
                <span className="text-slate-300 font-mono truncate">{witness.witness_id}</span>
                <span className="text-slate-500">Type</span>
                <span className="text-slate-300">{witness.witness_type || 'unknown'}</span>
                {witness.specificity_score != null && (
                  <>
                    <span className="text-slate-500">Specificity</span>
                    <span className="text-slate-300">{witness.specificity_score.toFixed(2)}</span>
                  </>
                )}
                {witness.selection_reason && (
                  <>
                    <span className="text-slate-500">Selection</span>
                    <span className="text-slate-300">{witness.selection_reason}</span>
                  </>
                )}
              </div>
            </div>

            {/* Full review text with highlight */}
            {witness.review_text && (
              <div>
                <h3 className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-2">
                  <FileText className="w-3 h-3" /> Source Review
                  {vendorName && (
                    <Link
                      to={`/reports?vendor_filter=${encodeURIComponent(vendorName)}`}
                      className="inline-flex items-center gap-1 text-xs text-cyan-400 hover:text-cyan-300 mt-1"
                    >
                      View reports <ExternalLink className="h-3 w-3" />
                    </Link>
                  )}
                  {witness.source_url && (
                    <a href={witness.source_url} target="_blank" rel="noopener noreferrer"
                       className="ml-auto text-cyan-400 hover:text-cyan-300 flex items-center gap-1">
                      <ExternalLink className="w-3 h-3" /> View original
                    </a>
                  )}
                </h3>
                <div className="bg-slate-950/60 rounded-lg p-4 border border-slate-700/30 text-sm text-slate-300 leading-relaxed max-h-80 overflow-y-auto">
                  {highlightExcerpt(witness.review_text, witness.excerpt_text)}
                </div>
              </div>
            )}

            {/* Evidence spans from this review */}
            {witness.evidence_spans && witness.evidence_spans.length > 0 && (
              <div>
                <h3 className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-2">
                  <ChevronRight className="w-3 h-3" />
                  Related Evidence Spans ({witness.evidence_spans.length} of {witness.all_evidence_span_count})
                </h3>
                <div className="space-y-2">
                  {witness.evidence_spans.map((span, i) => (
                    <div key={i} className="bg-slate-800/40 rounded-lg p-3 border border-slate-700/20">
                      <div className="flex items-center gap-2 mb-1">
                        <span className={clsx(
                          'text-xs px-1.5 py-0.5 rounded',
                          SIGNAL_COLORS[span.signal_type] || 'bg-slate-700/50 text-slate-300',
                        )}>
                          {span.signal_type?.replace(/_/g, ' ')}
                        </span>
                        {span.pain_category && (
                          <span className="text-xs text-slate-500">{span.pain_category}</span>
                        )}
                      </div>
                      <p className="text-xs text-slate-400 italic">
                        &ldquo;{span.raw_text || span.excerpt_text}&rdquo;
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

    </div>
  )
}

export { SourceBadge, SOURCE_COLORS, SIGNAL_COLORS }
