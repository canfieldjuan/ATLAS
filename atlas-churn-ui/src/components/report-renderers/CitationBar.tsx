import { useState } from 'react'
import { ChevronRight } from 'lucide-react'
import { SourceBadge } from '../EvidenceDrawer'
import type { CitationEntry } from './useCitationRegistry'

const COLLAPSED_MAX = 5

interface CitationBarProps {
  citations: CitationEntry[]
  vendorName: string
  onOpenWitness: (witnessId: string, vendorName: string) => void
}

export default function CitationBar({ citations, vendorName, onOpenWitness }: CitationBarProps) {
  const [expanded, setExpanded] = useState(false)

  if (citations.length === 0) return null

  const visible = expanded ? citations : citations.slice(0, COLLAPSED_MAX)

  return (
    <div className="mt-3 border-t border-slate-700/30 pt-2">
      <div className="flex items-center gap-1.5 flex-wrap">
        <span className="text-[10px] uppercase tracking-wider text-slate-500 mr-1">Sources</span>
        {visible.map((c) => (
          <button
            key={c.witnessId}
            type="button"
            onClick={() => onOpenWitness(c.witnessId, vendorName)}
            title={c.companyName ? `${c.companyName} — click to view evidence` : 'Click to view evidence'}
            className="text-[10px] font-mono px-1 py-0.5 rounded bg-cyan-900/30 text-cyan-300 border border-cyan-700/40 hover:bg-cyan-900/50 transition-colors cursor-pointer"
          >
            [{c.index}]
          </button>
        ))}
        {!expanded && citations.length > COLLAPSED_MAX && (
          <span className="text-[10px] text-slate-500 ml-0.5">+{citations.length - COLLAPSED_MAX}</span>
        )}
        {!expanded && (
          <button
            type="button"
            onClick={() => setExpanded(true)}
            className="text-[10px] text-slate-500 hover:text-slate-300 transition-colors flex items-center gap-0.5 ml-1"
          >
            <ChevronRight className="w-3 h-3" />
            details
          </button>
        )}
      </div>

      {expanded && (
        <div className="mt-2 space-y-1.5">
          {visible.map((c) => (
            <button
              key={c.witnessId}
              type="button"
              onClick={() => onOpenWitness(c.witnessId, vendorName)}
              className="flex items-start gap-2 w-full text-left rounded-md px-2 py-1.5 hover:bg-slate-800/60 transition-colors group"
            >
              <span className="text-[10px] font-mono text-cyan-400 shrink-0 mt-0.5">[{c.index}]</span>
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2 flex-wrap">
                  {c.companyName && (
                    <span className="text-xs text-slate-300 font-medium">{c.companyName}</span>
                  )}
                  {c.source && <SourceBadge source={c.source} />}
                </div>
                {c.excerptSnippet && (
                  <p className="text-xs text-slate-500 italic line-clamp-1 mt-0.5 group-hover:text-slate-400 transition-colors">
                    &ldquo;{c.excerptSnippet}&rdquo;
                  </p>
                )}
              </div>
            </button>
          ))}
          <button
            type="button"
            onClick={() => setExpanded(false)}
            className="text-[10px] text-slate-500 hover:text-slate-300 transition-colors pl-2"
          >
            collapse
          </button>
        </div>
      )}
    </div>
  )
}
