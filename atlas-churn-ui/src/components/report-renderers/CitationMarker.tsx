interface CitationMarkerProps {
  index: number
  witnessId: string
  vendorName: string
  onOpenWitness: (witnessId: string, vendorName: string) => void
  companyHint?: string
}

export default function CitationMarker({
  index, witnessId, vendorName, onOpenWitness, companyHint,
}: CitationMarkerProps) {
  return (
    <button
      type="button"
      onClick={(e) => { e.stopPropagation(); onOpenWitness(witnessId, vendorName) }}
      title={companyHint ? `${companyHint} — click to view evidence` : 'Click to view evidence'}
      className="text-[10px] text-cyan-400 hover:text-cyan-200 cursor-pointer align-super font-mono ml-0.5 px-0.5 rounded hover:bg-cyan-500/15 transition-colors"
    >
      [{index}]
    </button>
  )
}
