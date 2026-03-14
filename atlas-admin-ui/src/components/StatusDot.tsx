export default function StatusDot({ status }: { status: string }) {
  const color = status === 'error' ? 'bg-red-400' : status === 'completed' ? 'bg-emerald-400' : 'bg-slate-500'
  return <span className={`inline-block h-1.5 w-1.5 rounded-full ${color}`} />
}
