export default function AtlasRobotLogo({ className = 'h-7 w-7' }: { className?: string }) {
  return (
    <svg viewBox="-18 -32 36 36" className={className} aria-label="Atlas Robot">
      <defs>
        <radialGradient id="rl-body" cx="40%" cy="35%">
          <stop offset="0%" stopColor="#9abdd8" />
          <stop offset="100%" stopColor="#6a8caf" />
        </radialGradient>
        <filter id="rl-glow">
          <feGaussianBlur stdDeviation="1.2" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
      {/* Antenna stem */}
      <line x1={0} y1={-13} x2={0} y2={-25} stroke="#2e4a68" strokeWidth={2} strokeLinecap="round" />
      {/* Antenna ball */}
      <circle cx={0} cy={-28.5} r={3.5} fill="#00e5ff" filter="url(#rl-glow)" />
      {/* Head */}
      <circle cx={0} cy={0} r={13} fill="url(#rl-body)" />
      {/* Eyes */}
      <circle cx={-4.5} cy={-2} r={2.5} fill="#00e5ff" filter="url(#rl-glow)" />
      <circle cx={4.5} cy={-2} r={2.5} fill="#00e5ff" filter="url(#rl-glow)" />
    </svg>
  )
}
