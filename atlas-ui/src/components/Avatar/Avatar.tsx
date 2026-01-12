import clsx from 'clsx';
import { useAtlasStore } from '../../state/store';
export const Avatar: React.FC = () => {
  const { status, audioAnalysis } = useAtlasStore();
  const isActive = status !== 'idle';

  return (
    <div className="relative flex items-center justify-center w-96 h-96">
      
      {/* Background Ambience Layer - deep network */}
      <svg className="absolute w-full h-full opacity-60" viewBox="0 0 400 400">
        <defs>
          <radialGradient id="deep-glow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.2" />
            <stop offset="100%" stopColor="#000000" stopOpacity="0" />
          </radialGradient>
          <filter id="blur-node">
            <feGaussianBlur stdDeviation="3" />
          </filter>
        </defs>
        
        {/* Deep background glow */}
        <circle cx="200" cy="200" r="180" fill="url(#deep-glow)" />
        
        {/* Background faint web */}
        {[...Array(40)].map((_, i) => {
           // Random stable seed based on index
           const seed = i * 1337;
           const angle1 = (i / 40) * Math.PI * 2;
           const angle2 = ((i + 15) / 40) * Math.PI * 2;
           const r = 180;
           const x1 = 200 + Math.cos(angle1) * r;
           const y1 = 200 + Math.sin(angle1) * r;
           const x2 = 200 + Math.cos(angle2) * r;
           const y2 = 200 + Math.sin(angle2) * r;
           const controlR = 50 + (seed % 100); 
           const cx = 200 + Math.cos(angle1 + 1) * controlR;
           const cy = 200 + Math.sin(angle1 + 1) * controlR;

           return (
             <path 
               key={`bg-web-${i}`}
               d={`M ${x1} ${y1} Q ${cx} ${cy} ${x2} ${y2}`}
               stroke="#0e7490" 
               strokeWidth="0.5"
               fill="none"
               opacity="0.2"
             />
           );
        })}
      </svg>

      {/* Primary Neural Web - High Detail */}
      <svg 
        className={clsx(
          "absolute w-full h-full transition-all duration-500",
          isActive && "scale-105"
        )} 
        viewBox="0 0 400 400"
        style={{
           transform: audioAnalysis.isActive ? `scale(${1 + audioAnalysis.volume * 0.002})` : undefined
        }}
      >
        <defs>
          <linearGradient id="neural-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#ffffff" />
            <stop offset="100%" stopColor="#22d3ee" />
          </linearGradient>
          <filter id="glow-intense">
            <feGaussianBlur stdDeviation="2.5" result="coloredBlur"/>
            <feMerge>
              <feMergeNode in="coloredBlur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
          <mask id="fade-mask">
            <radialGradient id="fade-grad">
              <stop offset="60%" stopColor="white" />
              <stop offset="100%" stopColor="black" />
            </radialGradient>
            <circle cx="200" cy="200" r="200" fill="url(#fade-grad)" />
          </mask>
        </defs>

        <g mask="url(#fade-mask)">
          {/* Main Organic Fibers */}
          {[...Array(60)].map((_, i) => {
            const seed = i * 9382;
            const angleStart = (i / 60) * Math.PI * 2;
            // Connect to point roughly across but varied
            const angleEnd = angleStart + Math.PI + (Math.sin(seed) * 1.5);
            
            const rStart = 160 + (Math.sin(seed * 0.1) * 20);
            const rEnd = 160 + (Math.cos(seed * 0.1) * 20);

            const x1 = 200 + Math.cos(angleStart) * rStart;
            const y1 = 200 + Math.sin(angleStart) * rStart;
            
            const x2 = 200 + Math.cos(angleEnd) * rEnd;
            const y2 = 200 + Math.sin(angleEnd) * rEnd;

            // Control points for organic curve
            const cp1x = 200 + Math.cos(angleStart + 0.5) * 80;
            const cp1y = 200 + Math.sin(angleStart + 0.5) * 80;
            const cp2x = 200 + Math.cos(angleEnd - 0.5) * 80;
            const cp2y = 200 + Math.sin(angleEnd - 0.5) * 80;

            return (
              <path
                key={`fiber-${i}`}
                d={`M ${x1} ${y1} C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${x2} ${y2}`}
                stroke="url(#neural-gradient)"
                strokeWidth={isActive ? Math.max(0.5, (Math.sin(i) + 1.5) * 0.8) : 0.5}
                fill="none"
                opacity={isActive ? 0.6 : 0.3}
                className="transition-all duration-1000"
              />
            );
          })}
          
          {/* Active Signal Pulses */}
          {isActive && [...Array(12)].map((_, i) => (
             <circle key={`pulse-${i}`} r="2" fill="#fff" filter="url(#glow-intense)">
               <animateMotion 
                 dur={`${2 + i % 3}s`} 
                 repeatCount="indefinite"
                 path={`M 200 200 Q ${200 + Math.cos(i) * 100} ${200 + Math.sin(i) * 100} ${200 + Math.cos(i) * 180} ${200 + Math.sin(i) * 180}`}
               />
               <animate attributeName="opacity" values="0;1;0" dur={`${2 + i % 3}s`} repeatCount="indefinite" />
             </circle>
          ))}
        </g>
        
        {/* Central Neural Cluster */}
        <g filter="url(#glow-intense)">
            {/* Core Nodes */}
            {[...Array(15)].map((_, i) => {
                const angle = (i / 15) * Math.PI * 2;
                const r = 25 + (i % 3) * 10;
                const x = 200 + Math.cos(angle) * r;
                const y = 200 + Math.sin(angle) * r;
                const size = 3 + (i % 4);
                
                return (
                    <circle 
                        key={`node-${i}`} 
                        cx={x} cy={y} r={size} 
                        fill="#fff" 
                        opacity={isActive ? 0.9 : 0.6}
                    />
                );
            })}
            
            {/* Inner connections */}
            {[...Array(20)].map((_, i) => {
                const angle = (i / 20) * Math.PI * 2;
                const x = 200 + Math.cos(angle) * 35;
                const y = 200 + Math.sin(angle) * 35;
                return (
                    <line 
                        key={`core-line-${i}`}
                        x1="200" y1="200" x2={x} y2={y}
                        stroke="#22d3ee"
                        strokeWidth="1"
                        opacity="0.6"
                    />
                );
            })}
            
            {/* Central Nucleus */}
            <circle cx="200" cy="200" r="12" fill="#fff" />
            <circle cx="200" cy="200" r="20" fill="none" stroke="#22d3ee" strokeWidth="2" opacity="0.8">
               <animateTransform attributeName="transform" type="scale" values="1;1.2;1" dur="2s" repeatCount="indefinite" />
            </circle>
        </g>
      </svg>
      
      {/* Dynamic Text Overlay */}
      {status !== 'idle' && (
          <div className="absolute top-[85%] font-mono text-cyan-400 text-xs tracking-[0.3em] opacity-80">
            {status.toUpperCase()}
          </div>
      )}
    </div>
  );
};
