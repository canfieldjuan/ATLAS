import clsx from 'clsx';
import { useAtlasStore } from '../../state/store';
import { useMemo } from 'react';

interface Particle {
  id: number;
  x: number;
  y: number;
  size: number;
  duration: number;
  delay: number;
}

const generateParticles = (): Particle[] => {
  const particles: Particle[] = [];
  for (let i = 0; i < 20; i++) {
    particles.push({
      id: i,
      x: Math.random() * 100,
      y: Math.random() * 100,
      size: Math.random() * 4 + 2,
      duration: Math.random() * 3 + 2,
      delay: Math.random() * 2,
    });
  }
  return particles;
};

export const Avatar: React.FC = () => {
  const { status } = useAtlasStore();
  const particles = useMemo(() => generateParticles(), []);

  const getOrbColors = () => {
    switch (status) {
      case 'listening':
        return {
          primary: 'from-blue-400 via-cyan-500 to-blue-600',
          glow: 'shadow-[0_0_80px_30px_rgba(59,130,246,0.6)]',
          particle: 'bg-blue-400',
          ring: 'border-blue-400/50',
        };
      case 'processing':
        return {
          primary: 'from-amber-400 via-yellow-500 to-orange-500',
          glow: 'shadow-[0_0_80px_30px_rgba(245,158,11,0.6)]',
          particle: 'bg-yellow-400',
          ring: 'border-yellow-400/50',
        };
      case 'speaking':
        return {
          primary: 'from-purple-400 via-violet-500 to-purple-600',
          glow: 'shadow-[0_0_100px_40px_rgba(168,85,247,0.7)]',
          particle: 'bg-purple-400',
          ring: 'border-purple-400/50',
        };
      case 'error':
        return {
          primary: 'from-red-400 via-rose-500 to-red-600',
          glow: 'shadow-[0_0_80px_30px_rgba(239,68,68,0.6)]',
          particle: 'bg-red-400',
          ring: 'border-red-400/50',
        };
      default:
        return {
          primary: 'from-slate-300 via-slate-400 to-slate-500',
          glow: 'shadow-[0_0_40px_15px_rgba(148,163,184,0.3)]',
          particle: 'bg-slate-400',
          ring: 'border-slate-400/30',
        };
    }
  };

  const colors = getOrbColors();
  const isActive = status !== 'idle';

  return (
    <div className="relative flex items-center justify-center w-72 h-72">
      {/* Particle field */}
      <div className="absolute inset-0 overflow-hidden rounded-full">
        {particles.map((particle) => (
          <div
            key={particle.id}
            className={clsx(
              "absolute rounded-full opacity-60 blur-[1px]",
              colors.particle,
              isActive ? 'animate-float' : 'opacity-20'
            )}
            style={{
              left: `${particle.x}%`,
              top: `${particle.y}%`,
              width: `${particle.size}px`,
              height: `${particle.size}px`,
              animationDuration: `${particle.duration}s`,
              animationDelay: `${particle.delay}s`,
            }}
          />
        ))}
      </div>

      {/* Outer rotating ring */}
      <div
        className={clsx(
          "absolute w-64 h-64 rounded-full border-2 transition-all duration-700",
          colors.ring,
          isActive ? 'animate-spin-slow opacity-60' : 'opacity-20'
        )}
        style={{ animationDuration: '8s' }}
      />

      {/* Second rotating ring (opposite direction) */}
      <div
        className={clsx(
          "absolute w-56 h-56 rounded-full border transition-all duration-700",
          colors.ring,
          isActive ? 'animate-reverse-spin opacity-40' : 'opacity-10'
        )}
        style={{ animationDuration: '12s' }}
      />

      {/* Orbital dots */}
      {isActive && (
        <div className="absolute w-60 h-60 animate-spin-slow" style={{ animationDuration: '6s' }}>
          <div className={clsx("absolute top-0 left-1/2 -translate-x-1/2 w-3 h-3 rounded-full", colors.particle, "shadow-lg")} />
          <div className={clsx("absolute bottom-0 left-1/2 -translate-x-1/2 w-2 h-2 rounded-full", colors.particle, "opacity-60")} />
        </div>
      )}

      {/* Main 3D Orb */}
      <div
        className={clsx(
          "relative w-40 h-40 rounded-full transition-all duration-500",
          colors.glow,
          status === 'listening' && 'scale-110',
          status === 'speaking' && 'scale-105 animate-pulse-slow',
          status === 'processing' && 'animate-pulse'
        )}
      >
        {/* Orb gradient background */}
        <div
          className={clsx(
            "absolute inset-0 rounded-full bg-gradient-to-br transition-all duration-500",
            colors.primary
          )}
        />

        {/* 3D highlight (top-left shine) */}
        <div className="absolute inset-0 rounded-full bg-gradient-to-br from-white/40 via-transparent to-transparent" />

        {/* Inner depth shadow */}
        <div className="absolute inset-4 rounded-full bg-gradient-to-br from-transparent via-black/10 to-black/30" />

        {/* Core glow */}
        <div className="absolute inset-8 rounded-full bg-white/20 blur-md" />

        {/* Center bright spot */}
        <div className="absolute top-6 left-6 w-8 h-8 rounded-full bg-white/50 blur-sm" />

        {/* Status indicator ring */}
        {status === 'listening' && (
          <div className="absolute inset-0 rounded-full border-4 border-white/30 animate-ping" />
        )}
      </div>

      {/* Ambient glow rings */}
      <div
        className={clsx(
          "absolute w-48 h-48 rounded-full transition-all duration-700 blur-xl",
          isActive ? 'opacity-30' : 'opacity-10',
          `bg-gradient-to-r ${colors.primary}`
        )}
      />

      {/* Pulse rings for speaking */}
      {status === 'speaking' && (
        <>
          <div className="absolute w-52 h-52 rounded-full border border-purple-400/40 animate-ping" style={{ animationDuration: '1.5s' }} />
          <div className="absolute w-60 h-60 rounded-full border border-purple-300/20 animate-ping" style={{ animationDuration: '2s' }} />
        </>
      )}
    </div>
  );
};
