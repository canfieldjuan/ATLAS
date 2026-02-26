"use client";

import { motion } from "framer-motion";
import type { SectorData } from "@/lib/types";

interface GlobalHeatmapProps {
  sectors: SectorData[];
  onSectorClick?: (sector: SectorData) => void;
  activeSectorId?: string;
}

function stressToColor(score: number): string {
  if (score >= 0.8) return "from-orange-500/90 to-red-600/80";
  if (score >= 0.6) return "from-orange-400/70 to-amber-500/60";
  if (score >= 0.4) return "from-cobalt/50 to-blue-500/40";
  return "from-cobalt/30 to-slate-600/20";
}

function stressGlow(score: number): string {
  if (score >= 0.8) return "shadow-[0_0_30px_rgba(249,115,22,0.6)]";
  if (score >= 0.6) return "shadow-[0_0_20px_rgba(249,115,22,0.3)]";
  if (score >= 0.4) return "shadow-[0_0_15px_rgba(37,99,235,0.3)]";
  return "shadow-[0_0_8px_rgba(37,99,235,0.15)]";
}

function stressLabel(score: number): string {
  if (score >= 0.8) return "CRITICAL";
  if (score >= 0.6) return "ELEVATED";
  if (score >= 0.4) return "MODERATE";
  return "BASELINE";
}

export default function GlobalHeatmap({
  sectors,
  onSectorClick,
  activeSectorId,
}: GlobalHeatmapProps) {
  return (
    <section className="w-full">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-bold tracking-wider uppercase text-slate-200">
            Global Pressure Heatmap
          </h2>
          <p className="text-xs text-slate-500 mt-1">
            Sector linguistic stress intensity — click to drill down
          </p>
        </div>
        <div className="flex items-center gap-4 text-[10px] text-slate-500">
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-sm bg-gradient-to-br from-cobalt/30 to-slate-600/20" />
            Baseline
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-sm bg-gradient-to-br from-cobalt/50 to-blue-500/40" />
            Moderate
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-sm bg-gradient-to-br from-orange-400/70 to-amber-500/60" />
            Elevated
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-sm bg-gradient-to-br from-orange-500/90 to-red-600/80" />
            Critical
          </span>
        </div>
      </div>
      <div className="grid grid-cols-3 gap-3">
        {sectors.map((sector) => (
          <motion.button
            key={sector.id}
            onClick={() => onSectorClick?.(sector)}
            whileHover={{ scale: 1.03, y: -2 }}
            whileTap={{ scale: 0.98 }}
            className={`
              relative overflow-hidden rounded-lg border p-4 text-left
              bg-gradient-to-br ${stressToColor(sector.stress_score)}
              ${stressGlow(sector.stress_score)}
              ${activeSectorId === sector.id ? "border-hazard ring-1 ring-hazard/50" : "border-slate-700/50"}
              transition-colors cursor-pointer
            `}
          >
            <div className="relative z-10">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-semibold text-white/90 tracking-wide">
                  {sector.name}
                </span>
                <span
                  className={`text-[9px] font-bold tracking-widest px-1.5 py-0.5 rounded ${
                    sector.stress_score >= 0.8
                      ? "bg-red-500/30 text-red-300"
                      : sector.stress_score >= 0.6
                        ? "bg-orange-500/30 text-orange-300"
                        : "bg-cobalt/30 text-blue-300"
                  }`}
                >
                  {stressLabel(sector.stress_score)}
                </span>
              </div>
              <div className="text-2xl font-mono font-bold text-white/95 mb-1">
                {(sector.stress_score * 100).toFixed(0)}%
              </div>
              <div className="flex items-center gap-3 text-[10px] text-white/60">
                <span>{sector.posts_per_minute} posts/min</span>
                <span>·</span>
                <span>T-{sector.news_delay_predicted}h</span>
              </div>
            </div>
            {/* Animated pulse for critical sectors */}
            {sector.stress_score >= 0.8 && (
              <motion.div
                className="absolute inset-0 bg-orange-500/10 rounded-lg"
                animate={{ opacity: [0, 0.3, 0] }}
                transition={{ duration: 2, repeat: Infinity }}
              />
            )}
          </motion.button>
        ))}
      </div>
    </section>
  );
}
