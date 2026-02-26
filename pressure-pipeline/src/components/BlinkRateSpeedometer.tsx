"use client";

import { motion } from "framer-motion";
import { Activity } from "lucide-react";

interface BlinkRateSpeedometerProps {
  postsPerMinute: number;
  maxRate?: number;
}

export default function BlinkRateSpeedometer({
  postsPerMinute,
  maxRate = 600,
}: BlinkRateSpeedometerProps) {
  const ratio = Math.min(postsPerMinute / maxRate, 1);
  const angle = -135 + ratio * 270; // -135° to +135° arc
  const circumference = 2 * Math.PI * 80;
  const strokeDash = ratio * circumference * 0.75;

  const status =
    ratio >= 0.8
      ? { label: "CRITICAL", color: "text-red-400" }
      : ratio >= 0.6
        ? { label: "HIGH", color: "text-hazard" }
        : ratio >= 0.4
          ? { label: "ELEVATED", color: "text-amber-400" }
          : { label: "NORMAL", color: "text-cobalt-light" };

  return (
    <div className="bg-obsidian/50 border border-slate-800 rounded-xl p-4">
      <div className="flex items-center gap-2 mb-3">
        <Activity size={14} className="text-hazard" />
        <h3 className="text-sm font-bold tracking-wider uppercase text-slate-200">
          Blink Rate
        </h3>
      </div>
      <div className="flex flex-col items-center">
        <div className="relative w-44 h-28 overflow-hidden">
          <svg
            viewBox="0 0 200 120"
            className="w-full h-full"
            aria-label={`Blink rate gauge: ${postsPerMinute} posts per minute`}
          >
            {/* Background arc */}
            <path
              d="M 20 110 A 80 80 0 0 1 180 110"
              fill="none"
              stroke="#1e293b"
              strokeWidth="12"
              strokeLinecap="round"
            />
            {/* Active arc */}
            <motion.path
              d="M 20 110 A 80 80 0 0 1 180 110"
              fill="none"
              stroke={ratio >= 0.8 ? "#ef4444" : ratio >= 0.6 ? "#f97316" : "#2563eb"}
              strokeWidth="12"
              strokeLinecap="round"
              strokeDasharray={`${strokeDash} ${circumference}`}
              initial={{ strokeDasharray: `0 ${circumference}` }}
              animate={{ strokeDasharray: `${strokeDash} ${circumference}` }}
              transition={{ duration: 1.5, ease: "easeOut" }}
            />
            {/* Needle */}
            <motion.line
              x1="100"
              y1="110"
              x2="100"
              y2="40"
              stroke="#f8fafc"
              strokeWidth="2"
              strokeLinecap="round"
              style={{ transformOrigin: "100px 110px" }}
              initial={{ rotate: -135 }}
              animate={{ rotate: angle }}
              transition={{ duration: 1.5, ease: "easeOut", type: "spring" }}
            />
            {/* Center dot */}
            <circle cx="100" cy="110" r="5" fill="#f8fafc" />
          </svg>
        </div>
        <div className="text-center -mt-2">
          <span className="text-3xl font-mono font-bold text-white">
            {postsPerMinute}
          </span>
          <span className="text-xs text-slate-500 ml-1">posts/min</span>
        </div>
        <span className={`text-[10px] font-bold tracking-widest mt-1 ${status.color}`}>
          {status.label}
        </span>
      </div>
    </div>
  );
}
