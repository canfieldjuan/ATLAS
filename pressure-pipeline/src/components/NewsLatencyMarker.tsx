"use client";

import { motion } from "framer-motion";
import { Clock, CheckCircle, AlertTriangle } from "lucide-react";
import type { NewsEvent } from "@/lib/types";

interface NewsLatencyMarkerProps {
  events: NewsEvent[];
}

export default function NewsLatencyMarker({ events }: NewsLatencyMarkerProps) {
  return (
    <div className="bg-obsidian/50 border border-slate-800 rounded-xl p-4">
      <div className="flex items-center gap-2 mb-4">
        <Clock size={14} className="text-cobalt-light" />
        <h3 className="text-sm font-bold tracking-wider uppercase text-slate-200">
          News Latency
        </h3>
        <span className="text-[10px] text-slate-500 ml-auto">
          Prediction → Headline Gap
        </span>
      </div>
      <div className="relative">
        {/* Vertical timeline line */}
        <div className="absolute left-4 top-0 bottom-0 w-px bg-slate-700" />

        <div className="space-y-4">
          {events.map((event, idx) => (
            <motion.div
              key={event.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.1 }}
              className="relative pl-10"
            >
              {/* Timeline dot */}
              <div
                className={`absolute left-2.5 top-1 w-3 h-3 rounded-full border-2 ${
                  event.actual_at
                    ? "bg-green-500/30 border-green-500"
                    : "bg-hazard/30 border-hazard animate-pulse"
                }`}
              />
              <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-3">
                <div className="flex items-start justify-between gap-2 mb-1">
                  <span className="text-xs text-slate-300 font-medium leading-tight">
                    {event.headline}
                  </span>
                  {event.actual_at ? (
                    <CheckCircle size={14} className="text-green-500 shrink-0" />
                  ) : (
                    <AlertTriangle size={14} className="text-hazard shrink-0" />
                  )}
                </div>
                <div className="flex items-center gap-3 text-[10px]">
                  <span className="text-slate-500 uppercase">
                    {event.sector}
                  </span>
                  <span className="font-mono font-bold text-hazard">
                    {event.actual_at ? "+" : "~"}
                    {event.delay_hours}h gap
                  </span>
                  {!event.actual_at && (
                    <span className="text-amber-400/80 font-bold">
                      PENDING
                    </span>
                  )}
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
}
