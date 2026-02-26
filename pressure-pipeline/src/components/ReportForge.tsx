"use client";

import { motion } from "framer-motion";
import {
  FileText,
  Share2,
  Shield,
  AlertTriangle,
  Clock,
  TrendingUp,
} from "lucide-react";
import type { Report } from "@/lib/types";

interface ReportForgeProps {
  reports: Report[];
}

function severityColor(
  severity: "low" | "medium" | "high" | "critical"
): string {
  switch (severity) {
    case "critical":
      return "border-red-500/50 bg-red-500/10 text-red-300";
    case "high":
      return "border-orange-500/50 bg-orange-500/10 text-orange-300";
    case "medium":
      return "border-amber-500/50 bg-amber-500/10 text-amber-300";
    default:
      return "border-slate-600/50 bg-slate-600/10 text-slate-400";
  }
}

export default function ReportForge({ reports }: ReportForgeProps) {
  return (
    <section className="w-full">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-bold tracking-wider uppercase text-slate-200">
            Report Forge
          </h2>
          <p className="text-xs text-slate-500 mt-1">
            Executive Readiness Reports — auto-generated intelligence
          </p>
        </div>
        <div className="flex items-center gap-1 text-[10px] text-slate-500">
          <Shield size={12} />
          <span>CLASSIFIED</span>
        </div>
      </div>

      <div className="space-y-6">
        {reports.map((report, idx) => (
          <motion.div
            key={report.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.15 }}
            className="bg-obsidian/50 border border-slate-800 rounded-xl overflow-hidden"
          >
            {/* Report Header */}
            <div className="p-5 border-b border-slate-800">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <h3 className="text-base font-bold text-white mb-1">
                    {report.title}
                  </h3>
                  <p className="text-[10px] text-slate-500 uppercase tracking-wider">
                    Generated{" "}
                    {new Date(report.generated_at).toLocaleDateString(
                      "en-US",
                      {
                        year: "numeric",
                        month: "short",
                        day: "numeric",
                        hour: "2-digit",
                        minute: "2-digit",
                      }
                    )}
                  </p>
                </div>

                {/* Confidence Score */}
                <div className="text-center">
                  <div
                    className={`text-4xl font-mono font-black ${
                      report.confidence_score >= 80
                        ? "text-red-400"
                        : report.confidence_score >= 60
                          ? "text-hazard"
                          : "text-cobalt-light"
                    }`}
                  >
                    {report.confidence_score}%
                  </div>
                  <div className="text-[9px] text-slate-500 uppercase tracking-widest">
                    Confidence
                  </div>
                </div>
              </div>

              {/* T-Minus Countdown */}
              <div className="mt-4 flex items-center gap-3 bg-slate-900/60 rounded-lg px-4 py-2.5 border border-slate-800">
                <Clock size={16} className="text-hazard" />
                <div>
                  <div className="text-xs text-slate-400">
                    Predicted Peak Pressure
                  </div>
                  <div className="text-lg font-mono font-bold text-hazard">
                    T-{report.t_minus_hours}h
                  </div>
                </div>
                <div className="ml-auto flex items-center gap-1 text-[10px] text-slate-500">
                  <TrendingUp size={12} />
                  <span>Expected news breakout in {report.t_minus_hours} hours</span>
                </div>
              </div>
            </div>

            {/* Summary */}
            <div className="p-5 border-b border-slate-800">
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">
                Executive Summary
              </h4>
              <p className="text-sm text-slate-300 leading-relaxed">
                {report.summary}
              </p>
            </div>

            {/* Evidence Gallery */}
            <div className="p-5 border-b border-slate-800">
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3">
                Evidence Gallery — Linguistic Smoking Guns
              </h4>
              <div className="space-y-3">
                {report.evidence.map((ev) => (
                  <div
                    key={ev.id}
                    className={`border rounded-lg p-3 ${severityColor(ev.severity)}`}
                  >
                    <div className="flex items-start gap-2">
                      <AlertTriangle
                        size={14}
                        className="shrink-0 mt-0.5"
                      />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm leading-relaxed">
                          {ev.type === "quote" && (
                            <span className="italic">&ldquo;{ev.content}&rdquo;</span>
                          )}
                          {ev.type !== "quote" && ev.content}
                        </p>
                        <div className="flex items-center gap-3 mt-2 text-[10px] opacity-70">
                          <span>{ev.source}</span>
                          <span>·</span>
                          <span>
                            {new Date(ev.timestamp).toLocaleString("en-US", {
                              month: "short",
                              day: "numeric",
                              hour: "2-digit",
                              minute: "2-digit",
                            })}
                          </span>
                          <span
                            className={`uppercase font-bold tracking-wider px-1.5 py-0.5 rounded text-[8px] ${
                              ev.severity === "critical"
                                ? "bg-red-500/20"
                                : ev.severity === "high"
                                  ? "bg-orange-500/20"
                                  : "bg-slate-600/20"
                            }`}
                          >
                            {ev.severity}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Export Actions */}
            <div className="p-4 flex items-center gap-3">
              <button className="flex items-center gap-2 px-4 py-2 bg-cobalt hover:bg-cobalt-light text-white text-xs font-bold uppercase tracking-wider rounded-lg transition-colors">
                <FileText size={14} />
                PDF Export
              </button>
              <button className="flex items-center gap-2 px-4 py-2 border border-slate-700 hover:border-slate-500 text-slate-300 text-xs font-bold uppercase tracking-wider rounded-lg transition-colors">
                <Share2 size={14} />
                Secure Share Link
              </button>
            </div>
          </motion.div>
        ))}
      </div>
    </section>
  );
}
