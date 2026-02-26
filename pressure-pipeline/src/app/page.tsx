"use client";

import { useState } from "react";
import { Activity, BarChart3, Shield, Zap } from "lucide-react";
import GlobalHeatmap from "@/components/GlobalHeatmap";
import BehavioralOscilloscope from "@/components/BehavioralOscilloscope";
import BlinkRateSpeedometer from "@/components/BlinkRateSpeedometer";
import NewsLatencyMarker from "@/components/NewsLatencyMarker";
import ReportForge from "@/components/ReportForge";
import mockData from "@/data/mockData.json";
import type { SectorData, MockData } from "@/lib/types";

const data = mockData as MockData;

export default function Home() {
  const [activeSector, setActiveSector] = useState<SectorData | null>(null);

  const currentSector = activeSector ?? data.sectors[0];
  const avgStress =
    data.sectors.reduce((s, sec) => s + sec.stress_score, 0) /
    data.sectors.length;
  const criticalCount = data.sectors.filter(
    (s) => s.stress_score >= 0.8
  ).length;

  return (
    <div className="min-h-screen bg-obsidian">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-slate-800 bg-obsidian/90 backdrop-blur-md">
        <div className="max-w-[1400px] mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-hazard to-red-600 flex items-center justify-center">
              <Zap size={18} className="text-white" />
            </div>
            <div>
              <h1 className="text-base font-bold tracking-widest uppercase text-slate-100">
                Behavioral Pressure Pipeline
              </h1>
              <p className="text-[10px] text-slate-500 tracking-wider">
                ATLAS · Linguistic Disruption Intelligence
              </p>
            </div>
          </div>

          {/* Status pills */}
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-900/60 border border-slate-800 rounded-lg">
              <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
              <span className="text-[10px] text-slate-400 font-mono">
                LIVE MONITORING
              </span>
            </div>
            <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-900/60 border border-slate-800 rounded-lg">
              <Activity size={12} className="text-hazard" />
              <span className="text-[10px] text-slate-400 font-mono">
                {criticalCount} CRITICAL SECTOR{criticalCount !== 1 ? "S" : ""}
              </span>
            </div>
            <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-900/60 border border-slate-800 rounded-lg">
              <BarChart3 size={12} className="text-cobalt-light" />
              <span className="text-[10px] text-slate-400 font-mono">
                AVG STRESS: {(avgStress * 100).toFixed(0)}%
              </span>
            </div>
            <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-900/60 border border-slate-800 rounded-lg">
              <Shield size={12} className="text-slate-500" />
              <span className="text-[10px] text-slate-400 font-mono">
                CLASSIFIED
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-[1400px] mx-auto px-6 py-8 space-y-8">
        {/* Hero: Global Heatmap */}
        <GlobalHeatmap
          sectors={data.sectors}
          onSectorClick={setActiveSector}
          activeSectorId={activeSector?.id}
        />

        {/* Oscilloscope + Sidebar gauges */}
        <div className="grid grid-cols-12 gap-6">
          <div className="col-span-8">
            <BehavioralOscilloscope data={data.timeseries} />
          </div>
          <div className="col-span-4 space-y-6">
            <BlinkRateSpeedometer
              postsPerMinute={currentSector.posts_per_minute}
            />
            <NewsLatencyMarker events={data.news_events} />
          </div>
        </div>

        {/* Report Forge */}
        <ReportForge reports={data.reports} />
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800 mt-12">
        <div className="max-w-[1400px] mx-auto px-6 py-4 flex items-center justify-between text-[10px] text-slate-600">
          <span>
            Behavioral Pressure Pipeline v1.0 — Atlas Intelligence System
          </span>
          <span>Data refresh: 5s · All timestamps UTC</span>
        </div>
      </footer>
    </div>
  );
}

