"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  AreaChart,
} from "recharts";
import type { TimeSeriesPoint } from "@/lib/types";

interface BehavioralOscilloscopeProps {
  data: TimeSeriesPoint[];
}

function CustomTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: Array<{ name: string; value: number; color: string }>;
  label?: string;
}) {
  if (!active || !payload) return null;
  return (
    <div className="bg-obsidian border border-slate-700 rounded-lg p-3 shadow-xl">
      <p className="text-xs text-slate-400 mb-2 font-mono">{label}</p>
      {payload.map((entry) => (
        <div key={entry.name} className="flex items-center gap-2 text-xs mb-1">
          <span
            className="w-2 h-2 rounded-full"
            style={{ backgroundColor: entry.color }}
          />
          <span className="text-slate-300">{entry.name}:</span>
          <span className="font-mono font-bold text-white">
            {entry.value.toFixed(2)}
          </span>
        </div>
      ))}
    </div>
  );
}

export default function BehavioralOscilloscope({
  data,
}: BehavioralOscilloscopeProps) {
  return (
    <section className="w-full">
      <div className="mb-4">
        <h2 className="text-lg font-bold tracking-wider uppercase text-slate-200">
          Behavioral Oscilloscope
        </h2>
        <p className="text-xs text-slate-500 mt-1">
          Linguistic shift tracking — Complexity · Temporal · Identity Gap
        </p>
      </div>
      <div className="bg-obsidian/50 border border-slate-800 rounded-xl p-4">
        {/* Main multi-line chart */}
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={data}>
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="rgba(100,116,139,0.15)"
            />
            <XAxis
              dataKey="timestamp"
              tick={{ fill: "#64748b", fontSize: 10 }}
              axisLine={{ stroke: "#334155" }}
            />
            <YAxis
              domain={[0, 1]}
              tick={{ fill: "#64748b", fontSize: 10 }}
              axisLine={{ stroke: "#334155" }}
              tickFormatter={(v: number) => v.toFixed(1)}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ fontSize: "11px", color: "#94a3b8" }}
              iconType="circle"
            />
            {/* Complexity Line (Prefrontal Cortex) */}
            <Line
              type="monotone"
              dataKey="complexity"
              name="Complexity (PFC)"
              stroke="#2563eb"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, fill: "#2563eb" }}
            />
            {/* Temporal Line (Survival Mode) */}
            <Line
              type="monotone"
              dataKey="temporal"
              name="Temporal Shift"
              stroke="#f97316"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, fill: "#f97316" }}
            />
            {/* Identity Gap (Us vs Them) */}
            <Line
              type="monotone"
              dataKey="identity_gap"
              name="Identity Gap"
              stroke="#ef4444"
              strokeWidth={2}
              strokeDasharray="6 3"
              dot={false}
              activeDot={{ r: 4, fill: "#ef4444" }}
            />
          </LineChart>
        </ResponsiveContainer>

        {/* Identity Gap divergence area */}
        <div className="mt-2">
          <p className="text-[10px] text-slate-500 mb-2 uppercase tracking-wider">
            Identity Divergence Zone (Us vs. Them Split)
          </p>
          <ResponsiveContainer width="100%" height={100}>
            <AreaChart data={data}>
              <defs>
                <linearGradient
                  id="identityGradient"
                  x1="0"
                  y1="0"
                  x2="0"
                  y2="1"
                >
                  <stop offset="0%" stopColor="#ef4444" stopOpacity={0.4} />
                  <stop offset="100%" stopColor="#ef4444" stopOpacity={0.05} />
                </linearGradient>
              </defs>
              <XAxis dataKey="timestamp" hide />
              <YAxis domain={[0, 1]} hide />
              <Area
                type="monotone"
                dataKey="identity_gap"
                stroke="#ef4444"
                fill="url(#identityGradient)"
                strokeWidth={1.5}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </section>
  );
}
