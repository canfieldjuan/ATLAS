import { Cpu, MemoryStick, Server, Thermometer, Wifi } from 'lucide-react'
import type { SystemResources } from '../types'
import { gaugeColor, gaugeTextColor } from '../utils'

export default function SystemResourcesBar({ data }: { data: SystemResources }) {
  return (
    <div className="animate-enter mb-8 rounded-xl border border-slate-800/80 bg-slate-900/40 p-4" style={{ animationDelay: '200ms' }}>
      <div className="flex flex-wrap items-center gap-6">
        {/* CPU */}
        <div className="flex min-w-[140px] flex-1 items-center gap-3">
          <Cpu className={`h-4 w-4 shrink-0 ${gaugeTextColor(data.cpu_percent)}`} />
          <div className="flex-1">
            <div className="mb-1 flex items-baseline justify-between">
              <span className="text-[10px] font-semibold uppercase tracking-widest text-slate-500">CPU</span>
              <span className={`font-mono text-xs font-medium ${gaugeTextColor(data.cpu_percent)}`}>
                {data.cpu_percent.toFixed(0)}%
              </span>
            </div>
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
              <div
                className={`h-full rounded-full transition-all duration-500 ${gaugeColor(data.cpu_percent)}`}
                style={{ width: `${Math.min(data.cpu_percent, 100)}%` }}
              />
            </div>
          </div>
        </div>

        {/* RAM */}
        <div className="flex min-w-[180px] flex-1 items-center gap-3">
          <MemoryStick className={`h-4 w-4 shrink-0 ${gaugeTextColor(data.mem_percent)}`} />
          <div className="flex-1">
            <div className="mb-1 flex items-baseline justify-between">
              <span className="text-[10px] font-semibold uppercase tracking-widest text-slate-500">RAM</span>
              <span className={`font-mono text-xs font-medium ${gaugeTextColor(data.mem_percent)}`}>
                {data.mem_used_gb.toFixed(1)}/{data.mem_total_gb.toFixed(0)} GB
              </span>
            </div>
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
              <div
                className={`h-full rounded-full transition-all duration-500 ${gaugeColor(data.mem_percent)}`}
                style={{ width: `${Math.min(data.mem_percent, 100)}%` }}
              />
            </div>
          </div>
        </div>

        {/* GPU VRAM */}
        {data.gpu && (
          <div className="flex min-w-[200px] flex-1 items-center gap-3">
            <Server className={`h-4 w-4 shrink-0 ${gaugeTextColor(data.gpu.vram_percent)}`} />
            <div className="flex-1">
              <div className="mb-1 flex items-baseline justify-between">
                <span className="text-[10px] font-semibold uppercase tracking-widest text-slate-500" title={data.gpu.name}>
                  VRAM
                </span>
                <span className={`font-mono text-xs font-medium ${gaugeTextColor(data.gpu.vram_percent)}`}>
                  {data.gpu.vram_used_gb.toFixed(1)}/{data.gpu.vram_total_gb.toFixed(0)} GB
                </span>
              </div>
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${gaugeColor(data.gpu.vram_percent)}`}
                  style={{ width: `${Math.min(data.gpu.vram_percent, 100)}%` }}
                />
              </div>
            </div>
          </div>
        )}

        {/* GPU Temp */}
        {data.gpu && (
          <div className="flex min-w-[100px] items-center gap-3">
            <Thermometer className={`h-4 w-4 shrink-0 ${gaugeTextColor(data.gpu.temperature_c, [60, 80])}`} />
            <div>
              <span className="text-[10px] font-semibold uppercase tracking-widest text-slate-500">Temp</span>
              <p className={`font-mono text-xs font-medium ${gaugeTextColor(data.gpu.temperature_c, [60, 80])}`}>
                {data.gpu.temperature_c}&deg;C
              </p>
            </div>
          </div>
        )}

        {/* Network */}
        <div className="flex min-w-[100px] items-center gap-3">
          <Wifi className="h-4 w-4 shrink-0 text-cyan-400" />
          <div>
            <span className="text-[10px] font-semibold uppercase tracking-widest text-slate-500">Net</span>
            <p className="font-mono text-xs font-medium text-cyan-400">
              {data.net_mbps.toFixed(1)} Mb/s
            </p>
          </div>
        </div>

        {/* GPU Name */}
        {data.gpu && (
          <div className="hidden text-[10px] text-slate-600 xl:block">
            {data.gpu.name}
          </div>
        )}
      </div>
    </div>
  )
}
