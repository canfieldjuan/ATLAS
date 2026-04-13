import { ArrowDown } from "lucide-react";
import { Link } from "react-router-dom";

export function HeroSection() {
  return (
    <section className="relative min-h-[90vh] flex items-center justify-center overflow-hidden">
      {/* Background grid effect */}
      <div
        className="absolute inset-0 opacity-[0.03]"
        style={{
          backgroundImage:
            "linear-gradient(rgba(34,197,94,0.3) 1px, transparent 1px), linear-gradient(90deg, rgba(34,197,94,0.3) 1px, transparent 1px)",
          backgroundSize: "60px 60px",
        }}
      />

      {/* Gradient orbs */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary-500/10 rounded-full blur-[120px]" />
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-accent-cyan/10 rounded-full blur-[120px]" />

      <div className="relative z-10 mx-auto max-w-4xl px-6 text-center">
        <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-primary-500/30 bg-primary-500/10 px-4 py-1.5">
          <div className="h-2 w-2 rounded-full bg-primary-500 animate-pulse" />
          <span className="text-sm font-medium text-primary-400">
            AI Systems Architect
          </span>
        </div>

        <h1 className="text-5xl sm:text-6xl lg:text-7xl font-black leading-[1.1] tracking-tight">
          Building AI Systems{" "}
          <span className="text-gradient">That Actually Work</span>
        </h1>

        <p className="mt-6 text-lg sm:text-xl text-surface-200/80 max-w-2xl mx-auto leading-relaxed">
          Not chatbot demos. Not prompt engineering tutorials.{" "}
          <span className="text-white font-medium">
            Production infrastructure
          </span>{" "}
          where non-deterministic LLM outputs become deterministic, repeatable
          pipelines.
        </p>

        <div className="mt-10 flex flex-col sm:flex-row items-center justify-center gap-4">
          <Link
            to="/projects"
            className="inline-flex items-center gap-2 rounded-lg bg-primary-500 px-6 py-3 text-sm font-semibold text-surface-900 hover:bg-primary-400 transition-colors"
          >
            See the Systems
          </Link>
          <Link
            to="/framework"
            className="inline-flex items-center gap-2 rounded-lg border border-surface-700 px-6 py-3 text-sm font-semibold text-surface-200 hover:border-surface-200/50 hover:text-white transition-colors"
          >
            AI Dev Skill Framework
          </Link>
        </div>

        {/* Hard numbers */}
        <div className="mt-14 grid grid-cols-2 sm:grid-cols-4 gap-6 max-w-3xl mx-auto">
          {[
            { value: "370K", label: "Lines of Code", sub: "Python + TypeScript + React" },
            { value: "190", label: "MCP Tools", sub: "11 servers" },
            { value: "<300ms", label: "Voice Pipeline", sub: "Wake to response" },
            { value: "5", label: "NPU Models", sub: "Concurrent on 3 cores" },
          ].map((stat) => (
            <div key={stat.label} className="text-center">
              <div className="text-2xl sm:text-3xl font-black text-white">{stat.value}</div>
              <div className="text-xs font-medium text-primary-400 mt-1">{stat.label}</div>
              <div className="text-[10px] text-surface-200/40 mt-0.5">{stat.sub}</div>
            </div>
          ))}
        </div>

        <a
          href="#what-this-is"
          className="mt-10 inline-flex flex-col items-center gap-2 text-surface-200/40 hover:text-surface-200/60 transition-colors"
        >
          <span className="text-xs uppercase tracking-widest">Scroll</span>
          <ArrowDown size={16} className="animate-bounce" />
        </a>
      </div>
    </section>
  );
}
