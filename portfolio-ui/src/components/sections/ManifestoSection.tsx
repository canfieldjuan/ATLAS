import { AlertTriangle, CheckCircle, Zap } from "lucide-react";

const points = [
  {
    icon: AlertTriangle,
    title: "The problem with AI dev content",
    body: 'Most tutorials stop at "call the API and print the response." YouTube is full of chatbot demos, wrapper apps, and prompt engineering tips. Almost nobody talks about what happens when you need that LLM output to be reliable enough to push to a CRM, trigger a webhook, or run unattended at 3AM.',
    color: "text-amber-400",
  },
  {
    icon: Zap,
    title: "The gap between chatbot coding and production AI",
    body: "There's an entire discipline between prompting and production that barely has a name. Calibration loops. Fail-open patterns. Archetype classification. Score normalization. Token economics. Edge/cloud splits. Making non-deterministic outputs deterministic isn't a prompt trick — it's systems engineering.",
    color: "text-accent-cyan",
  },
  {
    icon: CheckCircle,
    title: "What this site actually shows",
    body: "Real systems. Real decisions. The tedious parts — debugging a tool call that silently drops context, figuring out why your autonomous task worked 99 times and failed on the 100th, choosing between local inference at $0 and cloud inference at $0.003/call. This is what AI-first development actually looks like.",
    color: "text-primary-400",
  },
];

export function ManifestoSection() {
  return (
    <section id="what-this-is" className="py-24 px-6">
      <div className="mx-auto max-w-4xl">
        <h2 className="text-3xl font-bold text-white mb-12 text-center">
          Why This Exists
        </h2>

        <div className="space-y-8">
          {points.map((point) => (
            <div
              key={point.title}
              className="flex gap-4 p-6 rounded-xl border border-surface-700/50 bg-surface-800/30"
            >
              <point.icon
                className={`${point.color} mt-1 flex-shrink-0`}
                size={24}
              />
              <div>
                <h3 className="text-lg font-semibold text-white mb-2">
                  {point.title}
                </h3>
                <p className="text-surface-200/80 leading-relaxed">
                  {point.body}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
