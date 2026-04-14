import { SeoHead } from "@/components/seo/SeoHead";
import { skillTiers, gapTable } from "@/content/skills/framework";

export default function Framework() {
  return (
    <>
      <SeoHead
        meta={{
          title: "AI Developer Skill Framework",
          description:
            "From AI-Assisted Coder to AI Systems Architect. A hierarchy mapping traditional dev seniority against AI-augmented development — what each level looks like and what skills actually matter.",
          canonicalPath: "/framework",
          jsonLd: {
            "@context": "https://schema.org",
            "@type": "Article",
            headline: "AI Developer Skill Framework",
            description:
              "A hierarchy of AI development skills from beginner to systems architect.",
            author: {
              "@type": "Person",
              name: "Juan Canfield",
            },
          },
        }}
      />

      <section className="py-16 px-6">
        <div className="mx-auto max-w-4xl">
          <header className="mb-16">
            <h1 className="text-4xl font-bold text-white mb-4">
              AI Developer Skill Framework
            </h1>
            <p className="text-surface-200/70 max-w-2xl leading-relaxed">
              Traditional dev seniority (junior to architect) mapped against
              AI-augmented development. The jump from "person who uses AI" to
              "AI-augmented developer" isn't about prompting tricks — it's
              about{" "}
              <span className="text-white font-medium">
                judgment
              </span>
              .
            </p>
          </header>

          {/* Tiers */}
          <div className="space-y-12 mb-20">
            {skillTiers.map((tier) => (
              <div
                key={tier.level}
                className="rounded-xl border border-surface-700/50 bg-surface-800/30 overflow-hidden"
              >
                {/* Tier header */}
                <div className="flex items-center gap-4 p-6 border-b border-surface-700/50">
                  <div
                    className={`h-12 w-12 rounded-xl flex items-center justify-center text-lg font-bold ${
                      tier.level === 4
                        ? "bg-gradient-to-br from-primary-500 to-accent-cyan text-surface-900"
                        : tier.level === 3
                          ? "bg-primary-500/20 text-primary-400"
                          : tier.level === 2
                            ? "bg-accent-cyan/20 text-accent-cyan"
                            : "bg-surface-700/50 text-surface-200"
                    }`}
                  >
                    L{tier.level}
                  </div>
                  <div>
                    <h2 className="text-xl font-bold text-white">
                      {tier.title}
                    </h2>
                    <p className="text-sm text-surface-200/60">
                      {tier.subtitle}
                    </p>
                  </div>
                </div>

                {/* Two-column comparison */}
                <div className="grid grid-cols-1 md:grid-cols-2 divide-y md:divide-y-0 md:divide-x divide-surface-700/50">
                  <div className="p-6">
                    <h3 className="text-xs uppercase tracking-widest text-surface-200/40 mb-4">
                      Traditional Dev
                    </h3>
                    <ul className="space-y-2">
                      {tier.traditional.map((item, i) => (
                        <li
                          key={i}
                          className="text-sm text-surface-200/70 flex items-start gap-2"
                        >
                          <span className="text-surface-200/30 mt-1">-</span>
                          {item}
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div className="p-6">
                    <h3 className="text-xs uppercase tracking-widest text-primary-400/60 mb-4">
                      AI-Augmented
                    </h3>
                    <ul className="space-y-2">
                      {tier.aiAugmented.map((item, i) => (
                        <li
                          key={i}
                          className="text-sm text-surface-200/80 flex items-start gap-2"
                        >
                          <span className="text-primary-500 mt-1">-</span>
                          {item}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>

                {/* Key skill */}
                <div className="px-6 py-4 bg-surface-700/20 border-t border-surface-700/50">
                  <p className="text-sm">
                    <span className="font-semibold text-primary-400">
                      Key differentiator:{" "}
                    </span>
                    <span className="text-surface-200/80">
                      {tier.keySkill}
                    </span>
                  </p>
                </div>
              </div>
            ))}
          </div>

          {/* Gap table */}
          <section>
            <h2 className="text-2xl font-bold text-white mb-6">
              The Translation Gaps
            </h2>
            <p className="text-surface-200/60 mb-8">
              Skills that exist in traditional development but require
              completely different thinking in AI-augmented work.
            </p>
            <div className="rounded-xl border border-surface-700/50 overflow-hidden">
              <div className="grid grid-cols-2 bg-surface-700/30 px-6 py-3 border-b border-surface-700/50">
                <span className="text-xs uppercase tracking-widest text-surface-200/40">
                  Traditional Skill
                </span>
                <span className="text-xs uppercase tracking-widest text-primary-400/60">
                  AI Equivalent Most People Miss
                </span>
              </div>
              {gapTable.map((row, i) => (
                <div
                  key={i}
                  className={`grid grid-cols-2 px-6 py-4 ${
                    i < gapTable.length - 1
                      ? "border-b border-surface-700/30"
                      : ""
                  }`}
                >
                  <span className="text-sm text-surface-200/70">
                    {row.traditional}
                  </span>
                  <span className="text-sm text-surface-200/80">
                    {row.aiEquivalent}
                  </span>
                </div>
              ))}
            </div>
          </section>
        </div>
      </section>
    </>
  );
}
