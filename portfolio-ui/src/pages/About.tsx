import { SeoHead } from "@/components/seo/SeoHead";
import { Link } from "react-router-dom";

export default function About() {
  return (
    <>
      <SeoHead
        meta={{
          title: "About",
          description:
            "AI Systems Architect who skipped the traditional dev ladder. Building production AI systems — MCP servers, autonomous pipelines, edge compute — without writing syntax by hand.",
          canonicalPath: "/about",
        }}
      />

      <section className="py-16 px-6">
        <div className="mx-auto max-w-3xl">
          <h1 className="text-4xl font-bold text-white mb-8">About</h1>

          <div className="prose-custom space-y-6">
            <p>
              I don't fit neatly into a tier on any traditional developer
              skill chart. I can't write Python syntax from memory. I've never
              taken a CS course. But I architect and ship production AI systems
              that handle real data, real failure modes, and run unattended.
            </p>

            <p>
              My path skipped the traditional dev ladder entirely. Instead of
              learning syntax then frameworks then systems thinking, I started
              with systems thinking — and use AI as the execution layer where
              others use memorized syntax.
            </p>

            <h2>What I Actually Build</h2>

            <p>
              <Link
                to="/projects/atlas"
                className="text-primary-400 hover:underline"
              >
                Atlas
              </Link>{" "}
              started as "Hey Atlas, turn off the TV" and evolved into a
              multi-domain intelligence platform: 7 MCP servers, 100+ tools,
              a B2B churn pipeline scraping 16 review sources, autonomous
              tasks running on cron, edge compute on ARM boards, and a
              stratified reasoning engine with dual memory (Postgres + Neo4j).
            </p>

            <p>
              <Link
                to="/projects/finetunelab"
                className="text-primary-400 hover:underline"
              >
                FineTuneLab.ai
              </Link>{" "}
              is an end-to-end LLM fine-tuning platform — 225 API endpoints,
              12+ provider adapters, training pipelines with checkpoint
              management, LLM-as-Judge evaluation, and GraphRAG knowledge
              grounding.
            </p>

            <h2>The Skill I'm Trying to Name</h2>

            <p>
              There's an entire discipline between "prompt engineering" and
              "ML engineering" that barely has a name. It's the work of making
              non-deterministic AI outputs deterministic enough for production.
              Calibration loops. Fail-open patterns. Token economics. Score
              normalization. Edge/cloud architecture splits.
            </p>

            <p>
              Most AI dev content online covers the first 5% of this. The
              remaining 95% — the tedious, unglamorous, essential parts — is
              what I build every day and what this site documents.
            </p>

            <h2>Why This Matters</h2>

            <p>
              AI-first development is not "development but with Copilot." It's
              a fundamentally different discipline where the runtime is
              probabilistic, the failure modes are novel, and the traditional
              rules about deterministic code don't apply until you make them
              apply. That's the work. That's what I do.
            </p>
          </div>
        </div>
      </section>
    </>
  );
}
