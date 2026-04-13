import type { SkillTier } from "@/types";

export const skillTiers: SkillTier[] = [
  {
    level: 1,
    title: "AI-Assisted Coder",
    subtitle: "Using AI as a better Stack Overflow",
    traditional: [
      "Writes code that works, sometimes",
      "Copies from docs and forums",
      "Understands syntax and basic patterns",
      "Needs code review for almost everything",
    ],
    aiAugmented: [
      "Uses AI to generate code snippets without understanding them",
      "Accepts output at face value — can't spot subtle bugs",
      "Prompts are vague: 'make a login page'",
      "Ships faster but accumulates hidden technical debt",
      "Can't distinguish between correct and confidently wrong output",
    ],
    keySkill: "Recognizing that AI output requires the same scrutiny as any code review",
  },
  {
    level: 2,
    title: "AI-Augmented Developer",
    subtitle: "Knowing what to delegate and what to verify",
    traditional: [
      "Understands systems, not just syntax",
      "Makes design decisions within bounded scope",
      "Debugs across layers (DB, API, frontend)",
      "Writes tests, handles edge cases",
    ],
    aiAugmented: [
      "Writes specific, constrained prompts with architectural context",
      "Reviews AI output like a code review — catches bad abstractions",
      "Uses AI for boilerplate acceleration, owns the architecture",
      "Understands model context limits and strengths",
      "Knows when AI is confidently wrong — verifies before shipping",
    ],
    keySkill: "Judgment about what AI can and cannot reliably produce",
  },
  {
    level: 3,
    title: "AI-Native Engineer",
    subtitle: "Orchestrating AI across systems",
    traditional: [
      "Owns entire subsystems end-to-end",
      "Makes tradeoffs (perf vs maintainability, scope vs deadline)",
      "Mentors team, sets patterns and conventions",
      "Debugs problems nobody else can",
    ],
    aiAugmented: [
      "Orchestrates AI across multiple workstreams and agents",
      "Builds reusable AI workflows, not just one-off prompts",
      "Creates system prompts and tool definitions for team productivity",
      "Uses AI to explore solution spaces: '3 approaches with tradeoffs'",
      "Knows when to throw away AI output and think from scratch",
    ],
    keySkill: "Designing the human-AI collaboration pattern, not just using the tool",
  },
  {
    level: 4,
    title: "AI Systems Architect",
    subtitle: "Building systems where AI is a first-class runtime component",
    traditional: [
      "Designs systems that evolve over years",
      "Thinks in interfaces, boundaries, and failure modes",
      "Balances business constraints with technical reality",
      "Makes expensive-to-reverse decisions correctly",
    ],
    aiAugmented: [
      "Designs production systems with AI as a core component, not an add-on",
      "Chooses between LLMs, embeddings, agents, and traditional code per use case",
      "Builds calibration loops that make non-deterministic output deterministic",
      "Architects for cost, latency, reliability, and graceful AI failure",
      "Creates composable, replaceable AI scaffolding — not hardcoded prompts",
      "Defines safety boundaries: what AI must never do in the system",
    ],
    keySkill: "Making non-deterministic AI outputs deterministic and production-ready",
  },
  {
    level: 5,
    title: "AI Platform Operator",
    subtitle: "Keeping autonomous AI systems alive in production",
    traditional: [
      "Runs infrastructure at scale — uptime, cost, capacity",
      "Debugs failures across distributed systems at 3AM",
      "Manages vendor relationships and SLAs",
      "Owns budget, headcount, and operational efficiency",
    ],
    aiAugmented: [
      "Manages cost across multiple LLM providers — routes workloads to the cheapest model that meets quality requirements",
      "Debugs silent failures in autonomous pipelines: GPU disconnects, cache poisoning, runaway cron schedules, fallback paths that quietly drain API budgets",
      "Designs circuit breakers, orphan recovery, and graceful degradation so the system self-heals before anyone notices",
      "Operates multi-node distributed inference: edge NPU, local vLLM, cloud batch API, each with different cost and latency profiles",
      "Monitors prompt contract drift — when the LLM starts returning 'medium' confidence for every vendor, that's an operational signal, not a code bug",
      "Makes the build-vs-buy decision on model hosting daily: local GPU saves money until the PCIe clip breaks, then cloud fallback saves the product",
    ],
    keySkill: "Keeping autonomous AI systems running reliably when nobody is watching",
  },
];

export const gapTable = [
  {
    traditional: "Reading documentation",
    aiEquivalent: "Understanding model cards, context limits, and token economics",
  },
  {
    traditional: "Code review",
    aiEquivalent: "Prompt output review — AI doesn't flag its own bad ideas",
  },
  {
    traditional: "Writing tests",
    aiEquivalent: "Validating AI output against edge cases it didn't consider",
  },
  {
    traditional: "System design",
    aiEquivalent: "Deciding where AI adds value vs where it adds risk",
  },
  {
    traditional: "Debugging",
    aiEquivalent: "Diagnosing whether a failure is your code, your prompt, or the model",
  },
  {
    traditional: "Performance tuning",
    aiEquivalent: "Token optimization, caching strategies, local vs cloud LLM tradeoffs",
  },
  {
    traditional: "CI/CD",
    aiEquivalent: "Evaluation pipelines, regression detection, model version management",
  },
  {
    traditional: "On-call / incident response",
    aiEquivalent: "Diagnosing whether the failure is the model, the prompt, the cache, the provider, the hardware, or the data — usually it's the one you didn't instrument",
  },
  {
    traditional: "Cost management",
    aiEquivalent: "Token economics across providers, batch API discounts, caching hit rates, and the fallback path that silently costs 10x when local inference goes down",
  },
];
