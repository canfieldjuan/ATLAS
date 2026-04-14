import type { InsightPost } from "@/types";

export const lessonModelContextNotModelProblem: InsightPost = {
  slug: "its-not-a-model-problem-its-a-context-problem",
  title: "It's Not a Model Problem — It's a Context Problem",
  description:
    "A graveyard of abandoned projects taught me one lesson: I kept blaming the model when the failure was mine. Limited systems don't produce unlimited results. Your capability to lead them is the bottleneck, not their capability to perform.",
  date: "2026-04-13",
  type: "lesson",
  tags: [
    "model selection",
    "context design",
    "tiered inference",
    "production AI",
    "AI leadership",
  ],
  project: "atlas",
  seoTitle: "It's Not a Model Problem, It's a Context Problem: Lessons from a Project Graveyard",
  seoDescription:
    "Production lesson: abandoned AI projects aren't model failures — they're design failures. Expecting unlimited results from limited systems. Why tiered model use with 3 models under one provider beats 20 scattered models.",
  targetKeyword: "ai model context problem",
  secondaryKeywords: [
    "llm model selection production",
    "tiered model inference",
    "ai project failure reasons",
  ],
  faq: [
    {
      question: "Why do AI projects fail?",
      answer:
        "Most AI project failures get blamed on the model — 'it hallucinated,' 'it wasn't smart enough,' 'it couldn't handle the task.' But the real failure is almost always design: vague prompts, missing context, no output constraints, wrong model for the task, or expecting general intelligence from a statistical text generator. The model is a subcontractor. If the subcontractor produces bad work, look at the instructions you gave them.",
    },
    {
      question: "Why use 3 models from one provider instead of 20 from many?",
      answer:
        "Consistency. Every provider has different tokenization, different system prompt handling, different tool calling formats, different failure modes. Using 20 models means debugging 20 different behavior patterns. Three models from one provider (e.g., Anthropic's Opus/Sonnet/Haiku) share the same patterns, the same strengths, the same quirks. You learn one provider deeply instead of 20 providers superficially. The result is predictable behavior across your entire system.",
    },
  ],
  content: `
<h2>The Project Graveyard</h2>
<p>Before Atlas worked, there were projects that didn't. A knowledge management system that produced summaries that sounded great but missed key details. A customer service bot that answered confidently with wrong information. An email drafting tool that wrote professional messages with fabricated specifics.</p>

<p>Every time, my diagnosis was the same: the model isn't good enough. I need a better model. I need more parameters. I need GPT-4 instead of GPT-3.5. I need Claude instead of GPT-4.</p>

<p>Every time, I was wrong.</p>

<h2>It's Not a Model Problem</h2>
<p>The model was never the bottleneck. I was. Specifically, my design was:</p>

<ul>
  <li><strong>Vague context:</strong> I gave the model a task description and expected it to figure out the domain. It did — approximately, inconsistently, differently each time.</li>
  <li><strong>No output constraints:</strong> I asked for "a summary" and got whatever the model felt like producing. Sometimes 3 sentences, sometimes 3 paragraphs. Sometimes focused, sometimes wandering.</li>
  <li><strong>Wrong scope expectations:</strong> I expected a system with a 128K context window to "understand" a 500-page knowledge base. It can read 128K tokens at once. It can't <em>understand</em> anything — it predicts the next token given what it sees.</li>
  <li><strong>Model blame instead of design blame:</strong> When the output was bad, I swapped models. The output got slightly different but not better, because the instructions were the same.</li>
</ul>

<p>The pattern is simple and it took me too long to see it: <strong>expecting a limited system to produce unlimited results is a design failure, not a model failure.</strong></p>

<h2>The General Contractor Analogy</h2>
<p>Using an LLM is no different than being a general contractor managing subcontractors. You don't hand a plumber the blueprints for the entire house and say "figure out your part." You tell them exactly which fixtures go where, what pipe diameter to use, which code to follow, and where the shutoff valves are.</p>

<p>If the plumber does bad work, you look at two things: did you hire the right sub for the job, and did you give them clear instructions? If you hired an electrician to do plumbing, that's a model selection problem. If you hired a plumber but gave them vague directions, that's a context problem.</p>

<p>Most AI failures are the second kind. The model was capable. The instructions were insufficient.</p>

<p>This translates directly to how most people use AI and how they think about it. They treat models as general intelligence — ask anything, get the right answer. But models are specialized tools that perform within the constraints you set. Your capability to lead them is the ceiling on their output quality.</p>

<h2>But That Doesn't Mean Use 20 Models</h2>
<p>Once you realize the problem is context and instruction quality, the temptation is to go the other direction: use a specialized model for every task. One model for extraction. One for classification. One for synthesis. One for summarization. Pick the best model for each job.</p>

<p>I tried this. It's worse.</p>

<p>Twenty models means twenty different behavior patterns. Different tokenization. Different system prompt handling. Different tool calling formats. Different failure modes. Different rate limits. Different latency profiles. Every model junction is a potential inconsistency. The output of Model A feeds into Model B, which was trained differently, interprets differently, and produces output that Model C downstream doesn't expect.</p>

<p>You'll never have consistent results across a system built on 20 different models. The integration tax alone — debugging why the chain broke, which model deviated, which handoff lost context — eats more engineering time than the models save.</p>

<h2>Tiered Model Use: 3 Models, One Provider</h2>
<p>What actually works: pick one provider and use their model tiers.</p>

<p>Atlas uses Anthropic as the primary provider with three tiers:</p>
<ul>
  <li><strong>Haiku</strong> — fast, cheap, good enough for classification, structured extraction, simple routing decisions</li>
  <li><strong>Sonnet</strong> — balanced, handles most reasoning tasks, campaign drafting, report synthesis</li>
  <li><strong>Opus</strong> — heavyweight, reserved for complex cross-vendor reasoning, battle card generation, tasks where nuance directly impacts quality</li>
</ul>

<p>Three models. Same provider. Same API format. Same tool calling spec. Same tokenizer. Same behavioral patterns. When Haiku produces unexpected output, the debugging process is the same as when Opus does — because they share a foundation.</p>

<p>Local models (Qwen3 via Ollama/vLLM) handle the high-volume batch work where cost matters more than peak quality. But the design principle holds: minimize provider diversity, maximize familiarity with fewer models.</p>

<h2>Consistency Is the Product</h2>
<p>At the end of the pipeline, a churn signal or a battle card or a campaign email is only useful if the user can trust it. Trust comes from consistency. If the system produces great output 90% of the time and subtly wrong output 10% of the time, users learn to distrust all of it.</p>

<p>Consistency doesn't come from using the "best" model. It comes from:</p>
<ul>
  <li><strong>Clear constraints</strong> on every LLM call — what to extract, what format, what values are valid</li>
  <li><strong>Same behavioral patterns</strong> across the system — one provider, tiered by capability</li>
  <li><strong>Validation layers</strong> that catch the 10% before it reaches the user</li>
  <li><strong>Your skill as the GC</strong> — knowing which sub to call for which job, and writing instructions they can't misinterpret</li>
</ul>

<p>The abandoned projects in my graveyard didn't fail because the models were bad. They failed because I hadn't learned to lead them yet. That's the actual skill gap in AI development — not prompting tricks, not model selection, but the ability to design systems where limited tools produce reliable results.</p>
`,
};
