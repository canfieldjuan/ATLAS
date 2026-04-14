import type { InsightPost } from "@/types";

export const lessonTestingLlmSystems: InsightPost = {
  slug: "testing-llm-systems-is-expensive",
  title: "Testing LLM Systems Is Expensive and Every Test Is a Shot in the Dark",
  description:
    "You can't unit test a probabilistic system the way you test deterministic code. Every test run costs real tokens, results aren't reproducible, and the failure modes are novel. Here's how testing actually works in production AI.",
  date: "2026-04-11",
  type: "lesson",
  tags: [
    "testing",
    "LLM evaluation",
    "non-determinism",
    "production AI",
    "cost of testing",
  ],
  project: "atlas",
  seoTitle: "Testing LLM Systems: Why Every Test Is a Shot in the Dark",
  seoDescription:
    "Production lesson: testing AI systems costs real money, results aren't reproducible, and traditional testing patterns break down. How to build confidence in a probabilistic system.",
  targetKeyword: "testing llm systems production",
  secondaryKeywords: [
    "llm evaluation",
    "ai testing strategy",
    "non-deterministic testing",
  ],
  faq: [
    {
      question: "How do you test a system where the same input gives different outputs?",
      answer:
        "You don't test the output — you test the constraints around it. Does the JSON schema validate? Are all required fields present? Is the confidence score within expected range? Does the downstream consumer handle the output correctly? You build validation layers around the LLM, not assertion tests on its output.",
    },
    {
      question: "How much does testing an LLM pipeline cost?",
      answer:
        "Every test run through the enrichment pipeline costs real tokens. A full regression test across 100 sample reviews at cloud API rates can cost $10-30. Run that 5 times while debugging a prompt change and you've spent $50-150 just testing. Local inference reduces the per-run cost to near-zero but adds latency and potentially misses quality regressions that only show up with frontier models.",
    },
  ],
  content: `
<h2>The Problem</h2>
<p>Traditional testing is built on a simple premise: given the same input, you should get the same output. Write a test, assert the result, done. LLMs destroy this premise entirely.</p>

<p>Run the same review through enrichment twice and you'll get different pain categories, different confidence scores, sometimes different competitor mentions. The extraction is correct both times — just differently correct. How do you write a test for that?</p>

<h2>What Doesn't Work</h2>

<h3>Exact output matching</h3>
<p>Obvious but worth stating: you cannot assert that the LLM returns exactly "pricing_complaint" as the pain category. It might return "cost_concerns" or "pricing_issues" and all three are valid.</p>

<h3>Snapshot testing</h3>
<p>"Record the output once and compare future runs." This breaks immediately because the output changes every run. You'd be updating snapshots constantly.</p>

<h3>Mock the LLM</h3>
<p>This is the most common mistake. You mock the LLM in tests to make them deterministic, and now your tests pass but they're not testing the thing that actually breaks. The LLM <em>is</em> the system. Mocking it is like testing a car by mocking the engine.</p>

<h2>What Actually Works</h2>

<h3>Schema validation testing</h3>
<p>Don't test what the LLM says — test that what it says <em>fits the contract</em>. Does the output parse as valid JSON? Are all required fields present? Is the pain_category one of the 12 canonical values? Is the confidence_score between 0 and 1? These tests are deterministic even when the output isn't.</p>

<h3>Constraint testing</h3>
<p>Given a review that clearly mentions Salesforce, does the output include "salesforce" in competitor_mentions? Given a review with explicit praise, is the sentiment not "negative"? These are loose constraints that accommodate variation while catching obvious failures.</p>

<h3>Statistical testing over batches</h3>
<p>Run 50 reviews through the pipeline. Are pain categories distributed roughly as expected? Is the average confidence score in a reasonable range? Does at least 80% of the batch produce valid output? This catches systematic regressions without requiring per-item determinism.</p>

<h3>Validation layers, not assertion tests</h3>
<p>The real insight: you don't test the LLM — you build <em>production validation</em> that runs on every real invocation. Field ownership contracts, witness verification, confidence thresholds. If the enrichment output doesn't meet quality gates, it gets re-queued for repair. The validation IS the test, running continuously in production.</p>

<h2>The Cost Problem</h2>
<p>Every test costs money. Cloud API: $0.01-0.10 per review enrichment. Run a 100-review regression suite 5 times while iterating on a prompt and you've spent $25-50. That's just one pipeline stage.</p>

<p>Local inference helps — near-zero per-token cost. But local models might not reproduce the behavior you're trying to test. A bug that appears with Claude might not appear with Qwen3. So you end up needing to test on both.</p>

<p>Each test is genuinely a shot in the dark. You're not verifying a known-good state — you're sampling from a probability distribution and hoping the sample represents the population. This is fundamentally different from deterministic testing and most developers coming from traditional codebases never adjust their mental model.</p>

<h2>The Mental Shift</h2>
<p>Stop trying to make LLM tests behave like unit tests. Instead:</p>
<ul>
  <li><strong>Test the infrastructure deterministically:</strong> Does the pipeline handle malformed LLM output? Does retry logic work? Does the repair queue pick up failed items?</li>
  <li><strong>Test the LLM statistically:</strong> Over N runs, is quality above threshold? Are failure rates below threshold?</li>
  <li><strong>Test the contracts continuously:</strong> Every production run validates field ownership, schema compliance, and confidence gates. The production system IS the test harness.</li>
</ul>

<p>This means accepting that you will never have 100% test coverage of LLM behavior. You can have 100% coverage of the infrastructure around it. That's where your confidence comes from.</p>
`,
};
