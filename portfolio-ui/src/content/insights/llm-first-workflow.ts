import type { InsightPost } from "@/types";

export const llmFirstWorkflow: InsightPost = {
  slug: "replacing-state-machine-with-llm-tools",
  title: "Replacing a 1,000-Line State Machine with 150 Lines of LLM + Tools",
  description:
    "How I replaced a rigid LangGraph state machine with a system prompt, 3 tools, and an execution loop — cutting 85% of the code while making the system more flexible.",
  date: "2026-03-15",
  type: "build-log",
  tags: [
    "LLM tool calling",
    "state machines",
    "workflow design",
    "refactoring",
    "production patterns",
  ],
  project: "atlas",
  seoTitle:
    "LLM Tool Calling vs State Machines: A Production Refactoring Case Study",
  seoDescription:
    "Build log: replacing a 1,000-line LangGraph booking workflow with 150 lines of LLM + tool calling. What worked, what broke, and why the tedious parts matter.",
  targetKeyword: "llm tool calling production",
  secondaryKeywords: [
    "langgraph vs tool calling",
    "llm workflow patterns",
    "ai first application design",
  ],
  faq: [
    {
      question: "When should you use a state machine vs LLM tool calling?",
      answer:
        "Use state machines when the workflow is truly linear and deterministic — form wizards, checkout flows. Use LLM + tools when the conversation can branch unpredictably, when slot filling order doesn't matter, or when you need natural language understanding of user intent. The booking workflow had too many edge cases for rigid routing.",
    },
    {
      question: "How do you detect when the LLM workflow is complete?",
      answer:
        "Check if the final tool (e.g., 'book_appointment') appeared in the tools_executed list after the LLM response. If the LLM returns an empty response without calling the completion tool, keep the workflow alive with a fallback prompt. Don't rely on the LLM saying 'done' — check the tool execution log.",
    },
  ],
  content: `
<h2>The Before: 1,000+ Lines of Rigid Routing</h2>
<p>The Atlas booking workflow started as a LangGraph state machine. It had regex patterns for extracting dates, times, and service types. It had conditional edges for every conversation branch. It had template strings for every response. It worked — until it didn't.</p>

<p>The problems were predictable in hindsight:</p>
<ul>
  <li>Every new service type required new regex patterns and routing edges</li>
  <li>Users who said things slightly differently ("next Tuesday" vs "this coming Tuesday" vs "Tuesday the 15th") broke the date extraction</li>
  <li>The conversation felt robotic — because it was following a rigid graph, not understanding intent</li>
  <li>Maintaining 1,000+ lines of routing logic for a booking flow was absurd</li>
</ul>

<h2>The After: System Prompt + 3 Tools + Execution Loop</h2>
<p>The replacement uses <code>execute_with_tools()</code> with a <code>tools_override</code> parameter. The entire workflow is:</p>
<ul>
  <li>A system prompt describing the booking agent's role and available services</li>
  <li>3 tools: <code>lookup_availability</code>, <code>lookup_customer</code>, <code>book_appointment</code></li>
  <li>Conversation context persistence via <code>WorkflowStateManager</code></li>
  <li>Completion detection: check if <code>book_appointment</code> was in <code>tools_executed</code></li>
</ul>

<p>~150 lines. The LLM handles natural language understanding, slot filling in any order, clarification questions, and conversational tone. The tools handle the deterministic parts — checking the calendar, looking up customers, creating the booking.</p>

<h2>The Tedious Parts Nobody Talks About</h2>

<h3>Tool XML Stripping</h3>
<p>Local LLMs (Qwen3 via Ollama) wrap tool calls in XML tags: <code>&lt;tool_call&gt;</code>, <code>&lt;function=...&gt;</code>, sometimes inside <code>&lt;think&gt;</code> blocks. The <code>_strip_tool_xml()</code> function in tool_executor.py handles all these formats on ALL return paths. Miss one path and you get raw XML in the user's response.</p>

<h3>Empty Response Handling</h3>
<p>Sometimes the LLM returns nothing — no text, no tool call. In a state machine, this crashes. In the tool-calling pattern, you need a convention: empty response without completion = keep workflow alive with "Sorry, could you repeat that?" This sounds trivial but it's the difference between a robust system and one that randomly hangs.</p>

<h3>Parameter Type Mapping</h3>
<p>The tool registry maps <code>"int"</code> to JSON Schema <code>"integer"</code>. If you register a tool with <code>param_type: "integer"</code>, it silently fails because Ollama's Modelfile template uses Go's capitalized field names (<code>$prop.Type</code>, <code>$prop.Description</code>). Getting these two type systems to agree cost more debugging time than writing the actual workflow logic.</p>

<h3>Cancel Pattern Pass-Through</h3>
<p>Users say "cancel my appointment" during a booking flow. The booking workflow has <code>_CANCEL_PATTERNS</code> that detect appointment-keyword cancellations and pass them through to the LLM instead of treating them as booking intent. Without this, saying "cancel" during booking starts a new cancellation workflow inside the booking workflow.</p>

<h2>What This Pattern Teaches About AI-First Design</h2>
<p>The state machine treated the LLM as a text generator inside a deterministic frame. The tool-calling pattern inverts this: the LLM is the orchestrator, and the tools are the deterministic anchors.</p>

<p>This is the fundamental design shift in AI-first applications. You don't put AI inside your architecture — you put your architecture inside AI, with deterministic escape hatches (tools) at every point where you need reliability.</p>

<p>The 85% code reduction is nice. But the real win is that the system now handles conversation patterns I never explicitly coded for, because the LLM generalizes where the regex couldn't.</p>
`,
};
