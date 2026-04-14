import type { InsightPost } from "@/types";

export const lessonPromptingRagGrounding: InsightPost = {
  slug: "prompting-is-a-science-rag-is-harder-than-you-think",
  title: "Prompting Is a Science, RAG Is Harder Than You Think, and Models Will Confidently Lie to You",
  description:
    "Model reasoning is powerful — but only when directed. Left to its own devices it produces plausible-sounding information that's slightly off. Prompting Claude Code is not prompt engineering. And RAG is far more complex than bolting a vector DB onto a model.",
  date: "2026-04-13",
  type: "lesson",
  tags: [
    "prompt engineering",
    "RAG",
    "grounding",
    "model reasoning",
    "context window",
    "production AI",
  ],
  project: "atlas",
  seoTitle: "Prompt Engineering vs Prompting, RAG Complexity, and Model Grounding in Production",
  seoDescription:
    "Production lessons: model reasoning needs direction or it drifts. Prompting is a science, not just typing instructions. RAG is far more than a vector DB. Context fills fast. Data retrieval needs validation.",
  targetKeyword: "prompt engineering production rag",
  secondaryKeywords: [
    "rag complexity production",
    "llm grounding",
    "prompt engineering vs prompting",
    "context window management",
  ],
  faq: [
    {
      question: "What's the difference between prompting and prompt engineering?",
      answer:
        "Prompting is typing instructions into a chat window. Prompt engineering is designing structured instructions that constrain model behavior across thousands of invocations — field contracts, output schemas, role definitions, few-shot examples, and guardrails that prevent drift. One is a conversation; the other is a specification.",
    },
    {
      question: "Why is RAG harder than it looks?",
      answer:
        "The demo version is easy: embed documents, query a vector DB, stuff results into the prompt. The production version requires chunk size tuning, relevance scoring, source attribution, context budget management, retrieval validation (did we actually get useful results?), and grounding checks (did the model use the retrieved context or hallucinate anyway?). Most RAG failures aren't retrieval failures — they're grounding failures.",
    },
  ],
  content: `
<h2>Model Reasoning Is Powerful — But Only If Directed</h2>
<p>There's a seductive idea that you can point an LLM at a problem, say "figure it out," and get a good answer. And sometimes you do. But in production, "sometimes" is the enemy. What you actually get is information that <em>sounds</em> good but is slightly off — confidently stated, well-structured, and wrong in ways that are hard to catch without domain expertise.</p>

<p>The model isn't lying. It's doing exactly what it's designed to do: produce plausible text. "Plausible" and "correct" overlap most of the time, which is what makes the gaps so dangerous. You trust the output because 95% of it is right, and the 5% that's wrong is phrased with the same confidence as the 95%.</p>

<p>Undirected reasoning is a liability. Directed reasoning — where you tell the model exactly what to extract, what format to use, what values are acceptable, and what to do when it's uncertain — is where the value lives. The difference is the prompt.</p>

<h2>Prompting Is a Science and an Art</h2>
<p>This is the lesson that took the longest to internalize: models interpret ambiguity. If your instructions aren't airtight, the model will fill in the gaps with its best guess. And its best guess changes between runs, between contexts, between model versions.</p>

<p>Examples of what "not solid instructions" produces:</p>
<ul>
  <li><strong>Vague:</strong> "Extract the pain points from this review" → Model invents pain categories that don't match your taxonomy. Every run uses slightly different labels.</li>
  <li><strong>Directed:</strong> "Extract pain points. Use ONLY these categories: pricing_complaint, feature_gap, support_quality, integration_issue, performance, migration_difficulty, ux_complexity. If none apply, return empty array." → Consistent, filterable, queryable output.</li>
</ul>

<p>The difference isn't cleverness — it's constraint. Good prompt engineering removes degrees of freedom until the model can only produce what you need.</p>

<h3>Prompting Claude Code is NOT prompt engineering</h3>
<p>This distinction matters and most people conflate them. When you prompt Claude Code, you're having a conversation with a capable assistant. You can be loose, contextual, iterative. The model has your codebase, your conversation history, and tools to verify its work.</p>

<p>Prompt engineering for a production pipeline is the opposite. The prompt runs thousands of times against different inputs with no conversation history. It must produce structured output that downstream systems parse mechanically. There's no human to catch a bad response. Every ambiguity is a bug. Every optional field the model decides to skip is a null pointer exception three stages downstream.</p>

<p>These are fundamentally different disciplines. Being good at one doesn't make you good at the other. I learned this by failing — shipping prompts that worked great in testing (conversational, iterative) and broke in production (batch, unsupervised, no second chances).</p>

<h2>RAG Is More Complex Than a Vector DB and a Model</h2>
<p>The tutorial version of RAG:</p>
<ol>
  <li>Embed your documents into a vector database</li>
  <li>When the user asks a question, embed the query</li>
  <li>Find the nearest document chunks</li>
  <li>Stuff them into the prompt</li>
  <li>Ask the model to answer</li>
</ol>

<p>This works in demos. In production, every step has failure modes that the demo doesn't show.</p>

<h3>Context fills fast</h3>
<p>Retrieve 10 relevant chunks at 500 tokens each and you've used 5,000 tokens of context before the model even starts reasoning. Add the system prompt, conversation history, and the actual query, and you're at 8,000-10,000 tokens of input. The model's reasoning happens in whatever context space is left. Stuff too much retrieval and the model loses the thread.</p>

<p>This means you need a context budget. Not "retrieve everything relevant" — retrieve the <em>most</em> relevant items that fit within a token ceiling. That requires relevance scoring, not just vector similarity. A chunk can be semantically similar to the query and still be useless for answering it.</p>

<h3>Retrieval needs validation</h3>
<p>Did you actually retrieve useful results? Vector similarity doesn't guarantee relevance. A query about "Salesforce pricing complaints" might retrieve chunks about "Salesforce integration features" because the word "Salesforce" dominates the embedding. The retrieval looks successful (high similarity score) but the content doesn't answer the question.</p>

<p>In Atlas, the RAG client validates retrieved sources: semantic search plus temporal filtering, source tracking with structured <code>SearchSource[]</code> objects, and a <code>gather_context()</code> function that manages the retrieval-to-prompt pipeline with pre-fetched source support for both text and voice paths.</p>

<h3>Grounding is the real problem</h3>
<p>Even with good retrieval, the model might ignore the retrieved context and answer from its training data. This is the grounding problem — you gave the model evidence, but it went with its gut instead.</p>

<p>You need to detect this. If the model's answer doesn't reference the retrieved sources, it's probably hallucinating. If the answer contradicts the sources, it's definitely hallucinating. The witness system in the churn pipeline exists precisely for this: verifying that evidence spans actually appear in source material before any intelligence reaches a report.</p>

<h3>Dual memory is not optional at scale</h3>
<p>Atlas uses PostgreSQL for structured conversation data (turns, metadata, quality signals) and Neo4j for the knowledge graph (entities, relationships, episodic traces). These serve different retrieval patterns:</p>
<ul>
  <li><strong>Postgres:</strong> "What did the user say in the last 5 conversations?" — temporal, structured, exact</li>
  <li><strong>Neo4j:</strong> "What do we know about this vendor's relationship to this competitor?" — relational, semantic, connected</li>
</ul>
<p>Trying to serve both patterns from a single vector DB produces bad results for both. The RAG client unifies retrieval across both stores, but the stores themselves are purpose-built for their access patterns.</p>

<h2>All Learned by Failing</h2>
<p>None of this came from reading documentation. It came from:</p>
<ul>
  <li>Shipping a prompt that worked in testing and produced garbage in batch processing</li>
  <li>Watching RAG retrieval return high-similarity, low-relevance chunks that the model dutifully cited as evidence</li>
  <li>Discovering that 10 retrieved chunks left the model no room to reason, producing worse answers than 3 chunks</li>
  <li>Finding that the model ignored retrieved context entirely when the question was "easy enough" to answer from training data — producing a confident, outdated, wrong answer</li>
  <li>Realizing that the knowledge graph and the conversation store needed different databases because their query patterns are fundamentally different</li>
</ul>
<p>Each failure was a lesson in the gap between "this works in a demo" and "this works at scale, unattended, thousands of times." That gap is where production AI engineering lives.</p>
`,
};
