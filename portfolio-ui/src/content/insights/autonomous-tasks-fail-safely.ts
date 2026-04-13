import type { InsightPost } from "@/types";

export const autonomousTasksFailSafely: InsightPost = {
  slug: "making-autonomous-ai-tasks-fail-safely",
  title: "Making Autonomous AI Tasks Fail Safely",
  description:
    "36 scheduled tasks running unattended — cron jobs, interval triggers, and event hooks. How I built fail-open patterns, skip-synthesis conventions, and notification delivery so things break gracefully at 3AM.",
  date: "2026-04-01",
  type: "build-log",
  tags: [
    "autonomous agents",
    "reliability",
    "fail-open patterns",
    "task scheduling",
    "production AI",
  ],
  project: "atlas",
  seoTitle: "Autonomous AI Task Patterns: Fail-Open Design for Production",
  seoDescription:
    "Build log: 36 autonomous AI tasks running on cron and interval schedules. Fail-open patterns, skip-synthesis conventions, and LLM synthesis for notification delivery.",
  targetKeyword: "autonomous ai tasks production",
  secondaryKeywords: [
    "ai task scheduling",
    "fail-open ai patterns",
    "llm autonomous agents reliability",
  ],
  content: `
<h2>The Task Inventory</h2>
<p>Atlas runs 36 autonomous tasks without human intervention:</p>

<p><strong>Cron-scheduled:</strong></p>
<ul>
  <li>Nightly memory sync (3:00 AM) — conversation turns → knowledge graph</li>
  <li>Data cleanup (3:30 AM) — expired sessions, orphaned records</li>
  <li>Pattern learning (2:00 AM) — behavioral pattern extraction</li>
  <li>Preference learning (2:30 AM) — user preference inference</li>
  <li>Device health check (6:00 AM) — IoT device status survey</li>
  <li>Proactive actions (6:30 AM) — suggested automations</li>
  <li>Morning briefing (7:00 AM) — daily summary synthesis</li>
  <li>Gmail digest (7:05 AM) — email triage and summary</li>
</ul>

<p><strong>Interval-triggered:</strong></p>
<ul>
  <li>Security summary (every 6 hours)</li>
  <li>Calendar reminder (every 5 minutes)</li>
  <li>Action escalation (every 4 hours)</li>
  <li>Anomaly detection (every 15 minutes)</li>
</ul>

<p><strong>Event-triggered:</strong></p>
<ul>
  <li>Departure auto-fix (on presence departure)</li>
  <li>B2B score calibration (weekly)</li>
</ul>

<h2>The Core Problem: LLMs Fail Silently</h2>
<p>When a traditional cron job fails, it throws an error and you get a stack trace. When an LLM-backed task fails, it might return an empty response, a hallucinated summary, or a confident-sounding answer about data it never received. Silent failure in autonomous tasks is worse than loud failure — you don't know the morning briefing was wrong until you act on it.</p>

<h2>Pattern 1: Fail-Open</h2>
<p>Every task handler is wrapped in try/except at the runner level. If a handler throws, the task is marked as failed with the exception details, but the scheduler continues. One broken task doesn't take down the others.</p>
<p>Inside handlers, the same principle applies. If the memory quality detector can't compute similarity (model not loaded), the conversation turn stores normally without quality signals. If the email provider is down, the gmail digest returns a skip signal instead of crashing.</p>

<h2>Pattern 2: Skip-Synthesis Convention</h2>
<p>Most tasks follow a two-phase pattern: (1) gather raw data, (2) synthesize with LLM. The LLM synthesis step takes 10-20 seconds and costs compute. If the raw data is empty — no new emails, no device issues, no anomalies — running the LLM is waste.</p>
<p>The convention: handlers return <code>"_skip_synthesis": "No new emails since last check"</code> in their result dict. The runner checks this field before calling the LLM. If present, the fallback message is used directly as the task result.</p>
<p>This cut parallel task execution from ~27 seconds to ~13 seconds, because empty tasks skip the LLM entirely.</p>

<h2>Pattern 3: LLM Synthesis with Domain Skills</h2>
<p>When synthesis DOES run, it's not a bare LLM call. Each task has an associated skill — a markdown document loaded at synthesis time that gives the LLM domain context. The morning briefing skill knows what a briefing should look like. The email triage skill knows how to prioritize messages.</p>
<p>The raw handler result (a dict of data) is passed to <code>_synthesize_with_skill()</code>, which loads the skill markdown, constructs a system prompt, and calls the LLM with the data as context. The LLM reasons over structured data, not raw text.</p>

<h2>Pattern 4: Notification Delivery</h2>
<p>After successful synthesis, results push to ntfy (a lightweight notification service). But not every task should notify:</p>
<ul>
  <li>Global toggle: <code>autonomous_config.notify_results</code> + <code>settings.alerts.ntfy_enabled</code></li>
  <li>Per-task opt-out: <code>task.metadata["notify"] = False</code></li>
  <li>Priority control: <code>task.metadata["notify_priority"]</code> maps to ntfy priority levels</li>
</ul>
<p>The morning briefing notifies at default priority. Anomaly detection notifies at high priority. Data cleanup doesn't notify at all.</p>

<h2>What Breaks at 3AM</h2>
<p>Real failures I've debugged:</p>
<ul>
  <li><strong>Ollama cold start:</strong> The LLM model unloads from VRAM after idle timeout. The 3AM memory sync task was the first to need it, adding 30+ seconds of model load time. Fix: the task runner pre-warms the model if synthesis is needed.</li>
  <li><strong>Database connection pool exhaustion:</strong> Multiple tasks running in parallel each grabbed connections. asyncpg's pool ran dry. Fix: shared connection pool with per-task connection limits.</li>
  <li><strong>Memory quality detector model load:</strong> The embedding model for repetition detection wasn't loaded at 3AM (loaded lazily by the intent router, which doesn't run overnight). Fix: <code>svc.is_loaded</code> guard — if the model isn't loaded, skip quality detection rather than triggering a cold load.</li>
</ul>

<h2>The Meta-Lesson</h2>
<p>Building autonomous AI tasks is 20% "make the LLM do the thing" and 80% "make sure the system is still healthy when nobody's watching." Every pattern above exists because something broke at 3AM and I had to figure out why from logs the next morning. That's the real work of production AI.</p>
`,
};
