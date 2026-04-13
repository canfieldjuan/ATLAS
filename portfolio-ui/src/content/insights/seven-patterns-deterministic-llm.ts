import type { InsightPost } from "@/types";

export const deterministicDeepDive: InsightPost = {
  slug: "seven-patterns-deterministic-llm-systems",
  title: "7 Patterns I Learned the Hard Way for Making LLM Systems Deterministic",
  description:
    "Every one of these patterns exists because something broke in production. Validation that catches LLM drift. Caching that poisons itself. Dedup that inflates signals. Circuit breakers that false-trigger. The lessons behind building systems where non-deterministic outputs become reliable.",
  date: "2026-04-13",
  type: "case-study",
  tags: [
    "deterministic AI",
    "production patterns",
    "LLM validation",
    "caching",
    "circuit breakers",
    "deduplication",
    "production failures",
  ],
  project: "atlas",
  seoTitle:
    "7 Patterns for Deterministic LLM Systems (Learned from Production Failures)",
  seoDescription:
    "Real production patterns for making LLM outputs deterministic: validation contracts, exact-match caching, circuit breakers, identity resolution, and the failures that taught each one.",
  targetKeyword: "deterministic llm production patterns",
  secondaryKeywords: [
    "llm output validation",
    "llm caching production",
    "ai system reliability",
  ],
  faq: [
    {
      question: "What does deterministic mean when working with LLMs?",
      answer:
        "The LLM itself is non-deterministic -- same input can produce different output. Deterministic means the system around it produces consistent, reliable results regardless. You achieve this through validation contracts, caching, scoring normalization, and fallback paths. The infrastructure is rigid; the intelligence inside it is flexible.",
    },
    {
      question: "How do you test LLM outputs in production?",
      answer:
        "You don't test the LLM output directly -- you test the contract. Does the output have the required fields? Are the values in the allowed ranges? Does the urgency score match the churn signals? Does the extracted competitor actually appear in the source text? Deterministic checks on non-deterministic output.",
    },
  ],
  content: `
<h2>These Aren't Best Practices. They're Scar Tissue.</h2>
<p>Every pattern in this post exists because something broke. Not in a demo. Not in a tutorial. In a production system processing 25,000+ reviews across 56 vendors with autonomous tasks running unattended at 3AM.</p>

<p>If you're wrapping an API and returning the response, none of this matters to you yet. If you're building systems where LLM output feeds into databases, triggers emails, generates reports, and runs without human supervision -- this is the engineering that nobody's YouTube tutorial covers.</p>

<h2>1. Validate the Contract, Not the Content</h2>

<p><strong>The failure:</strong> Early enrichment runs produced reviews where <code>urgency_score</code> was a string like "high" instead of a number. Downstream SQL queries casting <code>(enrichment->>'urgency_score')::numeric</code> silently failed. Reports showed zero urgency for vendors that were actually churning hard.</p>

<p><strong>The pattern:</strong> Every LLM output passes through a validation function before it touches the database. Not "does this look reasonable" -- does it match the contract exactly.</p>

<pre><code>def _validate_enrichment(result, source_row=None):
    if "urgency_score" not in result:
        return False

    urgency = result.get("urgency_score")

    # LLM returned a string? Try to coerce it.
    if isinstance(urgency, str):
        try:
            urgency = float(urgency)
            result["urgency_score"] = urgency
        except (ValueError, TypeError):
            return False  # Reject -- can't use this

    # Must be numeric, must be in range
    if not isinstance(urgency, (int, float)):
        return False
    if urgency < 0 or urgency > 10:
        return False

    # Boolean coercion: churn_signals fields
    # LLMs return "yes", "true", "True", 1, "1" -- normalize all of them
    signals = result["churn_signals"]
    for field in CHURN_SIGNAL_BOOL_FIELDS:
        coerced = _coerce_bool(signals.get(field))
        if coerced is None:
            return False  # Unrecognizable -- reject the whole result
        signals[field] = coerced

    return True</code></pre>

<p><strong>The lesson:</strong> The LLM will return "high" when you asked for a number. It'll return "yes" when you need a boolean. It'll return 11 on a 0-10 scale. Don't trust the output. Validate the contract. Coerce what you can, reject what you can't. This function has prevented thousands of bad records from entering the pipeline.</p>

<h2>2. Low-Fidelity Detection: Not All Enrichment Is Equal</h2>

<p><strong>The failure:</strong> Reddit posts about "switching from Copper" got enriched as high-urgency churn signals for Copper CRM. But "copper" is also a metal. Reviews from StackOverflow about "close connections" got tagged as churn signals for Close CRM. The enrichment was technically valid -- the LLM extracted what it saw -- but the context was wrong.</p>

<p><strong>The pattern:</strong> A post-enrichment fidelity check that detects when source context doesn't support the extracted signals.</p>

<pre><code>def _detect_low_fidelity_reasons(row, result):
    source = row.get("source", "").lower()
    combined = " ".join(str(row.get(f) or "") for f in
                        ("summary", "review_text", "pros", "cons"))

    reasons = []

    # Does the review actually mention the vendor?
    if source in NOISY_SOURCES:
        if not text_mentions_name(combined, row.get("vendor_name")):
            reasons.append("vendor_absent_noisy_source")

    # Ambiguous vendor names need extra commercial context
    if vendor_name_normalized in {"copper", "close"}:
        if not has_commercial_context(combined):
            reasons.append("ambiguous_vendor_no_commercial_context")

    # Technical Q&A isn't churn intelligence
    if source in {"stackoverflow", "github"}:
        if has_technical_context(combined) and not has_commercial_context(combined):
            reasons.append("technical_question_context")

    return reasons  # Non-empty = quarantine this review</code></pre>

<p><strong>The lesson:</strong> The LLM doesn't know that "copper" is both a CRM product and a metal. You do. Build that knowledge into deterministic post-processing. Every noisy source (Reddit, HackerNews, Quora, Twitter) gets a fidelity check. Reviews that fail get quarantined -- not deleted, not silently included. Quarantined. You can review them later. You can't un-corrupt a report.</p>

<h2>3. Exact-Match LLM Response Caching</h2>

<p><strong>The failure:</strong> The enrichment pipeline processed the same review multiple times when tasks restarted. Same review text, same prompt, same model -- different output each time. The non-determinism wasn't a bug; it was the LLM doing what LLMs do. But it meant we were paying for the same work twice and getting inconsistent results.</p>

<p><strong>The worse failure:</strong> We added caching. Then a malformed LLM response got cached. Every subsequent request for that vendor returned the garbled cached entry. The cache checked the entry existed but didn't validate the content. One bad response poisoned every future request.</p>

<p><strong>The pattern:</strong> A declared cache registry where every cacheable pipeline stage is explicitly registered with a namespace, mode, and rationale.</p>

<pre><code>CORE_CACHE_STRATEGIES = (
    CacheStrategy(
        stage_id="b2b_enrichment.tier1",
        mode="exact",
        namespace="b2b_enrichment.tier1",
        rationale="Single-review extraction is a stable exact-repeat workload.",
    ),
    CacheStrategy(
        stage_id="win_loss.strategy",
        mode="exact",
        namespace="win_loss.strategy",
        rationale="Win/loss strategy is deterministic given the same vendor signals.",
    ),
    # ... 14 registered stages total
)</code></pre>

<p>And the critical guard that prevents cache poisoning:</p>

<pre><code># Before storing in cache:
parsed = parse_json_response(raw, recover_truncated=True)
if isinstance(parsed, dict) and not parsed.get("_parse_fallback"):
    # Valid JSON, not a parse recovery -- safe to cache
    await store_cached_text(request, response_text=clean_output(raw))
# else: DON'T cache it. Let the next request try again.</code></pre>

<p><strong>The lesson:</strong> Caching LLM responses sounds simple until you realize you're caching non-deterministic output and treating it as deterministic truth. Two rules: (1) validate before caching, not just before using, (2) make the cache registry explicit -- if a stage isn't registered, it can't cache. No accidental caching of one-off exploratory prompts.</p>

<h2>4. Circuit Breakers for Batch LLM Pipelines</h2>

<p><strong>The failure:</strong> The enrichment repair task ran in a loop: fetch a batch of reviews, send to LLM for repair, persist results, fetch next batch. When the LLM provider had a bad hour, every batch came back as failures. The loop kept running -- fetching, failing, fetching, failing -- burning rate limits and API credits on reviews that couldn't be repaired.</p>

<p><strong>The second failure:</strong> We added a circuit breaker that stopped after 2 consecutive rounds with zero promotions. Then the Anthropic Batch API started returning "pending" for items that were still processing from a previous run. These "deferred" rounds had zero promoted, zero failed, zero shadowed -- <code>round_total = 0</code>. The circuit breaker counted them as "no progress" and tripped after 2 deferred rounds. The batch API was working fine; our circuit breaker was too aggressive.</p>

<p><strong>The pattern:</strong></p>

<pre><code># After each round:
round_total = round_promoted + round_shadowed + round_failed

# Breaker 1: High failure rate
if round_total > 0 and round_failed > round_total * failure_rate_threshold:
    circuit_breaker_reason = f"high failure rate ({round_failed}/{round_total})"
    break

# Breaker 2: No progress (but only when work was actually attempted)
if round_total > 0 and round_promoted == 0:
    consecutive_no_progress += 1
elif round_total > 0:
    consecutive_no_progress = 0

# All-deferred rounds (round_total == 0) don't count as no-progress
if consecutive_no_progress >= no_progress_max_rounds:
    circuit_breaker_reason = f"{consecutive_no_progress} consecutive rounds with no progress"
    break</code></pre>

<p><strong>The lesson:</strong> The <code>round_total > 0</code> guard is the entire lesson. A batch API that says "I'm still working on it" is not the same as "I tried and failed." If you don't distinguish between "nothing happened" and "nothing worked," your circuit breaker will fight your batch processor.</p>

<h2>5. Cross-Source Identity Resolution</h2>

<p><strong>The failure:</strong> The same review appeared on G2, Capterra, and TrustRadius. Each scrape imported it as a separate review. The enrichment pipeline processed it three times. The churn signal aggregation counted it three times. One angry customer looked like three angry customers. Competitor mentions were inflated 3x. Urgency scores were skewed. Every downstream report was wrong.</p>

<p><strong>The pattern:</strong> Three-tier deduplication at import time:</p>

<pre><code># Tier 1: Exact content match
content_hash = hash(normalize(summary + review_text + pros + cons))

# Tier 2: Identity match (same person, same day, same score)
identity_key = f"{vendor}|{reviewer_normalized}|{date}|{rating}"

# Tier 3: Fuzzy match (similar reviewer name + close date + close rating)
if reviewer_stem_overlap and date_within_tolerance and rating_within_tolerance:
    if text_similarity >= 0.82:
        mark_as_duplicate(survivor=highest_quality_source)</code></pre>

<p>Then the critical part -- every analytics query must exclude duplicates:</p>

<pre><code>WHERE duplicate_of_review_id IS NULL</code></pre>

<p>We audited 150+ queries across the codebase. Found 50+ that were missing this filter. Product profiles, blog generation, campaign trends, dashboard counts, churn alerts -- all inflated by duplicate reviews. One filter predicate, applied everywhere, fixed all of them.</p>

<p><strong>The lesson:</strong> Deduplication isn't a feature. It's a data integrity requirement that touches every query in the system. If you add it as an afterthought, you'll find it's missing in half your queries. Build it into your query helpers from day one. We have a shared <code>_canonical_review_predicate()</code> that every analytics surface calls. No exceptions.</p>

<h2>6. Tiered Processing: Not Everything Needs an LLM</h2>

<p><strong>The failure:</strong> Early Atlas sent every review through a single large LLM call that extracted everything at once: pain category, competitors, urgency, buyer role, pricing mentions, feature gaps, switching triggers. The cost was $0.05 per review. With 25,000 reviews, that's $1,250 per full enrichment pass. And 40% of reviews had no actionable signal at all.</p>

<p><strong>The pattern:</strong> Stratified processing where each tier has different cost and reliability guarantees.</p>

<pre><code>Layer 0: Raw Ingestion          -- $0, deterministic
    Scraping, dedup, import. No LLM.

Layer 1: Tiered Enrichment      -- $0.001-0.01/review
    Tier 1: Base extraction (Haiku) -- pain, competitors, urgency
    Tier 2: Deep extraction (conditional) -- only if Tier 1 found gaps

Layer 2: Signal Aggregation     -- $0, deterministic
    6 pools built from pure SQL aggregation. No LLM.
    Evidence, Segment, Temporal, Displacement, Category, Account.

Layer 3: Reasoning Synthesis    -- $0.19/vendor
    Expert LLM reasoning, but ONLY over pre-scored evidence.
    Scoped to competitive sets. Skips vendors whose data hasn't changed.

Layer 4: Downstream Artifacts   -- $0, deterministic
    Battle cards, reports, briefs. SQL templates + stored reasoning.
    No LLM calls. Just formatting precomputed intelligence.</code></pre>

<p><strong>The lesson:</strong> The most expensive mistake in AI systems is sending raw data to an LLM and asking it to figure everything out. Layer 2 (signal aggregation) does 80% of the analytical work with zero LLM cost. Layer 3 (synthesis) only runs when Layer 2 data has actually changed, verified by evidence hash comparison. The LLM is the most expensive instrument in the system -- use it only where nothing else can do the job.</p>

<h2>7. Prompt Contracts: Rules the LLM Will Break</h2>

<p><strong>The failure:</strong> The reasoning synthesis prompt had 21 rules. Rule 6 said: "When contradiction_rows are present, set confidence no higher than medium." Every vendor's packet included contradiction rows. Every vendor got "medium" confidence. The badge on every vendor in the UI showed "55%". The rule was technically correct -- contradictions existed. But the rule was too blunt. Two minor contradictions out of 50 strong witnesses still capped confidence at medium.</p>

<p><strong>The broader failure:</strong> The LLM followed some rules religiously and ignored others completely. It always obeyed "do not invent numbers" (Rule 1). It usually obeyed "every section needs confidence + data_gaps + citations" (Rule 3). It routinely ignored "omit thin segments instead of overstating them" (Rule 15) -- it would rather say something about a thin segment than leave the field empty.</p>

<p><strong>The pattern:</strong> Accept that prompt rules are probabilistic. Build deterministic post-processing for the rules that matter most.</p>

<pre><code># The prompt says: "include citations for every claim"
# The LLM sometimes forgets. So we check:
if not section.get("citations"):
    section["citations"] = extract_sids_from_section(section)
    if not section["citations"]:
        section["confidence"] = "low"  # No citations = low confidence

# The prompt says: "don't invent numbers"
# The LLM obeys this one. But we verify anyway:
for numeric_claim in extract_numeric_claims(section):
    if numeric_claim["source_id"] not in valid_source_ids:
        flag_validation_warning("ungrounded_numeric_claim", numeric_claim)</code></pre>

<p><strong>The lesson:</strong> A prompt is a suggestion. A validation function is a guarantee. Write the rules in the prompt so the LLM tries to follow them. Then write the validation so the system catches when it doesn't. The prompt shapes the distribution of outputs. The validator clips the tails. You need both.</p>

<h2>The Meta-Pattern</h2>
<p>Every one of these patterns follows the same structure:</p>
<ol>
  <li><strong>Trust the LLM to try.</strong> It's the best tool for understanding natural language, extracting structured data, and generating synthesis.</li>
  <li><strong>Don't trust the LLM to succeed.</strong> Validate the output. Coerce what you can. Reject what you can't. Cache what's valid. Circuit-break what's failing.</li>
  <li><strong>Build the deterministic shell around the non-deterministic core.</strong> The LLM is one component. The validation, caching, dedup, scoring, and circuit-breaking are the system.</li>
</ol>

<p>This is what separates an API wrapper from a production AI system. The API call is 5 lines. The infrastructure around it is 370,000.</p>
\\`,
};
