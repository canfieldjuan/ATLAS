import { Link } from 'react-router-dom'
import { useEffect } from 'react'
import { ArrowLeft } from 'lucide-react'
import PublicLayout from '../components/PublicLayout'

export default function Methodology() {
  useEffect(() => {
    document.title = 'Our Methodology | Churn Signals'
  }, [])

  return (
    <PublicLayout>
      <article className="max-w-3xl mx-auto px-6 pt-12 pb-24">
        <Link
          to="/blog"
          className="inline-flex items-center gap-1.5 text-sm text-slate-400 hover:text-white transition-colors mb-8"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to blog
        </Link>

        <h1 className="text-3xl sm:text-4xl font-bold leading-tight mb-8">Our Methodology</h1>

        <div className="blog-prose">
          <h2>How We Collect Data</h2>
          <p>
            Churn Signals aggregates public user reviews from B2B software review platforms
            including G2, Capterra, TrustRadius, GetApp, Software Advice, and community sources
            like Reddit and Hacker News. Reviews are collected programmatically and cross-referenced
            for deduplication.
          </p>
          <p>
            We do not solicit, incentivize, or fabricate reviews. All data comes from publicly available
            sources. Verified review platforms (G2, Capterra, etc.) confirm reviewer identity;
            community sources (Reddit, HN) do not. Both are included in our analysis with appropriate
            weighting (see Data Source Weighting below).
          </p>

          <h2>What Is a "Churn Signal"?</h2>
          <p>
            A churn signal is a review or review excerpt where a user explicitly mentions switching away
            from a product, actively evaluating alternatives, planning to cancel, or expressing intent to
            leave. We identify churn signals through keyword matching and NLP classification, then manually
            validate a sample for accuracy.
          </p>
          <p>
            Not every negative review is a churn signal. A user complaining about a feature but staying on
            the platform is a pain point, not a churn signal. We distinguish between dissatisfaction
            (complaints) and intent to leave (churn).
          </p>

          <h2>Urgency Scoring</h2>
          <p>
            Each churn signal receives an urgency score from 0 to 10 based on the language intensity
            and specificity of the switching intent:
          </p>
          <ul>
            <li><strong>0-2:</strong> Mild dissatisfaction, passing mention of frustration</li>
            <li><strong>3-4:</strong> Active dissatisfaction, considering alternatives casually</li>
            <li><strong>5-6:</strong> Evaluating competitors, comparing specific options</li>
            <li><strong>7-8:</strong> Decision made, planning migration or in active evaluation</li>
            <li><strong>9-10:</strong> Already switched or in final stages of leaving</li>
          </ul>
          <p>
            Urgency scores are assigned by an LLM classifier and spot-checked for calibration.
            Average urgency for a vendor reflects the intensity of dissatisfaction across its churn signals.
          </p>

          <h2>Product Profiles</h2>
          <p>
            Vendor profiles (strengths, weaknesses, pain points, use-case fit) are synthesized from the
            full review corpus — not just churn signals. This provides a balanced view: what a product
            does well and where it falls short.
          </p>

          <h2>Review Enrichment</h2>
          <p>
            Raw reviews are enriched with structured metadata including:
          </p>
          <ul>
            <li>Pain categories (pricing, features, support, UX, reliability, integrations)</li>
            <li>Company size indicators (when available from reviewer profile)</li>
            <li>Competitor mentions and migration direction</li>
            <li>Sentiment classification</li>
          </ul>

          <h2>Data Source Weighting</h2>
          <p>
            Not all review sources carry equal reliability. We categorize sources into two tiers
            and weight them accordingly in our analysis:
          </p>
          <ul>
            <li>
              <strong>Verified platforms (weight: 0.8-1.0):</strong> G2, Capterra, Gartner Peer Insights,
              TrustRadius, PeerSpot, GetApp, and Software Advice. These platforms verify reviewer identity
              and/or employment, providing higher confidence that the reviewer has actual product experience.
            </li>
            <li>
              <strong>Community sources (weight: 0.4-0.6):</strong> Reddit, Hacker News, Twitter/X, forums,
              and blog comments. These sources provide valuable signal but lack identity verification.
              Community feedback tends to be more candid but may include users with limited product exposure.
            </li>
          </ul>
          <p>
            Source weighting affects urgency score aggregation and pain category ranking. A pricing complaint
            from a verified G2 reviewer carries more weight than the same complaint from an anonymous Reddit post.
            However, community sources often surface issues that verified platforms miss (e.g., migration pain,
            integration problems), so we include both in our analysis.
          </p>
          <p>
            Every blog post includes a source distribution breakdown so readers can assess the data foundation
            for themselves.
          </p>

          <h2>Deduplication</h2>
          <p>
            Reviews are deduplicated by source URL and content hash. If the same review appears on
            multiple platforms, we count it once. Cross-posted reviews are flagged and consolidated.
          </p>

          <h2>Limitations</h2>
          <ul>
            <li>
              <strong>Selection bias:</strong> People who write reviews skew toward strong opinions
              (very happy or very frustrated). Our data overrepresents extreme experiences.
            </li>
            <li>
              <strong>Recency bias:</strong> Our analysis windows are typically 3-12 months.
              Older product issues may be resolved; newer issues may not yet appear in reviews.
            </li>
            <li>
              <strong>Review volume varies:</strong> Popular products (Salesforce, HubSpot) have
              thousands of reviews. Niche products may have dozens. Conclusions drawn from small
              samples are less reliable.
            </li>
            <li>
              <strong>LLM classification:</strong> Urgency scoring and pain categorization use
              AI classification, which has error rates. We spot-check but do not manually verify
              every classification.
            </li>
          </ul>

          <h2>Affiliate Relationships</h2>
          <p>
            Some articles contain affiliate links. When you purchase through these links, we may earn
            a commission at no additional cost to you.
          </p>
          <p>
            <strong>Our editorial process is independent of affiliate relationships.</strong> We analyze
            and report on vendors regardless of whether we have an affiliate partnership with them.
            Affiliate partners receive the same critical treatment as non-partners — if the data shows
            problems, we report them.
          </p>
          <p>
            Articles containing affiliate links are marked with a disclosure notice at the top of the page.
          </p>

          <h2>Contact</h2>
          <p>
            Questions about our methodology? Reach us at{' '}
            <a href="mailto:outreach@atlasbizintel.co" className="text-cyan-400 hover:text-cyan-300">
              outreach@atlasbizintel.co
            </a>.
          </p>
        </div>
      </article>
    </PublicLayout>
  )
}
