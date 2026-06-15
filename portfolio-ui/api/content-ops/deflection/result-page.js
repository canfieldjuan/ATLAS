import { loadDeflectionReport } from "./atlas-report.js";
import { CHECKOUT_SOURCE, resultPath } from "./checkout.js";

const REQUEST_ID_RE = /^[A-Za-z0-9][A-Za-z0-9_.:-]{5,160}$/;
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const LIGHT_REPEAT_TICKET_THRESHOLD = 10;
const ASSISTED_CONTACT_COST = 13.5;
const PAID_QUESTION_LIMIT = 8;
const PAID_DETAIL_LIMIT = 5;
const PAID_PHRASE_LIMIT = 10;
const RESOLUTION_EVIDENCE_STATUS = "resolution_evidence";
const EVIDENCE_EXPORT_SCHEMA_VERSION = "deflection_evidence.v1";

function clean(value) {
  if (Array.isArray(value)) return clean(value[0]);
  return typeof value === "string" ? value.trim() : "";
}

function escapeHtml(value) {
  return clean(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function scriptJson(value) {
  return JSON.stringify(clean(value))
    .replace(/</g, "\\u003c")
    .replace(/\u2028/g, "\\u2028")
    .replace(/\u2029/g, "\\u2029");
}

function requestUrl(req) {
  const proto = clean(req.headers?.["x-forwarded-proto"]) || "https";
  const host = clean(req.headers?.["x-forwarded-host"]) || clean(req.headers?.host) || "localhost";
  return new URL(req.url || "/", `${proto}://${host}`);
}

function requestIdFrom(req, url) {
  const queryId = clean(req.query?.request_id) || clean(url.searchParams.get("request_id"));
  if (queryId) return queryId;
  const match = url.pathname.match(/\/services\/faq-deflection\/results\/([^/?#]+)/);
  return match ? decodeURIComponent(match[1]) : "";
}

function formatNumber(value) {
  return Number.isFinite(value) ? String(value) : "0";
}

function finiteCount(value) {
  return Number.isFinite(value) && value >= 0 ? value : 0;
}

function parsedCount(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) && numeric >= 0 ? numeric : 0;
}

function firstParsedCount(...values) {
  for (const value of values) {
    if (value === undefined || value === null || value === "") continue;
    return parsedCount(value);
  }
  return 0;
}

function formatMoney(value) {
  const numeric = Number.isFinite(value) && value >= 0 ? value : 0;
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(numeric);
}

function isObjectRecord(value) {
  return value && typeof value === "object" && !Array.isArray(value);
}

function customerWordingExamples(questions) {
  return questions
    .map((question) => clean(question && question.customer_wording))
    .filter((phrase) => phrase.length > 0)
    .slice(0, 5);
}

function renderCustomerWordingCard(questions) {
  const examples = customerWordingExamples(questions);
  return `<div class="customer-wording-card">
            <div class="customer-wording-header">
              <p>Customer wording</p>
              <span>Actual ticket phrases only</span>
            </div>
            ${
              examples.length > 0
                ? `<ul class="customer-wording-list" aria-label="Customer wording examples">
                    ${examples.map((phrase) => `<li>${escapeHtml(phrase)}</li>`).join("")}
                  </ul>`
                : `<p class="muted">No customer wording examples are shown because this snapshot did not include real ticket phrases. No invented SEO terms are displayed.</p>`
            }
          </div>`;
}

function renderResolutionEvidenceDiagnostic(summary) {
  const present = summary.support_ticket_resolution_evidence_present === true;
  const count = finiteCount(summary.support_ticket_resolution_evidence_count);
  const label = present ? "Present" : "Absent";
  const copy = present
    ? `${formatNumber(count)} resolved ticket rows can support publishable answer drafting.`
    : "This export supports a gap list only; publishable answers need agent replies or resolved ticket notes.";
  return `<div
            class="resolution-evidence ${present ? "present" : "absent"}"
            data-atlas-deflection-resolution-evidence
            data-resolution-evidence-present="${present ? "true" : "false"}"
          >
            <div>
              <span>Resolution evidence</span>
              <strong>${label}</strong>
            </div>
            <p>${escapeHtml(copy)}</p>
          </div>`;
}

function renderRepeatVolumeDiagnostic(summary) {
  const count = finiteCount(summary.repeat_ticket_count);
  const light = count < LIGHT_REPEAT_TICKET_THRESHOLD;
  const label = count > 0 ? `${formatNumber(count)} question-level repeat tickets` : "No repeated questions yet";
  const copy = light
    ? "This export is light on question-level repeat volume. Review the free snapshot before paying for the full report."
    : "This export has enough question-level repeat volume for a substantial paid report preview.";
  return `<div
            class="repeat-volume ${light ? "light" : "ready"}"
            data-atlas-deflection-repeat-volume
            data-repeat-volume-light="${light ? "true" : "false"}"
          >
            <div>
              <span>Question-level repeat volume</span>
              <strong>${escapeHtml(label)}</strong>
            </div>
            <p>${escapeHtml(copy)}</p>
          </div>`;
}

function renderSnapshot(report) {
  if (!report || !report.ok || !report.snapshot) {
    return `<section class="snapshot" aria-labelledby="snapshot-title">
          <h2 id="snapshot-title">Free snapshot</h2>
          <p class="muted">The hosted result page could not load the ATLAS snapshot yet. No estimates are shown until ATLAS returns real snapshot values.</p>
        </section>`;
  }
  const summary = report.snapshot.summary || {};
  const questions = Array.isArray(report.snapshot.top_questions)
    ? report.snapshot.top_questions
    : [];
  return `<section class="snapshot" aria-labelledby="snapshot-title">
          <h2 id="snapshot-title">Free snapshot</h2>
          <div class="metrics">
            <div><span>Questions found</span><strong>${escapeHtml(formatNumber(summary.generated))}</strong></div>
            <div><span>Question-level repeat tickets</span><strong>${escapeHtml(formatNumber(finiteCount(summary.repeat_ticket_count)))}</strong></div>
            <div><span>Evidence-backed answers</span><strong>${escapeHtml(formatNumber(summary.drafted_answer_count))}</strong></div>
            <div><span>Needs support proof</span><strong>${escapeHtml(formatNumber(summary.no_proven_answer_count))}</strong></div>
          </div>
          ${renderRepeatVolumeDiagnostic(summary)}
          ${renderResolutionEvidenceDiagnostic(summary)}
          <h2>Help-desk SEO targeting list</h2>
          <p class="muted">Use actual customer phrases from the uploaded tickets for help-center titles, internal-search synonyms, and FAQ wording. No keyword volume, ranking, or traffic promise is implied.</p>
          ${renderCustomerWordingCard(questions)}
          <div class="questions">
            ${questions
              .map(
                (item) => `<article>
              <p class="rank">Rank ${escapeHtml(formatNumber(item.rank))} - ${escapeHtml(formatNumber(item.weighted_frequency))} weighted mentions</p>
              <h3>${escapeHtml(item.question)}</h3>
              <p class="muted">${escapeHtml(item.customer_wording)}</p>
            </article>`,
              )
              .join("")}
          </div>
        </section>`;
}

function paidArtifactItems(report) {
  if (
    !report ||
    !report.ok ||
    report.artifact_status !== "unlocked" ||
    !isObjectRecord(report.artifact) ||
    !isObjectRecord(report.artifact.faq_result) ||
    !Array.isArray(report.artifact.faq_result.items)
  ) {
    return [];
  }
  return report.artifact.faq_result.items.filter(isObjectRecord);
}

function paidEvidenceExportHref(report, requestId) {
  if (
    !report ||
    !report.ok ||
    report.artifact_status !== "unlocked" ||
    !isObjectRecord(report.artifact) ||
    !isObjectRecord(report.artifact.evidence_export) ||
    clean(report.artifact.evidence_export.schema_version) !== EVIDENCE_EXPORT_SCHEMA_VERSION
  ) {
    return "";
  }
  const exportRequestId = clean(requestId) || clean(report.request_id);
  return exportRequestId
    ? `/api/content-ops/deflection/evidence-export?request_id=${encodeURIComponent(exportRequestId)}`
    : "";
}

function itemTicketCount(item) {
  return firstParsedCount(item.ticket_count, item.frequency, item.weighted_frequency);
}

function itemOpportunityScore(item) {
  return firstParsedCount(item.opportunity_score, item.weighted_frequency, item.frequency);
}

function hasResolutionEvidence(item) {
  return clean(item.answer_evidence_status) === RESOLUTION_EVIDENCE_STATUS;
}

function paidSummary(report, items) {
  const summary = isObjectRecord(report.artifact.summary) ? report.artifact.summary : {};
  const repeatTickets = finiteCount(summary.repeat_ticket_count);
  const publishableCount = items.filter(hasResolutionEvidence).length;
  const generated = parsedCount(summary.generated) || items.length;
  const needsProof = Math.max(0, generated - publishableCount);
  return {
    generated,
    repeatTickets,
    publishableCount,
    needsProof,
    supportCost: repeatTickets * ASSISTED_CONTACT_COST,
  };
}

function uniqueTexts(values, limit) {
  const seen = new Set();
  const out = [];
  for (const value of values) {
    const text = clean(value);
    if (!text) continue;
    const key = text.toLocaleLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(text);
    if (out.length >= limit) break;
  }
  return out;
}

function paidCustomerPhrases(items) {
  const values = [];
  for (const item of items) {
    values.push(clean(item.customer_wording) || clean(item.question));
    if (Array.isArray(item.term_mappings)) {
      for (const mapping of item.term_mappings) {
        if (isObjectRecord(mapping)) values.push(mapping.customer_term);
      }
    }
  }
  return uniqueTexts(values, PAID_PHRASE_LIMIT);
}

function textPreview(value, limit = 180) {
  const text = clean(value).replace(/\s+/g, " ");
  if (text.length <= limit) return text;
  return `${text.slice(0, Math.max(0, limit - 1)).trim()}...`;
}

function renderPaidMetricCards(summary) {
  return `<div class="paid-metrics" data-atlas-deflection-paid-summary>
            <div><span>Support tax estimate</span><strong>${escapeHtml(formatMoney(summary.supportCost))}</strong></div>
            <div><span>Repeat-ticket workload</span><strong>${escapeHtml(formatNumber(summary.repeatTickets))}</strong></div>
            <div><span>Ranked questions</span><strong>${escapeHtml(formatNumber(summary.generated))}</strong></div>
            <div><span>Publishable answers</span><strong>${escapeHtml(formatNumber(summary.publishableCount))}</strong></div>
          </div>`;
}

function renderPaidReadiness(summary, evidenceExportHref) {
  const hasAnswers = summary.publishableCount > 0;
  return `<div class="paid-readiness" data-atlas-deflection-paid-readiness>
            <div class="${hasAnswers ? "ready" : "needs-proof"}">
              <span>Publishable-answer readiness</span>
              <strong>${hasAnswers ? "Evidence-backed answers present" : "Gap list only"}</strong>
              <p>${hasAnswers
                ? escapeHtml(`${formatNumber(summary.publishableCount)} questions have uploaded resolution evidence behind draftable answers.`)
                : "No uploaded resolution evidence was present; use this as a prioritized content roadmap."}</p>
            </div>
            <div>
              <span>Complete evidence export</span>
              <strong>${evidenceExportHref ? "Available as JSON" : "Awaiting export artifact"}</strong>
              <p>${evidenceExportHref
                ? `Use the browser dashboard for the summary, and download the uncapped evidence archive for audit/detail review. <a class="paid-download-link" data-atlas-deflection-evidence-export-download href="${escapeHtml(evidenceExportHref)}" download>Download complete evidence JSON</a>.`
                : "The browser dashboard is available, but this artifact does not expose the complete evidence export yet."}</p>
            </div>
          </div>`;
}

function renderPaidQuestionTable(items) {
  const rows = items.slice(0, PAID_QUESTION_LIMIT);
  if (!rows.length) return "";
  return `<section id="paid-ranked-questions" class="paid-report-block">
            <h3>Top ranked questions</h3>
            <div class="paid-table-wrap">
              <table class="paid-table">
                <thead>
                  <tr><th>Rank</th><th>Question</th><th>Tickets</th><th>Opportunity</th><th>Status</th></tr>
                </thead>
                <tbody>
                  ${rows.map((item, index) => `<tr>
                    <td>${index + 1}</td>
                    <td>${escapeHtml(item.question || item.customer_wording || "Untitled question")}</td>
                    <td>${escapeHtml(formatNumber(itemTicketCount(item)))}</td>
                    <td>${escapeHtml(formatNumber(itemOpportunityScore(item)))}</td>
                    <td>${hasResolutionEvidence(item) ? "Evidence-backed" : "Needs proof"}</td>
                  </tr>`).join("")}
                </tbody>
              </table>
            </div>
          </section>`;
}

function renderPaidAnswerCards(items) {
  const rows = items.filter(hasResolutionEvidence).slice(0, PAID_DETAIL_LIMIT);
  return `<section id="paid-publishable-answers" class="paid-report-block">
            <h3>Publishable answers</h3>
            ${rows.length > 0
              ? `<div class="paid-card-grid">
                  ${rows.map((item) => `<article class="paid-card">
                    <p class="rank">${escapeHtml(formatNumber(itemTicketCount(item)))} tickets</p>
                    <h4>${escapeHtml(item.question || item.customer_wording || "Untitled question")}</h4>
                    <p>${escapeHtml(textPreview(item.answer || "Draft answer is backed by uploaded resolution evidence."))}</p>
                  </article>`).join("")}
                </div>`
              : `<p class="muted">No uploaded resolution evidence was present, so this report is a gap list rather than publishable-answer copy.</p>`}
          </section>`;
}

function renderPaidGapCards(items) {
  const rows = items.filter((item) => !hasResolutionEvidence(item)).slice(0, PAID_DETAIL_LIMIT);
  return `<section id="paid-gap-list" class="paid-report-block">
            <h3>No-proven-answer gaps</h3>
            ${rows.length > 0
              ? `<div class="paid-card-grid">
                  ${rows.map((item) => `<article class="paid-card">
                    <p class="rank">${escapeHtml(formatNumber(itemTicketCount(item)))} tickets</p>
                    <h4>${escapeHtml(item.question || item.customer_wording || "Untitled question")}</h4>
                    <p>Needs uploaded resolution evidence before ATLAS can mark this as publishable help-center copy.</p>
                  </article>`).join("")}
                </div>`
              : `<p class="muted">No unresolved answer gaps were present in this unlocked report.</p>`}
          </section>`;
}

function renderPaidPhrases(items) {
  const phrases = paidCustomerPhrases(items);
  return `<section id="paid-customer-phrases" class="paid-report-block">
            <h3>Top customer wording and SEO phrases</h3>
            ${phrases.length > 0
              ? `<ul class="paid-phrase-list">
                  ${phrases.map((phrase) => `<li>${escapeHtml(phrase)}</li>`).join("")}
                </ul>`
              : `<p class="muted">No customer wording phrases were available in the structured paid artifact.</p>`}
          </section>`;
}

function renderPaidArtifact(report, requestId = "") {
  const items = paidArtifactItems(report);
  if (!items.length) return "";
  const summary = paidSummary(report, items);
  const evidenceExportHref = paidEvidenceExportHref(report, requestId);

  return `<section class="paid-report" aria-labelledby="paid-report-title" data-atlas-deflection-paid-report>
          <div class="paid-report-header">
            <div>
              <h2 id="paid-report-title">Paid report dashboard</h2>
              <p class="muted">Unlocked from the ATLAS paid artifact after Stripe webhook verification. This browser view is consolidated for decision-making; the complete evidence export carries the uncapped audit trail.</p>
            </div>
            <nav class="paid-report-nav" aria-label="Paid report sections">
              <a href="#paid-ranked-questions">Questions</a>
              <a href="#paid-publishable-answers">Answers</a>
              <a href="#paid-gap-list">Gaps</a>
              <a href="#paid-customer-phrases">Phrases</a>
              ${evidenceExportHref
                ? `<a data-atlas-deflection-evidence-export-download href="${escapeHtml(evidenceExportHref)}" download>Evidence JSON</a>`
                : ""}
            </nav>
          </div>
          ${renderPaidMetricCards(summary)}
          ${renderPaidReadiness(summary, evidenceExportHref)}
          ${renderPaidQuestionTable(items)}
          ${renderPaidAnswerCards(items)}
          ${renderPaidGapCards(items)}
          ${renderPaidPhrases(items)}
        </section>`;
}

function renderArtifactRetryScript({ requestId, shouldRetry }) {
  if (!shouldRetry) return "";

  return `\n  <script data-atlas-deflection-artifact-retry>
    (() => {
      const retryMessage = document.getElementById("checkout-message");
      const retryRequestId = ${scriptJson(requestId)};
      const retryDelays = [1500, 3000, 5000, 8000, 13000];
      let retryAttempt = 0;
      const reportUrl = () => "/api/content-ops/deflection/report?request_id="
        + encodeURIComponent(retryRequestId);
      const pollArtifactUnlock = async () => {
        if (retryMessage) retryMessage.textContent = "Checking unlock status...";
        try {
          const response = await fetch(reportUrl(), {
            headers: { "Accept": "application/json" }
          });
          const payload = await response.json().catch(() => null);
          if (response.ok && payload && payload.artifact_status === "unlocked") {
            window.location.reload();
            return;
          }
        } catch {
          // Keep the paid boundary fail-closed; the customer can refresh manually.
        }
        if (retryAttempt >= retryDelays.length) {
          if (retryMessage) retryMessage.textContent = "Payment is confirmed. Refresh in a moment if the report is still locked.";
          return;
        }
        const delay = retryDelays[retryAttempt];
        retryAttempt += 1;
        window.setTimeout(pollArtifactUnlock, delay);
      };
      if (retryRequestId) {
        window.setTimeout(pollArtifactUnlock, retryDelays[retryAttempt]);
        retryAttempt += 1;
      }
    })();
  </script>`;
}

function renderResultPage({ requestId, accountId, checkoutStatus = "", report = null }) {
  const safeRequestId = escapeHtml(requestId);
  const safeCheckoutStatus = escapeHtml(checkoutStatus);
  const resultHref = resultPath(requestId || "missing-request", accountId, "");
  const artifactStatus = report && report.ok ? report.artifact_status : "snapshot_unavailable";
  const isUnlocked = artifactStatus === "unlocked";
  const shouldRetryArtifact = checkoutStatus === "success" && artifactStatus === "locked";
  const buttonDisabled = requestId && !isUnlocked && !shouldRetryArtifact ? "" : "disabled";
  const unlockHeading = isUnlocked
    ? "Full report unlocked"
    : shouldRetryArtifact
      ? "Payment processing"
      : "Unlock full report";
  const unlockCopy = isUnlocked
    ? "ATLAS has released the paid artifact for this request."
    : shouldRetryArtifact
      ? "Stripe returned successfully. This page is checking for the verified ATLAS unlock."
      : "$1,500 one-time Stripe Checkout session.";
  const unlockButtonLabel = isUnlocked
    ? "Report unlocked"
    : shouldRetryArtifact
      ? "Checking unlock status"
      : "Continue to Checkout";
  const statusBanner =
    checkoutStatus === "success"
      ? `<div class="notice success">Checkout returned successfully. ATLAS unlocks the paid report only after Stripe sends the verified webhook.</div>`
      : checkoutStatus === "cancel"
        ? `<div class="notice">Checkout was cancelled. The report is still locked; you can restart Checkout when ready.</div>`
      : "";

  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="robots" content="noindex,nofollow">
  <title>FAQ Deflection Report | Juan Canfield</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #020617; color: #f8fafc; }
    a { color: #86efac; }
    .shell { max-width: 1120px; margin: 0 auto; padding: 48px 24px; }
    .grid { display: grid; grid-template-columns: minmax(0, 1fr) 360px; gap: 32px; align-items: start; }
    .eyebrow { display: inline-flex; margin-bottom: 20px; border: 1px solid rgba(34, 197, 94, .35); border-radius: 999px; padding: 6px 10px; color: #86efac; font-size: 12px; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }
    h1 { max-width: 760px; margin: 0; font-size: clamp(36px, 6vw, 56px); line-height: 1; letter-spacing: 0; }
    .lede { max-width: 760px; margin-top: 20px; color: rgba(226, 232, 240, .84); font-size: 18px; line-height: 1.7; }
    .panel { border: 1px solid rgba(30, 41, 59, .9); border-radius: 8px; background: rgba(15, 23, 42, .72); padding: 24px; }
    .meta { display: grid; gap: 14px; margin: 24px 0; }
    dt { color: rgba(226, 232, 240, .62); font-size: 13px; }
    dd { margin: 4px 0 0; color: #f8fafc; font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; overflow-wrap: anywhere; }
    button { width: 100%; border: 0; border-radius: 8px; background: #22c55e; color: #020617; padding: 13px 16px; font-weight: 800; cursor: pointer; }
    button:disabled { background: #1e293b; color: rgba(226, 232, 240, .55); cursor: not-allowed; }
    .snapshot { margin-top: 32px; border: 1px solid rgba(30, 41, 59, .9); border-radius: 8px; background: rgba(15, 23, 42, .42); padding: 24px; }
    .metrics { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 18px 0 24px; }
    .metrics div { border: 1px solid rgba(30, 41, 59, .9); border-radius: 8px; padding: 14px; background: rgba(2, 6, 23, .35); }
    .metrics span { display: block; color: rgba(226, 232, 240, .66); font-size: 13px; }
    .metrics strong { display: block; margin-top: 8px; font-size: 28px; }
    .resolution-evidence, .repeat-volume { display: flex; gap: 18px; justify-content: space-between; align-items: flex-start; margin: 0 0 24px; border: 1px solid rgba(30, 41, 59, .9); border-radius: 8px; padding: 16px; background: rgba(2, 6, 23, .35); }
    .repeat-volume { margin-bottom: 12px; }
    .resolution-evidence span, .repeat-volume span { display: block; color: rgba(226, 232, 240, .66); font-size: 13px; }
    .resolution-evidence strong, .repeat-volume strong { display: block; margin-top: 6px; font-size: 18px; }
    .resolution-evidence p, .repeat-volume p { max-width: 560px; margin: 0; color: rgba(226, 232, 240, .75); line-height: 1.55; }
    .resolution-evidence.present strong { color: #86efac; }
    .resolution-evidence.absent strong { color: #fde68a; }
    .repeat-volume.ready strong { color: #86efac; }
    .repeat-volume.light strong { color: #fde68a; }
    .customer-wording-card { margin: 18px 0 24px; border: 1px solid rgba(30, 41, 59, .9); border-radius: 8px; background: rgba(2, 6, 23, .35); padding: 18px; }
    .customer-wording-header { display: flex; gap: 10px; align-items: baseline; justify-content: space-between; }
    .customer-wording-header p { margin: 0; font-weight: 700; }
    .customer-wording-header span { color: #86efac; font-size: 11px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
    .customer-wording-list { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin: 16px 0 0; padding: 0; list-style: none; }
    .customer-wording-list li { border: 1px solid rgba(30, 41, 59, .9); border-radius: 6px; background: rgba(15, 23, 42, .62); padding: 9px 11px; color: #f8fafc; line-height: 1.45; }
    .questions { display: grid; gap: 14px; }
    .questions article { border-top: 1px solid rgba(30, 41, 59, .9); padding-top: 14px; }
    .questions h3 { margin: 6px 0; font-size: 17px; }
    .rank { color: #86efac; font-size: 12px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
    .paid-report { margin-top: 24px; border: 1px solid rgba(134, 239, 172, .28); border-radius: 8px; background: rgba(20, 83, 45, .16); padding: 24px; }
    .paid-report-header { display: flex; gap: 20px; align-items: flex-start; justify-content: space-between; }
    .paid-report-header h2 { margin-top: 0; }
    .paid-report-nav { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; min-width: 240px; }
    .paid-report-nav a { border: 1px solid rgba(134, 239, 172, .28); border-radius: 999px; padding: 7px 10px; background: rgba(2, 6, 23, .32); color: #bbf7d0; font-size: 12px; font-weight: 800; text-decoration: none; }
    .paid-metrics { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 20px 0; }
    .paid-metrics div, .paid-readiness > div, .paid-card, .paid-phrase-list li { border: 1px solid rgba(30, 41, 59, .9); border-radius: 8px; background: rgba(2, 6, 23, .35); }
    .paid-metrics div { padding: 14px; }
    .paid-metrics span, .paid-readiness span { display: block; color: rgba(226, 232, 240, .66); font-size: 13px; }
    .paid-metrics strong { display: block; margin-top: 8px; font-size: 26px; }
    .paid-readiness { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin: 0 0 22px; }
    .paid-readiness > div { padding: 16px; }
    .paid-readiness strong { display: block; margin-top: 6px; color: #86efac; font-size: 18px; }
    .paid-readiness .needs-proof strong { color: #fde68a; }
    .paid-readiness p { margin: 10px 0 0; color: rgba(226, 232, 240, .75); line-height: 1.55; }
    .paid-download-link { display: inline-block; margin-top: 8px; font-weight: 800; }
    .paid-report-block { margin-top: 24px; }
    .paid-report-block h3 { margin: 0 0 12px; font-size: 18px; }
    .paid-table-wrap { overflow-x: auto; border: 1px solid rgba(30, 41, 59, .9); border-radius: 8px; }
    .paid-table { width: 100%; border-collapse: collapse; min-width: 680px; }
    .paid-table th, .paid-table td { border-bottom: 1px solid rgba(30, 41, 59, .9); padding: 10px 12px; text-align: left; vertical-align: top; }
    .paid-table th { color: rgba(226, 232, 240, .66); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }
    .paid-card-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    .paid-card { padding: 14px; }
    .paid-card h4 { margin: 6px 0 8px; font-size: 16px; }
    .paid-card p:not(.rank) { margin: 0; color: rgba(226, 232, 240, .75); line-height: 1.55; }
    .paid-phrase-list { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin: 0; padding: 0; list-style: none; }
    .paid-phrase-list li { padding: 9px 11px; line-height: 1.45; }
    .notice { margin-top: 20px; border-radius: 8px; padding: 14px 16px; color: #fde68a; background: rgba(251, 191, 36, .1); border: 1px solid rgba(251, 191, 36, .28); }
    .success { color: #bbf7d0; background: rgba(34, 197, 94, .1); border-color: rgba(34, 197, 94, .28); }
    .muted { color: rgba(226, 232, 240, .68); line-height: 1.65; }
    @media (max-width: 920px) { .metrics, .paid-metrics, .paid-readiness { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
    @media (max-width: 820px) { .grid, .metrics, .customer-wording-list, .paid-metrics, .paid-readiness, .paid-card-grid, .paid-phrase-list { grid-template-columns: 1fr; } .shell { padding-top: 32px; } .customer-wording-header, .resolution-evidence, .repeat-volume, .paid-report-header { align-items: flex-start; flex-direction: column; } .paid-report-nav { justify-content: flex-start; min-width: 0; } }
  </style>
</head>
<body>
  <main
    class="shell"
    data-atlas-deflection-result
    data-atlas-deflection-request-id="${safeRequestId}"
    data-atlas-deflection-report-source="${CHECKOUT_SOURCE}"
    data-atlas-deflection-artifact-retry="${shouldRetryArtifact ? "true" : "false"}"
  >
    <a href="/services">Services</a>
    <div class="grid">
      <section>
        <div class="eyebrow">Locked report</div>
        <h1>FAQ deflection report is ready for review</h1>
        <p class="lede">The free snapshot and paid artifact stay separated. The full report unlocks only after Stripe confirms payment and ATLAS releases the artifact from its signed webhook path.</p>
        ${statusBanner}
        ${renderSnapshot(report)}
        ${renderPaidArtifact(report, requestId)}
      </section>
      <aside class="panel">
        <h2>${unlockHeading}</h2>
        <p class="muted">${unlockCopy}</p>
        <dl class="meta">
          <div><dt>Checkout metadata source</dt><dd>${CHECKOUT_SOURCE}</dd></div>
          <div><dt>request_id</dt><dd>${safeRequestId || "Missing request id"}</dd></div>
          <div><dt>canonical result path</dt><dd>${escapeHtml(resultHref)}</dd></div>
          <div><dt>artifact_status</dt><dd>${escapeHtml(artifactStatus)}</dd></div>
          <div><dt>checkout</dt><dd>${safeCheckoutStatus || "locked"}</dd></div>
        </dl>
        <button
          type="button"
          data-atlas-deflection-unlock
          data-checkout-source="${CHECKOUT_SOURCE}"
          data-checkout-request_id="${safeRequestId}"
          ${buttonDisabled}
        >${unlockButtonLabel}</button>
        <p id="checkout-message" class="muted" role="status"></p>
      </aside>
    </div>
  </main>
  <script>
    const button = document.querySelector("[data-atlas-deflection-unlock]");
    const message = document.getElementById("checkout-message");
    const requestId = ${scriptJson(requestId)};
    if (button && requestId) {
      button.addEventListener("click", async () => {
        button.disabled = true;
        message.textContent = "Opening Checkout...";
        try {
          const response = await fetch("/api/content-ops/deflection/checkout", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ request_id: requestId })
          });
          const payload = await response.json();
          if (!response.ok || !payload.url) throw new Error(payload.error || "checkout_failed");
          window.location.assign(payload.url);
        } catch {
          button.disabled = false;
          message.textContent = "Checkout could not be started.";
        }
      });
    }
  </script>
  ${renderArtifactRetryScript({ requestId, shouldRetry: shouldRetryArtifact })}
</body>
</html>`;
}

export { renderResultPage };

export default async function handler(req, res) {
  const url = requestUrl(req);
  const requestId = requestIdFrom(req, url);
  const accountId = clean(req.query?.account_id) || clean(url.searchParams.get("account_id"));
  if (requestId && !REQUEST_ID_RE.test(requestId)) {
    res.statusCode = 400;
    res.setHeader("Content-Type", "text/plain; charset=utf-8");
    res.end("Invalid request_id");
    return;
  }
  if (accountId && !UUID_RE.test(accountId)) {
    res.statusCode = 400;
    res.setHeader("Content-Type", "text/plain; charset=utf-8");
    res.end("Invalid account_id");
    return;
  }
  res.statusCode = 200;
  res.setHeader("Content-Type", "text/html; charset=utf-8");
  res.setHeader("Cache-Control", "no-store");
  const report =
    requestId
      ? await loadDeflectionReport({ requestId, accountId })
      : null;
  res.end(
    renderResultPage({
      requestId,
      accountId,
      checkoutStatus: clean(url.searchParams.get("checkout")),
      report,
    }),
  );
}
