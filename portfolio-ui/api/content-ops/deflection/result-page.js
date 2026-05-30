import { loadDeflectionReport } from "./atlas-report.js";
import { CHECKOUT_SOURCE, resultPath } from "./checkout.js";

const REQUEST_ID_RE = /^[A-Za-z0-9][A-Za-z0-9_.:-]{5,160}$/;
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

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
            <div><span>Evidence-backed answers</span><strong>${escapeHtml(formatNumber(summary.drafted_answer_count))}</strong></div>
            <div><span>Needs support proof</span><strong>${escapeHtml(formatNumber(summary.no_proven_answer_count))}</strong></div>
          </div>
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

function renderPaidArtifact(report) {
  const markdown =
    report &&
    report.ok &&
    report.artifact_status === "unlocked" &&
    report.artifact &&
    typeof report.artifact.markdown === "string"
      ? report.artifact.markdown.trim()
      : "";

  if (!markdown) return "";

  return `<section class="paid-report" aria-labelledby="paid-report-title" data-atlas-deflection-paid-report>
          <h2 id="paid-report-title">Full report</h2>
          <p class="muted">Unlocked from the ATLAS paid artifact after Stripe webhook verification.</p>
          <pre class="report-markdown">${escapeHtml(markdown)}</pre>
        </section>`;
}

function renderResultPage({ requestId, accountId, checkoutStatus = "", report = null }) {
  const safeRequestId = escapeHtml(requestId);
  const safeAccountId = escapeHtml(accountId);
  const safeCheckoutStatus = escapeHtml(checkoutStatus);
  const resultHref = resultPath(requestId || "missing-request", accountId, "");
  const artifactStatus = report && report.ok ? report.artifact_status : "snapshot_unavailable";
  const isUnlocked = artifactStatus === "unlocked";
  const buttonDisabled = requestId && accountId && !isUnlocked ? "" : "disabled";
  const unlockHeading = isUnlocked ? "Full report unlocked" : "Unlock full report";
  const unlockCopy = isUnlocked
    ? "ATLAS has released the paid artifact for this request."
    : "$1,500 one-time Stripe Checkout session.";
  const unlockButtonLabel = isUnlocked ? "Report unlocked" : "Continue to Checkout";
  const statusBanner =
    checkoutStatus === "success"
      ? `<div class="notice success">Checkout returned successfully. ATLAS unlocks the paid report only after Stripe sends the verified webhook.</div>`
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
    .metrics { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 18px 0 24px; }
    .metrics div { border: 1px solid rgba(30, 41, 59, .9); border-radius: 8px; padding: 14px; background: rgba(2, 6, 23, .35); }
    .metrics span { display: block; color: rgba(226, 232, 240, .66); font-size: 13px; }
    .metrics strong { display: block; margin-top: 8px; font-size: 28px; }
    .questions { display: grid; gap: 14px; }
    .questions article { border-top: 1px solid rgba(30, 41, 59, .9); padding-top: 14px; }
    .questions h3 { margin: 6px 0; font-size: 17px; }
    .rank { color: #86efac; font-size: 12px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
    .paid-report { margin-top: 24px; border: 1px solid rgba(134, 239, 172, .28); border-radius: 8px; background: rgba(20, 83, 45, .16); padding: 24px; }
    .report-markdown { margin: 18px 0 0; max-height: 720px; overflow: auto; border: 1px solid rgba(30, 41, 59, .9); border-radius: 8px; background: rgba(2, 6, 23, .55); padding: 18px; color: #e2e8f0; font: 13px/1.65 ui-monospace, SFMono-Regular, Menlo, monospace; white-space: pre-wrap; overflow-wrap: anywhere; }
    .notice { margin-top: 20px; border-radius: 8px; padding: 14px 16px; color: #fde68a; background: rgba(251, 191, 36, .1); border: 1px solid rgba(251, 191, 36, .28); }
    .success { color: #bbf7d0; background: rgba(34, 197, 94, .1); border-color: rgba(34, 197, 94, .28); }
    .muted { color: rgba(226, 232, 240, .68); line-height: 1.65; }
    @media (max-width: 820px) { .grid, .metrics { grid-template-columns: 1fr; } .shell { padding-top: 32px; } }
  </style>
</head>
<body>
  <main
    class="shell"
    data-atlas-deflection-result
    data-atlas-deflection-request-id="${safeRequestId}"
    data-atlas-deflection-account-id="${safeAccountId}"
    data-atlas-deflection-report-source="${CHECKOUT_SOURCE}"
  >
    <a href="/services">Services</a>
    <div class="grid">
      <section>
        <div class="eyebrow">Locked report</div>
        <h1>FAQ deflection report is ready for review</h1>
        <p class="lede">The free snapshot and paid artifact stay separated. The full report unlocks only after Stripe confirms payment and ATLAS releases the artifact from its signed webhook path.</p>
        ${statusBanner}
        ${renderSnapshot(report)}
        ${renderPaidArtifact(report)}
      </section>
      <aside class="panel">
        <h2>${unlockHeading}</h2>
        <p class="muted">${unlockCopy}</p>
        <dl class="meta">
          <div><dt>Checkout metadata source</dt><dd>${CHECKOUT_SOURCE}</dd></div>
          <div><dt>request_id</dt><dd>${safeRequestId || "Missing request id"}</dd></div>
          <div><dt>account_id</dt><dd>${safeAccountId || "Missing account id"}</dd></div>
          <div><dt>canonical result path</dt><dd>${escapeHtml(resultHref)}</dd></div>
          <div><dt>artifact_status</dt><dd>${escapeHtml(artifactStatus)}</dd></div>
          <div><dt>checkout</dt><dd>${safeCheckoutStatus || "locked"}</dd></div>
        </dl>
        <button type="button" data-atlas-deflection-unlock ${buttonDisabled}>${unlockButtonLabel}</button>
        <p id="checkout-message" class="muted" role="status"></p>
      </aside>
    </div>
  </main>
  <script>
    const button = document.querySelector("[data-atlas-deflection-unlock]");
    const message = document.getElementById("checkout-message");
    const requestId = ${scriptJson(requestId)};
    const accountId = ${scriptJson(accountId)};
    if (button && requestId && accountId) {
      button.addEventListener("click", async () => {
        button.disabled = true;
        message.textContent = "Opening Checkout...";
        try {
          const response = await fetch("/api/content-ops/deflection/checkout", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ request_id: requestId, account_id: accountId })
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
    requestId && accountId
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
