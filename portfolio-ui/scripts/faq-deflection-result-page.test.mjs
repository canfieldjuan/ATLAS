import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import handler, {
  CHECKOUT_SOURCE,
  buildStripeCheckoutBody,
  checkoutUrls,
  resultPath,
  validateConfiguredPrice,
  validatePayload,
} from "../api/content-ops/deflection/checkout.js";
import resultPageHandler, { renderResultPage } from "../api/content-ops/deflection/result-page.js";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const appSource = await readFile(resolve(root, "src/App.tsx"), "utf8");
const pageSource = await readFile(resolve(root, "src/pages/FaqDeflectionResult.tsx"), "utf8");
const vercelConfig = JSON.parse(await readFile(resolve(root, "vercel.json"), "utf8"));
const checkoutSource = await readFile(
  resolve(root, "api/content-ops/deflection/checkout.js"),
  "utf8",
);
const resultPageSource = await readFile(
  resolve(root, "api/content-ops/deflection/result-page.js"),
  "utf8",
);

async function test(name, fn) {
  try {
    await fn();
    console.log(`ok - ${name}`);
  } catch (error) {
    console.error(`not ok - ${name}`);
    throw error;
  }
}

function mockResponse() {
  return {
    statusCode: 200,
    headers: {},
    body: "",
    setHeader(name, value) {
      this.headers[name] = value;
    },
    end(value) {
      this.body = value;
    },
  };
}

await test("route is wired at the hosted FAQ deflection result path", () => {
  assert.match(appSource, /FaqDeflectionResult/);
  assert.match(appSource, /\/services\/faq-deflection\/results\/:requestId/);
  assert.deepEqual(vercelConfig.rewrites[0], {
    source: "/services/faq-deflection/results/:requestId",
    destination: "/api/content-ops/deflection/result-page?request_id=:requestId",
  });
});

await test("result page exposes validation markers and checkout metadata", () => {
  const html = renderResultPage({
    requestId: "content-ops-abc123",
    accountId: "2b2b950d-f64b-4852-bc30-f92a34cdf169",
  });
  for (const marker of [
    "data-atlas-deflection-result",
    "data-atlas-deflection-request-id",
    "data-atlas-deflection-unlock",
    "request_id",
    "account_id",
  ]) {
    assert.match(pageSource, new RegExp(marker));
    assert.match(resultPageSource, new RegExp(marker));
    assert.match(html, new RegExp(marker));
  }
  assert.match(pageSource, /content_ops_deflection_report/);
  assert.match(html, /content_ops_deflection_report/);
  assert.match(html, /content-ops-abc123/);
  assert.match(html, /2b2b950d-f64b-4852-bc30-f92a34cdf169/);
});

await test("result pages never embed ATLAS service credentials or paid-route calls", () => {
  for (const source of [pageSource, resultPageSource]) {
    assert.doesNotMatch(source, /ATLAS_B2B_JWT|ATLAS_API_BASE_URL|ATLAS_TOKEN/);
    assert.doesNotMatch(source, /\/paid\b/);
  }
});

await test("snapshot guard names paid report fields that must not render pre-payment", () => {
  for (const forbidden of ["markdown", "faq_result", "answer", "evidence_quotes", "source_ids"]) {
    assert.match(pageSource, new RegExp(`"${forbidden}"`));
  }
  assert.match(pageSource, /collectForbiddenKeys/);
});

await test("checkout endpoint validates request and account identifiers", () => {
  const valid = validatePayload({
    request_id: "content-ops-abc123",
    account_id: "2b2b950d-f64b-4852-bc30-f92a34cdf169",
  });
  assert.deepEqual(valid.errors, []);

  const invalid = validatePayload({
    request_id: "../bad",
    account_id: "not-a-uuid",
  });
  assert.deepEqual(invalid.errors, [
    "request_id must be a valid Content Ops request id",
    "account_id must be a valid ATLAS account UUID",
  ]);
});

await test("hosted result page rejects script-breaking account_id input", async () => {
  const res = mockResponse();
  await resultPageHandler(
    {
      url: "/services/faq-deflection/results/content-ops-abc123?account_id=%3C%2Fscript%3E%3Cimg%20src%3Dx%20onerror%3Dalert(1)%3E",
      headers: { host: "portfolio.example.com", "x-forwarded-proto": "https" },
      query: {
        request_id: "content-ops-abc123",
        account_id: "</script><img src=x onerror=alert(1)>",
      },
    },
    res,
  );
  assert.equal(res.statusCode, 400);
  assert.equal(res.body, "Invalid account_id");
});

await test("rendered inline script escapes script terminators defensively", () => {
  const html = renderResultPage({
    requestId: "content-ops-abc123",
    accountId: "</script><img src=x onerror=alert(1)>",
  });
  assert.doesNotMatch(html, /const accountId = "<\/script>/);
  assert.match(html, /const accountId = "\\u003c\/script>/);
});

await test("checkout body carries the Stripe metadata contract and one-time price", () => {
  delete process.env.STRIPE_DEFLECTION_REPORT_PRICE_ID;
  process.env.DEFLECTION_REPORT_AMOUNT_CENTS = "10";
  process.env.DEFLECTION_REPORT_CURRENCY = "USD";
  const body = buildStripeCheckoutBody({
    requestId: "content-ops-abc123",
    accountId: "2b2b950d-f64b-4852-bc30-f92a34cdf169",
    successUrl: "https://example.com/success",
    cancelUrl: "https://example.com/cancel",
  });

  assert.equal(body.get("mode"), "payment");
  assert.equal(body.get("metadata[source]"), CHECKOUT_SOURCE);
  assert.equal(body.get("metadata[account_id]"), "2b2b950d-f64b-4852-bc30-f92a34cdf169");
  assert.equal(body.get("metadata[request_id]"), "content-ops-abc123");
  assert.equal(body.get("payment_intent_data[metadata][source]"), CHECKOUT_SOURCE);
  assert.equal(body.get("line_items[0][quantity]"), "1");
  assert.equal(body.get("line_items[0][price_data][currency]"), "usd");
  assert.equal(body.get("line_items[0][price_data][unit_amount]"), "150000");
});

await test("configured Stripe Price must validate against webhook floor before Checkout", async () => {
  const previousFetch = globalThis.fetch;
  const previousPriceId = process.env.STRIPE_DEFLECTION_REPORT_PRICE_ID;
  process.env.STRIPE_DEFLECTION_REPORT_PRICE_ID = "price_deflection_low";
  const calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push({ url, options });
    return {
      ok: true,
      status: 200,
      async json() {
        return { active: true, currency: "usd", unit_amount: 149999 };
      },
    };
  };

  try {
    const result = await validateConfiguredPrice("sk_test_123");
    assert.deepEqual(result, {
      ok: false,
      message: "configured Stripe Price must be active, usd, and at least 150000 cents",
    });
  } finally {
    globalThis.fetch = previousFetch;
    if (previousPriceId === undefined) {
      delete process.env.STRIPE_DEFLECTION_REPORT_PRICE_ID;
    } else {
      process.env.STRIPE_DEFLECTION_REPORT_PRICE_ID = previousPriceId;
    }
  }
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, "https://api.stripe.com/v1/prices/price_deflection_low");
});

await test("checkout return path preserves result identifiers", () => {
  const path = resultPath(
    "content-ops-abc123",
    "2b2b950d-f64b-4852-bc30-f92a34cdf169",
    "success",
  );
  assert.equal(
    path,
    "/services/faq-deflection/results/content-ops-abc123?account_id=2b2b950d-f64b-4852-bc30-f92a34cdf169&checkout=success",
  );
});

await test("hosted result page handles the Checkout cancel return token", () => {
  const requestId = "content-ops-abc123";
  const accountId = "2b2b950d-f64b-4852-bc30-f92a34cdf169";
  const previousSuccessUrl = process.env.DEFLECTION_CHECKOUT_SUCCESS_URL;
  const previousCancelUrl = process.env.DEFLECTION_CHECKOUT_CANCEL_URL;
  delete process.env.DEFLECTION_CHECKOUT_SUCCESS_URL;
  delete process.env.DEFLECTION_CHECKOUT_CANCEL_URL;
  let checkoutStatus;
  try {
    const urls = checkoutUrls(
      {
        headers: { host: "portfolio.example.com", "x-forwarded-proto": "https" },
      },
      requestId,
      accountId,
    );
    assert.deepEqual(urls.errors, []);
    assert.equal(
      urls.cancelUrl,
      `https://portfolio.example.com${resultPath(requestId, accountId, "cancel")}`,
    );
    checkoutStatus = new URL(urls.cancelUrl).searchParams.get("checkout");
    assert.equal(checkoutStatus, "cancel");
  } finally {
    if (previousSuccessUrl === undefined) {
      delete process.env.DEFLECTION_CHECKOUT_SUCCESS_URL;
    } else {
      process.env.DEFLECTION_CHECKOUT_SUCCESS_URL = previousSuccessUrl;
    }
    if (previousCancelUrl === undefined) {
      delete process.env.DEFLECTION_CHECKOUT_CANCEL_URL;
    } else {
      process.env.DEFLECTION_CHECKOUT_CANCEL_URL = previousCancelUrl;
    }
  }

  const html = renderResultPage({
    requestId,
    accountId,
    checkoutStatus,
    report: {
      ok: true,
      snapshot: {
        summary: { generated: 1, drafted_answer_count: 0, no_proven_answer_count: 1 },
        top_questions: [],
      },
      artifact_status: "locked",
    },
  });
  assert.match(html, /Checkout was cancelled/);
  assert.match(html, /data-atlas-deflection-artifact-retry="false"/);
  assert.doesNotMatch(html, /script data-atlas-deflection-artifact-retry/);
  assert.match(html, /Continue to Checkout/);
  assert.doesNotMatch(html, /data-atlas-deflection-paid-report/);
});

await test("handler creates Stripe Checkout without calling privileged ATLAS paid route", async () => {
  assert.doesNotMatch(checkoutSource, /\/content-ops\/deflection-reports\/.+\/paid/);
  const previousFetch = globalThis.fetch;
  const previousSecret = process.env.STRIPE_SECRET_KEY;
  const previousPriceId = process.env.STRIPE_DEFLECTION_REPORT_PRICE_ID;
  const calls = [];
  process.env.STRIPE_SECRET_KEY = "sk_test_123";
  delete process.env.STRIPE_DEFLECTION_REPORT_PRICE_ID;
  globalThis.fetch = async (url, options) => {
    calls.push({ url, options });
    return {
      ok: true,
      status: 200,
      async json() {
        return { url: "https://checkout.stripe.com/c/session" };
      },
    };
  };

  const req = {
    method: "POST",
    headers: {
      host: "portfolio.example.com",
      "x-forwarded-proto": "https",
    },
    body: {
      request_id: "content-ops-abc123",
      account_id: "2b2b950d-f64b-4852-bc30-f92a34cdf169",
    },
  };
  const res = mockResponse();

  try {
    await handler(req, res);
  } finally {
    globalThis.fetch = previousFetch;
    if (previousSecret === undefined) {
      delete process.env.STRIPE_SECRET_KEY;
    } else {
      process.env.STRIPE_SECRET_KEY = previousSecret;
    }
    if (previousPriceId === undefined) {
      delete process.env.STRIPE_DEFLECTION_REPORT_PRICE_ID;
    } else {
      process.env.STRIPE_DEFLECTION_REPORT_PRICE_ID = previousPriceId;
    }
  }

  assert.equal(res.statusCode, 200);
  assert.deepEqual(JSON.parse(res.body), { url: "https://checkout.stripe.com/c/session" });
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, "https://api.stripe.com/v1/checkout/sessions");
  assert.equal(calls[0].options.body.get("metadata[source]"), CHECKOUT_SOURCE);
  assert.equal(calls[0].options.body.get("metadata[request_id]"), "content-ops-abc123");
});
