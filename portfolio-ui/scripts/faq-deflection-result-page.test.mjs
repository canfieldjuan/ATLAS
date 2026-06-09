import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import handler, {
  CHECKOUT_SOURCE,
  buildStripeCheckoutBody,
  checkoutUrls,
  resultPath,
  stripeCheckoutKeyConfig,
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

async function withEnv(nextEnv, fn) {
  const previous = {};
  for (const key of Object.keys(nextEnv)) {
    previous[key] = process.env[key];
    process.env[key] = nextEnv[key];
  }
  try {
    return await fn();
  } finally {
    for (const key of Object.keys(nextEnv)) {
      if (previous[key] === undefined) {
        delete process.env[key];
      } else {
        process.env[key] = previous[key];
      }
    }
  }
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
  ]) {
    assert.match(pageSource, new RegExp(marker));
    assert.match(resultPageSource, new RegExp(marker));
    assert.match(html, new RegExp(marker));
  }
  for (const source of [pageSource, resultPageSource]) {
    assert.match(source, /data-atlas-deflection-resolution-evidence/);
  }
  assert.match(pageSource, /content_ops_deflection_report/);
  assert.match(html, /content_ops_deflection_report/);
  assert.match(html, /content-ops-abc123/);
  assert.match(html, /data-checkout-source="content_ops_deflection_report"/);
  assert.match(html, /data-checkout-request_id="content-ops-abc123"/);
  assert.doesNotMatch(html, /2b2b950d-f64b-4852-bc30-f92a34cdf169/);
  assert.doesNotMatch(html, /data-checkout-account_id|data-atlas-deflection-account-id/);
  assert.doesNotMatch(pageSource, /account_id|data-checkout-account_id|data-atlas-deflection-account-id/);
  assert.match(resultPageSource, /data-checkout-source=/);
  assert.match(resultPageSource, /data-checkout-request_id=/);
  assert.match(pageSource, /data-checkout-source=/);
  assert.match(pageSource, /data-checkout-request_id=/);
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

await test("real snapshot page groups bounded customer wording examples", () => {
  const compactPageSource = pageSource.replace(/\s+/g, " ");
  for (const source of [pageSource, resultPageSource]) {
    assert.match(source, /Customer wording/);
    assert.match(source, /Help-desk SEO targeting list/);
    assert.match(source, /aria-label="Customer wording examples"/);
    assert.match(source, /No invented\s+SEO terms are displayed/);
    assert.doesNotMatch(source, /guaranteed (keyword|ranking|traffic|search)/i);
  }
  assert.match(pageSource, /customerWordingExamples/);
  assert.match(
    compactPageSource,
    /top_questions\s*\.map\(\(question\) => question\.customer_wording\.trim\(\)\)/,
  );
  assert.match(pageSource, /\.slice\(0, 5\)/);

  const topQuestions = Array.from({ length: 6 }, (_, index) => ({
    rank: index + 1,
    question: `Question ${index + 1}`,
    weighted_frequency: 6 - index,
    customer_wording: `customer phrase ${index + 1}`,
  }));
  const html = renderResultPage({
    requestId: "content-ops-abc123",
    accountId: "2b2b950d-f64b-4852-bc30-f92a34cdf169",
    report: {
      ok: true,
      snapshot: {
        summary: {
          generated: 6,
          drafted_answer_count: 2,
          no_proven_answer_count: 4,
          support_ticket_resolution_evidence_present: true,
          support_ticket_resolution_evidence_count: 2,
        },
        top_questions: topQuestions,
      },
      artifact_status: "locked",
    },
  });
  assert.match(html, /Help-desk SEO targeting list/);
  assert.match(html, /data-atlas-deflection-resolution-evidence/);
  assert.match(html, /data-resolution-evidence-present="true"/);
  assert.match(html, /Resolution evidence/);
  assert.match(html, /Present/);
  assert.match(html, /Customer wording/);
  assert.match(html, /aria-label="Customer wording examples"/);
  assert.match(html, /No keyword volume, ranking, or traffic promise is implied/);
  assert.match(resultPageSource, /<h2>Help-desk SEO targeting list<\/h2>/);
  assert.doesNotMatch(resultPageSource, /<h3>Help-desk SEO targeting list<\/h3>/);
  assert.ok(
    html.indexOf("<h2>Help-desk SEO targeting list</h2>") < html.indexOf("<h3>Question 1</h3>"),
    "server-rendered SEO list heading should sit above repeated question headings",
  );
  const list = html.match(
    /<ul class="customer-wording-list" aria-label="Customer wording examples">([\s\S]*?)<\/ul>/,
  );
  assert.ok(list, "customer wording examples list must render");
  assert.match(list[1], /customer phrase 1/);
  assert.match(list[1], /customer phrase 5/);
  assert.doesNotMatch(list[1], /customer phrase 6/);
  assert.match(html, /customer phrase 6/);

  const emptyHtml = renderResultPage({
    requestId: "content-ops-abc123",
    accountId: "2b2b950d-f64b-4852-bc30-f92a34cdf169",
    report: {
      ok: true,
      snapshot: {
        summary: {
          generated: 0,
          drafted_answer_count: 0,
          no_proven_answer_count: 0,
          support_ticket_resolution_evidence_present: false,
          support_ticket_resolution_evidence_count: 0,
        },
        top_questions: [],
      },
      artifact_status: "locked",
    },
  });
  assert.match(emptyHtml, /No customer wording examples are shown/);
  assert.match(emptyHtml, /No invented SEO terms are displayed/);
  assert.doesNotMatch(emptyHtml, /aria-label="Customer wording examples"/);
});

await test("checkout endpoint validates request and configured account identifiers", async () => {
  const env = { ATLAS_ACCOUNT_ID: "2b2b950d-f64b-4852-bc30-f92a34cdf169" };
  const valid = validatePayload({
    request_id: "content-ops-abc123",
  }, env);
  assert.deepEqual(valid.errors, []);
  assert.equal(valid.accountId, "2b2b950d-f64b-4852-bc30-f92a34cdf169");

  const legacy = validatePayload({
    request_id: "content-ops-abc123",
    account_id: "2b2b950d-f64b-4852-bc30-f92a34cdf169",
  }, env);
  assert.deepEqual(legacy.errors, []);

  const invalid = validatePayload({
    request_id: "../bad",
    account_id: "not-a-uuid",
  }, env);
  assert.deepEqual(invalid.errors, [
    "request_id must be a valid Content Ops request id",
    "account_id must be a valid ATLAS account UUID",
  ]);

  const mismatch = validatePayload({
    request_id: "content-ops-abc123",
    account_id: "3b2b950d-f64b-4852-bc30-f92a34cdf169",
  }, env);
  assert.deepEqual(mismatch.errors, [
    "account_id does not match the configured ATLAS account",
  ]);
});

await test("hosted result page loads report with configured account when URL omits account id", async () => {
  const previousFetch = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push({ url, options });
    return {
      status: calls.length === 1 ? 200 : 403,
      async text() {
        return JSON.stringify(
          calls.length === 1
            ? {
                summary: {
                  generated: 1,
                  drafted_answer_count: 0,
                  no_proven_answer_count: 1,
                  support_ticket_resolution_evidence_present: false,
                  support_ticket_resolution_evidence_count: 0,
                },
                top_questions: [
                  {
                    rank: 1,
                    question: "How do I export reports?",
                    weighted_frequency: 3,
                    customer_wording: "export reports",
                  },
                ],
              }
            : { detail: "locked" },
        );
      },
    };
  };
  const res = mockResponse();
  try {
    await withEnv({
      ATLAS_API_BASE_URL: "https://atlas.example.com",
      ATLAS_B2B_JWT: "secret-service-token",
      ATLAS_ACCOUNT_ID: "2b2b950d-f64b-4852-bc30-f92a34cdf169",
    }, async () => {
      await resultPageHandler(
        {
          url: "/services/faq-deflection/results/content-ops-abc123",
          headers: { host: "portfolio.example.com", "x-forwarded-proto": "https" },
          query: { request_id: "content-ops-abc123" },
        },
        res,
      );
    });
  } finally {
    globalThis.fetch = previousFetch;
  }

  assert.equal(res.statusCode, 200);
  assert.equal(calls.length, 2);
  assert.match(res.body, /How do I export reports\?/);
  assert.match(res.body, /data-atlas-deflection-artifact-retry="false"/);
  assert.doesNotMatch(res.body, /account_id=/);
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
    requestId: "</script><img src=x onerror=alert(1)>",
    accountId: "2b2b950d-f64b-4852-bc30-f92a34cdf169",
  });
  assert.doesNotMatch(html, /const requestId = "<\/script>/);
  assert.match(html, /const requestId = "\\u003c\/script>/);
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

await test("restricted Checkout key skips configured Price read preflight", async () => {
  const previousFetch = globalThis.fetch;
  const previousSecret = process.env.STRIPE_SECRET_KEY;
  const previousRak = process.env.ATLAS_SAAS_STRIPE_RAK;
  const previousPriceId = process.env.STRIPE_DEFLECTION_REPORT_PRICE_ID;
  const previousAccountId = process.env.ATLAS_ACCOUNT_ID;
  const calls = [];
  process.env.STRIPE_SECRET_KEY = "sk_live_full_secret";
  process.env.ATLAS_SAAS_STRIPE_RAK = "rk_test_checkout_only";
  process.env.STRIPE_DEFLECTION_REPORT_PRICE_ID = "price_deflection_report";
  process.env.ATLAS_ACCOUNT_ID = "2b2b950d-f64b-4852-bc30-f92a34cdf169";
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
    if (previousRak === undefined) {
      delete process.env.ATLAS_SAAS_STRIPE_RAK;
    } else {
      process.env.ATLAS_SAAS_STRIPE_RAK = previousRak;
    }
    if (previousPriceId === undefined) {
      delete process.env.STRIPE_DEFLECTION_REPORT_PRICE_ID;
    } else {
      process.env.STRIPE_DEFLECTION_REPORT_PRICE_ID = previousPriceId;
    }
    if (previousAccountId === undefined) {
      delete process.env.ATLAS_ACCOUNT_ID;
    } else {
      process.env.ATLAS_ACCOUNT_ID = previousAccountId;
    }
  }

  assert.equal(res.statusCode, 200);
  assert.deepEqual(JSON.parse(res.body), { url: "https://checkout.stripe.com/c/session" });
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, "https://api.stripe.com/v1/checkout/sessions");
  assert.equal(calls[0].options.body.get("line_items[0][price]"), "price_deflection_report");
});

await test("checkout key config prefers restricted key and rejects live secret fallback", () => {
  assert.deepEqual(
    stripeCheckoutKeyConfig({
      ATLAS_SAAS_STRIPE_RAK: "rk_test_checkout_only",
      STRIPE_SECRET_KEY: "sk_live_full_secret",
    }),
    { ok: true, key: "rk_test_checkout_only", source: "restricted" },
  );
  assert.deepEqual(
    stripeCheckoutKeyConfig({ STRIPE_SECRET_KEY: "sk_test_preview" }),
    { ok: true, key: "sk_test_preview", source: "fallback" },
  );
  assert.deepEqual(
    stripeCheckoutKeyConfig({ ATLAS_SAAS_STRIPE_SECRET_KEY: "sk_test_atlas_preview" }),
    { ok: true, key: "sk_test_atlas_preview", source: "fallback" },
  );
  assert.deepEqual(stripeCheckoutKeyConfig({ STRIPE_SECRET_KEY: "sk_live_full_secret" }), {
    ok: false,
    error: "checkout_restricted_key_required",
  });
  assert.deepEqual(stripeCheckoutKeyConfig({ ATLAS_SAAS_STRIPE_RAK: "sk_test_wrong" }), {
    ok: false,
    error: "checkout_restricted_key_invalid",
  });
});

await test("checkout return path preserves result identifiers", () => {
  const path = resultPath(
    "content-ops-abc123",
    "2b2b950d-f64b-4852-bc30-f92a34cdf169",
    "success",
  );
  assert.equal(
    path,
    "/services/faq-deflection/results/content-ops-abc123?checkout=success",
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
        summary: {
          generated: 1,
          drafted_answer_count: 0,
          no_proven_answer_count: 1,
          support_ticket_resolution_evidence_present: false,
          support_ticket_resolution_evidence_count: 0,
        },
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
  const previousRak = process.env.ATLAS_SAAS_STRIPE_RAK;
  const previousPriceId = process.env.STRIPE_DEFLECTION_REPORT_PRICE_ID;
  const previousAccountId = process.env.ATLAS_ACCOUNT_ID;
  const calls = [];
  process.env.STRIPE_SECRET_KEY = "sk_test_123";
  process.env.ATLAS_SAAS_STRIPE_RAK = "rk_test_checkout_only";
  process.env.ATLAS_ACCOUNT_ID = "2b2b950d-f64b-4852-bc30-f92a34cdf169";
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
    if (previousRak === undefined) {
      delete process.env.ATLAS_SAAS_STRIPE_RAK;
    } else {
      process.env.ATLAS_SAAS_STRIPE_RAK = previousRak;
    }
    if (previousPriceId === undefined) {
      delete process.env.STRIPE_DEFLECTION_REPORT_PRICE_ID;
    } else {
      process.env.STRIPE_DEFLECTION_REPORT_PRICE_ID = previousPriceId;
    }
    if (previousAccountId === undefined) {
      delete process.env.ATLAS_ACCOUNT_ID;
    } else {
      process.env.ATLAS_ACCOUNT_ID = previousAccountId;
    }
  }

  assert.equal(res.statusCode, 200);
  assert.deepEqual(JSON.parse(res.body), { url: "https://checkout.stripe.com/c/session" });
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, "https://api.stripe.com/v1/checkout/sessions");
  assert.equal(
    calls[0].options.headers.Authorization,
    `Basic ${Buffer.from("rk_test_checkout_only:").toString("base64")}`,
  );
  assert.equal(calls[0].options.body.get("metadata[source]"), CHECKOUT_SOURCE);
  assert.equal(calls[0].options.body.get("metadata[request_id]"), "content-ops-abc123");
});

await test("handler rejects live full secret fallback before calling Stripe", async () => {
  const previousFetch = globalThis.fetch;
  const previousSecret = process.env.STRIPE_SECRET_KEY;
  const previousRak = process.env.ATLAS_SAAS_STRIPE_RAK;
  const calls = [];
  process.env.STRIPE_SECRET_KEY = "sk_live_full_secret";
  delete process.env.ATLAS_SAAS_STRIPE_RAK;
  globalThis.fetch = async (url, options) => {
    calls.push({ url, options });
    throw new Error("Stripe should not be called");
  };

  const req = {
    method: "POST",
    headers: {
      host: "portfolio.example.com",
      "x-forwarded-proto": "https",
    },
    body: {
      request_id: "content-ops-abc123",
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
    if (previousRak === undefined) {
      delete process.env.ATLAS_SAAS_STRIPE_RAK;
    } else {
      process.env.ATLAS_SAAS_STRIPE_RAK = previousRak;
    }
  }

  assert.equal(res.statusCode, 503);
  assert.deepEqual(JSON.parse(res.body), { error: "checkout_restricted_key_required" });
  assert.equal(calls.length, 0);
});
