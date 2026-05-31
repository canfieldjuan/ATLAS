const CHECKOUT_SOURCE = "content_ops_deflection_report";
const DEFAULT_AMOUNT_CENTS = 150000;
const DEFAULT_CURRENCY = "usd";
const REQUEST_ID_RE = /^[A-Za-z0-9][A-Za-z0-9_.:-]{5,160}$/;
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

function json(res, statusCode, payload) {
  res.statusCode = statusCode;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.end(JSON.stringify(payload));
}

function clean(value) {
  return typeof value === "string" ? value.trim() : "";
}

function parseAmount(value) {
  const parsed = Number.parseInt(clean(value), 10);
  return Number.isFinite(parsed) && parsed >= DEFAULT_AMOUNT_CENTS
    ? parsed
    : DEFAULT_AMOUNT_CENTS;
}

function normalizeCurrency(value) {
  const normalized = clean(value).toLowerCase();
  return /^[a-z]{3}$/.test(normalized) ? normalized : DEFAULT_CURRENCY;
}

function stripeAuthHeaders(stripeSecretKey) {
  return {
    "Authorization": `Basic ${Buffer.from(`${stripeSecretKey}:`).toString("base64")}`,
  };
}

function stripeCheckoutKeyConfig(env = process.env) {
  const restrictedKey = clean(env.ATLAS_SAAS_STRIPE_RAK);
  if (restrictedKey) {
    if (!restrictedKey.startsWith("rk_")) {
      return { ok: false, error: "checkout_restricted_key_invalid" };
    }
    return { ok: true, key: restrictedKey, source: "restricted" };
  }

  const fallbackKey = clean(env.STRIPE_SECRET_KEY || env.ATLAS_SAAS_STRIPE_SECRET_KEY);
  if (!fallbackKey) {
    return { ok: false, error: "checkout_not_configured" };
  }
  if (fallbackKey.startsWith("sk_live_")) {
    return { ok: false, error: "checkout_restricted_key_required" };
  }
  return { ok: true, key: fallbackKey, source: "fallback" };
}

function publicBaseUrl(req) {
  const configured = clean(process.env.DEFLECTION_CHECKOUT_PUBLIC_BASE_URL);
  if (configured) return configured.replace(/\/+$/, "");
  const host = clean(req.headers?.["x-forwarded-host"]) || clean(req.headers?.host);
  if (host) {
    const proto = clean(req.headers?.["x-forwarded-proto"]) || "https";
    return `${proto}://${host}`.replace(/\/+$/, "");
  }
  const vercelUrl = clean(process.env.VERCEL_URL);
  return vercelUrl ? `https://${vercelUrl}`.replace(/\/+$/, "") : "";
}

async function readBody(req) {
  if (req.body && typeof req.body === "object") return req.body;
  if (typeof req.body === "string") return JSON.parse(req.body);

  const chunks = [];
  for await (const chunk of req) chunks.push(Buffer.from(chunk));
  const raw = Buffer.concat(chunks).toString("utf8");
  return raw ? JSON.parse(raw) : {};
}

function validatePayload(payload) {
  const requestId = clean(payload?.request_id);
  const accountId = clean(payload?.account_id);
  const errors = [];
  if (!requestId || !REQUEST_ID_RE.test(requestId)) {
    errors.push("request_id must be a valid Content Ops request id");
  }
  if (!accountId || !UUID_RE.test(accountId)) {
    errors.push("account_id must be a valid ATLAS account UUID");
  }
  return { requestId, accountId, errors };
}

function resultPath(requestId, accountId, checkout) {
  const path = `/services/faq-deflection/results/${encodeURIComponent(requestId)}`;
  const params = new URLSearchParams({ account_id: accountId });
  if (checkout) params.set("checkout", checkout);
  return `${path}?${params.toString()}`;
}

function checkoutUrls(req, requestId, accountId) {
  const baseUrl = publicBaseUrl(req);
  if (!baseUrl) return { errors: ["public base URL could not be resolved"] };
  const successUrl =
    clean(process.env.DEFLECTION_CHECKOUT_SUCCESS_URL) ||
    `${baseUrl}${resultPath(requestId, accountId, "success")}`;
  const cancelUrl =
    clean(process.env.DEFLECTION_CHECKOUT_CANCEL_URL) ||
    `${baseUrl}${resultPath(requestId, accountId, "cancel")}`;
  return { successUrl, cancelUrl, errors: [] };
}

function buildStripeCheckoutBody({ requestId, accountId, successUrl, cancelUrl }) {
  const params = new URLSearchParams();
  params.set("mode", "payment");
  params.set("success_url", successUrl);
  params.set("cancel_url", cancelUrl);
  params.set("metadata[source]", CHECKOUT_SOURCE);
  params.set("metadata[account_id]", accountId);
  params.set("metadata[request_id]", requestId);
  params.set("payment_intent_data[metadata][source]", CHECKOUT_SOURCE);
  params.set("payment_intent_data[metadata][account_id]", accountId);
  params.set("payment_intent_data[metadata][request_id]", requestId);
  params.set("client_reference_id", requestId);
  params.set("line_items[0][quantity]", "1");

  const priceId = clean(process.env.STRIPE_DEFLECTION_REPORT_PRICE_ID);
  if (priceId) {
    params.set("line_items[0][price]", priceId);
    return params;
  }

  const amount = parseAmount(process.env.DEFLECTION_REPORT_AMOUNT_CENTS);
  const currency = normalizeCurrency(process.env.DEFLECTION_REPORT_CURRENCY);
  params.set("line_items[0][price_data][currency]", currency);
  params.set("line_items[0][price_data][unit_amount]", String(amount));
  params.set("line_items[0][price_data][product_data][name]", "FAQ Deflection Report");
  params.set(
    "line_items[0][price_data][product_data][description]",
    "One-time unlock for the full support-ticket FAQ deflection report.",
  );
  return params;
}

async function validateConfiguredPrice(stripeSecretKey) {
  const priceId = clean(process.env.STRIPE_DEFLECTION_REPORT_PRICE_ID);
  if (!priceId) return { ok: true };

  const response = await fetch(
    `https://api.stripe.com/v1/prices/${encodeURIComponent(priceId)}`,
    {
      headers: stripeAuthHeaders(stripeSecretKey),
    },
  );
  const payload = await response.json().catch(() => null);
  if (!response.ok || !payload || typeof payload !== "object") {
    return {
      ok: false,
      message: "configured Stripe Price could not be validated",
    };
  }
  if (
    payload.active === false ||
    payload.currency !== DEFAULT_CURRENCY ||
    payload.unit_amount < DEFAULT_AMOUNT_CENTS
  ) {
    return {
      ok: false,
      message: "configured Stripe Price must be active, usd, and at least 150000 cents",
    };
  }
  return { ok: true };
}

async function createCheckoutSession(params, stripeSecretKey) {
  const response = await fetch("https://api.stripe.com/v1/checkout/sessions", {
    method: "POST",
    headers: {
      ...stripeAuthHeaders(stripeSecretKey),
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: params,
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok || !payload || typeof payload.url !== "string") {
    const message =
      payload && typeof payload.error?.message === "string"
        ? payload.error.message
        : "Stripe Checkout session could not be created";
    return { ok: false, status: response.status, message };
  }
  return { ok: true, url: payload.url };
}

export {
  CHECKOUT_SOURCE,
  buildStripeCheckoutBody,
  checkoutUrls,
  resultPath,
  stripeCheckoutKeyConfig,
  validateConfiguredPrice,
  validatePayload,
};

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    json(res, 405, { error: "method_not_allowed" });
    return;
  }

  const stripeKey = stripeCheckoutKeyConfig();
  if (!stripeKey.ok) {
    json(res, 503, { error: stripeKey.error });
    return;
  }

  let payload;
  try {
    payload = await readBody(req);
  } catch {
    json(res, 400, { error: "invalid_json" });
    return;
  }

  const { requestId, accountId, errors } = validatePayload(payload);
  if (errors.length > 0) {
    json(res, 400, { error: "invalid_checkout_request", details: errors });
    return;
  }

  const urls = checkoutUrls(req, requestId, accountId);
  if (urls.errors.length > 0) {
    json(res, 503, { error: "checkout_url_not_configured", details: urls.errors });
    return;
  }
  if (stripeKey.source !== "restricted") {
    const priceValidation = await validateConfiguredPrice(stripeKey.key);
    if (!priceValidation.ok) {
      json(res, 503, { error: "checkout_price_not_configured", details: priceValidation.message });
      return;
    }
  }

  const body = buildStripeCheckoutBody({
    requestId,
    accountId,
    successUrl: urls.successUrl,
    cancelUrl: urls.cancelUrl,
  });
  const session = await createCheckoutSession(body, stripeKey.key);
  if (!session.ok) {
    json(res, 502, { error: "stripe_checkout_failed", details: session.message });
    return;
  }

  json(res, 200, { url: session.url });
}
