const CHECKOUT_SOURCE = "content_ops_deflection_report";
const DEFAULT_AMOUNT_CENTS = 150000;
const DEFAULT_CURRENCY = "usd";
const DEFAULT_ATLAS_TIMEOUT_MS = 5000;
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

function atlasConfigFromEnv(env = process.env) {
  const timeoutMs = Number.parseInt(clean(env.ATLAS_PROXY_TIMEOUT_MS), 10);
  return {
    baseUrl: clean(env.ATLAS_API_BASE_URL),
    token: clean(env.ATLAS_B2B_JWT || env.ATLAS_TOKEN),
    timeoutMs:
      Number.isFinite(timeoutMs) && timeoutMs > 0 ? timeoutMs : DEFAULT_ATLAS_TIMEOUT_MS,
  };
}

function checkoutAuthorizationConfigErrors(config) {
  const errors = [];
  if (!config.baseUrl) errors.push("ATLAS_API_BASE_URL is not configured");
  if (!config.token) errors.push("ATLAS_B2B_JWT is not configured");
  return errors;
}

function atlasUrl(baseUrl, path) {
  return `${baseUrl.replace(/\/+$/, "")}/${path.replace(/^\/+/, "")}`;
}

function checkoutAuthorizationPath(requestId) {
  return `/api/v1/content-ops/deflection-reports/${encodeURIComponent(requestId)}/checkout-authorization`;
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

function validatePayload(payload, env = process.env) {
  const requestId = clean(payload?.request_id);
  const requestedAccountId = clean(payload?.account_id);
  const accountId = clean(env.ATLAS_ACCOUNT_ID);
  const errors = [];
  if (!requestId || !REQUEST_ID_RE.test(requestId)) {
    errors.push("request_id must be a valid Content Ops request id");
  }
  if (!accountId || !UUID_RE.test(accountId)) {
    errors.push("ATLAS_ACCOUNT_ID must be configured as an ATLAS account UUID");
  }
  if (requestedAccountId && !UUID_RE.test(requestedAccountId)) {
    errors.push("account_id must be a valid ATLAS account UUID");
  } else if (requestedAccountId && requestedAccountId !== accountId) {
    errors.push("account_id does not match the configured ATLAS account");
  }
  return { requestId, accountId, errors };
}

function resultPath(requestId, _accountId = "", checkout = "") {
  const path = `/services/faq-deflection/results/${encodeURIComponent(requestId)}`;
  const params = new URLSearchParams();
  if (checkout) params.set("checkout", checkout);
  const query = params.toString();
  return query ? `${path}?${query}` : path;
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

function normalizeCheckoutTerms(terms = {}) {
  const source = terms && typeof terms === "object" ? terms : {};
  const priceId = clean(source.price_id);
  const amount = Number.parseInt(String(source.amount_cents ?? ""), 10);
  const currency = clean(source.currency).toLowerCase();
  return {
    priceId,
    amount: Number.isFinite(amount) && amount > 0 ? amount : null,
    currency: /^[a-z]{3}$/.test(currency) ? currency : "",
  };
}

function stripeCheckoutIdempotencyKey({ requestId, accountId }) {
  return `deflection-checkout:${accountId}:${requestId}`;
}

function stripeCheckoutPriceId(checkout = null, env = process.env) {
  const terms = normalizeCheckoutTerms(checkout);
  return terms.priceId || clean(env.STRIPE_DEFLECTION_REPORT_PRICE_ID);
}

function stripeCheckoutInlineTerms(checkout = null, env = process.env) {
  const terms = normalizeCheckoutTerms(checkout);
  return {
    amount: terms.amount || parseAmount(env.DEFLECTION_REPORT_AMOUNT_CENTS),
    currency: terms.currency || normalizeCurrency(env.DEFLECTION_REPORT_CURRENCY),
  };
}

function validateInlineCheckoutTerms({ amount, currency }) {
  if (currency !== DEFAULT_CURRENCY || amount < DEFAULT_AMOUNT_CENTS) {
    return {
      ok: false,
      message: "configured inline Checkout amount must be usd and at least 150000 cents",
    };
  }
  return { ok: true };
}

function buildStripeCheckoutBody({
  requestId,
  accountId,
  successUrl,
  cancelUrl,
  checkout = null,
}) {
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

  const priceId = stripeCheckoutPriceId(checkout);
  if (priceId) {
    params.set("line_items[0][price]", priceId);
    return params;
  }

  const { amount, currency } = stripeCheckoutInlineTerms(checkout);
  params.set("line_items[0][price_data][currency]", currency);
  params.set("line_items[0][price_data][unit_amount]", String(amount));
  params.set("line_items[0][price_data][product_data][name]", "FAQ Deflection Report");
  params.set(
    "line_items[0][price_data][product_data][description]",
    "One-time unlock for the full support-ticket FAQ deflection report.",
  );
  return params;
}

async function validateConfiguredPrice(
  stripeSecretKey,
  priceId = process.env.STRIPE_DEFLECTION_REPORT_PRICE_ID,
) {
  priceId = clean(priceId);
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

async function authorizeCheckout({ requestId, env = process.env, fetchImpl = fetch }) {
  const config = atlasConfigFromEnv(env);
  const configErrors = checkoutAuthorizationConfigErrors(config);
  if (configErrors.length > 0) {
    return {
      ok: false,
      status: 503,
      error: "checkout_authorization_not_configured",
      details: configErrors,
    };
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), config.timeoutMs);
  try {
    const response = await fetchImpl(
      atlasUrl(config.baseUrl, checkoutAuthorizationPath(requestId)),
      {
        method: "POST",
        headers: {
          "Accept": "application/json",
          "Authorization": `Bearer ${config.token}`,
        },
        signal: controller.signal,
      },
    );
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      return {
        ok: false,
        status: response.status,
        error: "checkout_not_authorized",
        details:
          payload && typeof payload.detail === "string"
            ? payload.detail
            : "Deflection report is not ready for checkout.",
      };
    }
    if (
      !payload ||
      payload.status !== "authorized" ||
      !payload.checkout ||
      typeof payload.checkout !== "object"
    ) {
      return {
        ok: false,
        status: 502,
        error: "checkout_authorization_contract_violation",
      };
    }
    return { ok: true, checkout: payload.checkout };
  } catch (error) {
    return {
      ok: false,
      status: 502,
      error: "checkout_authorization_failed",
      details: error && typeof error === "object" ? error.name || "fetch_error" : "fetch_error",
    };
  } finally {
    clearTimeout(timeout);
  }
}

async function createCheckoutSession(params, stripeSecretKey, idempotencyKey) {
  const response = await fetch("https://api.stripe.com/v1/checkout/sessions", {
    method: "POST",
    headers: {
      ...stripeAuthHeaders(stripeSecretKey),
      "Content-Type": "application/x-www-form-urlencoded",
      "Idempotency-Key": idempotencyKey,
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
  authorizeCheckout,
  buildStripeCheckoutBody,
  checkoutUrls,
  resultPath,
  stripeCheckoutKeyConfig,
  stripeCheckoutIdempotencyKey,
  stripeCheckoutInlineTerms,
  stripeCheckoutPriceId,
  validateInlineCheckoutTerms,
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

  const { requestId, accountId, errors } = validatePayload(payload, process.env);
  if (errors.length > 0) {
    json(res, 400, { error: "invalid_checkout_request", details: errors });
    return;
  }

  const urls = checkoutUrls(req, requestId, accountId);
  if (urls.errors.length > 0) {
    json(res, 503, { error: "checkout_url_not_configured", details: urls.errors });
    return;
  }

  const authorization = await authorizeCheckout({ requestId });
  if (!authorization.ok) {
    json(res, authorization.status, {
      error: authorization.error,
      details: authorization.details,
    });
    return;
  }

  if (stripeKey.source !== "restricted") {
    const priceValidation = await validateConfiguredPrice(
      stripeKey.key,
      stripeCheckoutPriceId(authorization.checkout),
    );
    if (!priceValidation.ok) {
      json(res, 503, { error: "checkout_price_not_configured", details: priceValidation.message });
      return;
    }
  }
  if (!stripeCheckoutPriceId(authorization.checkout)) {
    const inlineValidation = validateInlineCheckoutTerms(
      stripeCheckoutInlineTerms(authorization.checkout),
    );
    if (!inlineValidation.ok) {
      json(res, 503, { error: "checkout_price_not_configured", details: inlineValidation.message });
      return;
    }
  }

  const body = buildStripeCheckoutBody({
    requestId,
    accountId,
    successUrl: urls.successUrl,
    cancelUrl: urls.cancelUrl,
    checkout: authorization.checkout,
  });
  const session = await createCheckoutSession(
    body,
    stripeKey.key,
    stripeCheckoutIdempotencyKey({ requestId, accountId }),
  );
  if (!session.ok) {
    json(res, 502, { error: "stripe_checkout_failed", details: session.message });
    return;
  }

  json(res, 200, { url: session.url });
}
