import { atlasUrl, clean, configFromEnv } from "./atlas-report.js";
import { resultPath } from "./checkout.js";

const REQUEST_ID_RE = /^[A-Za-z0-9][A-Za-z0-9_.:-]{5,160}$/;
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const SUBMIT_PATH = "/api/v1/content-ops/deflection-reports/submit";
const MAX_CSV_BYTES = 4 * 1024 * 1024;
const MAX_MULTIPART_OVERHEAD_BYTES = 256 * 1024;

export const config = {
  api: {
    bodyParser: false,
  },
};

function json(res, statusCode, payload) {
  res.statusCode = statusCode;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.setHeader("Cache-Control", "no-store");
  res.end(JSON.stringify(payload));
}

function header(req, name) {
  const lower = name.toLowerCase();
  return clean(req.headers?.[lower]) || clean(req.headers?.[name]);
}

function publicSubmitConfig(env = process.env) {
  const config = configFromEnv(env);
  const errors = [];
  if (!config.baseUrl) errors.push("atlas_base_url_missing");
  if (!config.token) errors.push("atlas_token_missing");
  if (!config.accountId || !UUID_RE.test(config.accountId)) {
    errors.push("atlas_account_missing");
  }
  return { config, errors };
}

function rejectOversizeContentLength(req) {
  const raw = header(req, "content-length");
  if (!raw) return null;
  const length = Number.parseInt(raw, 10);
  if (!Number.isFinite(length)) return null;
  if (length > MAX_CSV_BYTES + MAX_MULTIPART_OVERHEAD_BYTES) {
    return {
      ok: false,
      statusCode: 413,
      error: "deflection_submit_csv_too_large",
    };
  }
  return null;
}

async function readRawBody(req, maxBytes = MAX_CSV_BYTES + MAX_MULTIPART_OVERHEAD_BYTES) {
  if (Buffer.isBuffer(req.body)) {
    if (req.body.length > maxBytes) return { ok: false, statusCode: 413 };
    return { ok: true, body: req.body };
  }
  if (typeof req.body === "string") {
    const body = Buffer.from(req.body);
    if (body.length > maxBytes) return { ok: false, statusCode: 413 };
    return { ok: true, body };
  }
  if (req.body instanceof ArrayBuffer) {
    const body = Buffer.from(req.body);
    if (body.length > maxBytes) return { ok: false, statusCode: 413 };
    return { ok: true, body };
  }

  const chunks = [];
  let total = 0;
  try {
    for await (const chunk of req) {
      const buffer = Buffer.from(chunk);
      total += buffer.length;
      if (total > maxBytes) return { ok: false, statusCode: 413 };
      chunks.push(buffer);
    }
  } catch {
    return { ok: false, statusCode: 400 };
  }
  return { ok: true, body: Buffer.concat(chunks) };
}

function projectSubmitPayload(payload, accountId) {
  const requestId = clean(payload?.request_id);
  if (!requestId || !REQUEST_ID_RE.test(requestId)) {
    return null;
  }
  return {
    ok: true,
    request_id: requestId,
    account_id: accountId,
    result_path: resultPath(requestId, accountId, ""),
  };
}

async function forwardSubmit({ config, contentType, body, fetchImpl = fetch }) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), config.timeoutMs);
  let response;
  try {
    response = await fetchImpl(atlasUrl(config.baseUrl, SUBMIT_PATH), {
      method: "POST",
      headers: {
        "Accept": "application/json",
        "Authorization": `Bearer ${config.token}`,
        "Content-Type": contentType,
      },
      body,
      signal: controller.signal,
    });
  } catch (error) {
    return {
      ok: false,
      statusCode: error?.name === "AbortError" ? 504 : 502,
      error: "atlas_submit_unreachable",
    };
  } finally {
    clearTimeout(timeout);
  }
  const text = await response.text();
  let payload = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      return { ok: false, statusCode: 502, error: "atlas_submit_invalid_json" };
    }
  }
  if (!response.ok) {
    return {
      ok: false,
      statusCode: response.status >= 400 && response.status < 500 ? response.status : 502,
      error: "atlas_submit_failed",
      atlas_status: response.status,
    };
  }
  const projected = projectSubmitPayload(payload, config.accountId);
  if (!projected) {
    return { ok: false, statusCode: 502, error: "atlas_submit_contract_violation" };
  }
  return { ok: true, statusCode: 200, payload: projected };
}

export {
  MAX_CSV_BYTES,
  MAX_MULTIPART_OVERHEAD_BYTES,
  SUBMIT_PATH,
  forwardSubmit,
  projectSubmitPayload,
};

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    json(res, 405, { ok: false, error: "method_not_allowed" });
    return;
  }

  const contentType = header(req, "content-type");
  if (!contentType.includes("multipart/form-data")) {
    json(res, 415, { ok: false, error: "multipart_required" });
    return;
  }

  const { config: submitConfig, errors } = publicSubmitConfig();
  if (errors.length > 0) {
    json(res, 503, { ok: false, error: "atlas_submit_not_configured" });
    return;
  }

  const accountId = header(req, "x-atlas-account-id");
  if (!accountId || !UUID_RE.test(accountId) || accountId !== submitConfig.accountId) {
    json(res, 400, { ok: false, error: "invalid_account_id" });
    return;
  }

  const oversize = rejectOversizeContentLength(req);
  if (oversize) {
    json(res, oversize.statusCode, { ok: false, error: oversize.error });
    return;
  }

  const bodyResult = await readRawBody(req);
  if (!bodyResult.ok) {
    json(res, bodyResult.statusCode, {
      ok: false,
      error:
        bodyResult.statusCode === 413
          ? "deflection_submit_csv_too_large"
          : "deflection_submit_body_unreadable",
    });
    return;
  }

  let result;
  try {
    result = await forwardSubmit({ config: submitConfig, contentType, body: bodyResult.body });
  } catch {
    json(res, 502, { ok: false, error: "atlas_submit_unreachable" });
    return;
  }

  if (!result.ok) {
    json(res, result.statusCode, {
      ok: false,
      error: result.error,
      atlas_status: result.atlas_status,
    });
    return;
  }
  json(res, 200, result.payload);
}
