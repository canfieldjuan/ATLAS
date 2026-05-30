import { atlasUrl, clean, configFromEnv } from "./atlas-report.js";
import { resultPath } from "./checkout.js";

const REQUEST_ID_RE = /^[A-Za-z0-9][A-Za-z0-9_.:-]{5,160}$/;
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const SUBMIT_PATH = "/api/v1/content-ops/deflection-reports/submit";
const MAX_CSV_BYTES = 4 * 1024 * 1024;
const MAX_BLOB_CSV_BYTES = 50 * 1024 * 1024;
const MAX_MULTIPART_OVERHEAD_BYTES = 256 * 1024;
const BLOB_UPLOAD_PATH_PREFIX = "faq-deflection/uploads/";

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

function privateBlobSubmitConfig(env = process.env) {
  const result = publicSubmitConfig(env);
  const blobToken = clean(env.BLOB_READ_WRITE_TOKEN);
  if (!blobToken) result.errors.push("blob_token_missing");
  return { ...result, blobToken };
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

async function readJsonBody(req) {
  if (req.body && typeof req.body === "object" && !Buffer.isBuffer(req.body)) return req.body;
  if (typeof req.body === "string") return JSON.parse(req.body);
  if (Buffer.isBuffer(req.body)) return JSON.parse(req.body.toString("utf8"));
  const chunks = [];
  for await (const chunk of req) chunks.push(Buffer.from(chunk));
  const raw = Buffer.concat(chunks).toString("utf8");
  return raw ? JSON.parse(raw) : {};
}

async function streamToBuffer(stream, maxBytes) {
  const reader = stream.getReader();
  const chunks = [];
  let total = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      const buffer = Buffer.from(value);
      total += buffer.length;
      if (total > maxBytes) return { ok: false, statusCode: 413 };
      chunks.push(buffer);
    }
  } finally {
    reader.releaseLock();
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
  const headers = {
    "Accept": "application/json",
    "Authorization": `Bearer ${config.token}`,
  };
  if (contentType) headers["Content-Type"] = contentType;
  let response;
  try {
    response = await fetchImpl(atlasUrl(config.baseUrl, SUBMIT_PATH), {
      method: "POST",
      headers,
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

function validBlobPathname(value) {
  const pathname = clean(value);
  if (!pathname.startsWith(BLOB_UPLOAD_PATH_PREFIX)) return "";
  if (!pathname.toLowerCase().endsWith(".csv")) return "";
  if (pathname.includes("..") || pathname.includes("\\") || pathname.startsWith("/")) return "";
  return pathname;
}

function fileNameFromPathname(pathname) {
  const name = pathname.split("/").filter(Boolean).pop() || "tickets.csv";
  return name.toLowerCase().endsWith(".csv") ? name : "tickets.csv";
}

async function getPrivateBlob(pathname, options) {
  const { get } = await import("@vercel/blob");
  return get(pathname, options);
}

async function readPrivateCsvBlob({
  pathname,
  token,
  getBlobImpl = getPrivateBlob,
  maxBytes = MAX_BLOB_CSV_BYTES,
}) {
  const safePathname = validBlobPathname(pathname);
  if (!safePathname) return { ok: false, statusCode: 400, error: "invalid_blob_reference" };
  try {
    const result = await getBlobImpl(safePathname, {
      access: "private",
      token,
      useCache: false,
    });
    if (!result || result.statusCode !== 200 || !result.stream) {
      return { ok: false, statusCode: 404, error: "private_blob_unavailable" };
    }
    if (Number.isFinite(result.blob?.size) && result.blob.size > maxBytes) {
      return { ok: false, statusCode: 413, error: "deflection_submit_csv_too_large" };
    }
    const body = await streamToBuffer(result.stream, maxBytes);
    if (!body.ok) {
      return { ok: false, statusCode: 413, error: "deflection_submit_csv_too_large" };
    }
    return {
      ok: true,
      body: body.body,
      contentType: clean(result.blob?.contentType) || "text/csv",
      fileName: fileNameFromPathname(safePathname),
    };
  } catch {
    return { ok: false, statusCode: 502, error: "private_blob_unavailable" };
  }
}

async function submitPrivateBlob({
  config,
  blobToken,
  payload,
  fetchImpl = fetch,
  getBlobImpl = getPrivateBlob,
}) {
  const blob = await readPrivateCsvBlob({
    pathname: payload?.blob_pathname,
    token: blobToken,
    getBlobImpl,
  });
  if (!blob.ok) return blob;

  const form = new FormData();
  form.set("csv_file", new Blob([blob.body], { type: blob.contentType }), blob.fileName);
  form.set("support_platform", clean(payload?.support_platform));
  form.set("company_name", clean(payload?.company_name));
  form.set("contact_email", clean(payload?.contact_email));
  form.set("limit", clean(payload?.limit) || "1000");

  return forwardSubmit({ config, contentType: "", body: form, fetchImpl });
}

export {
  BLOB_UPLOAD_PATH_PREFIX,
  MAX_BLOB_CSV_BYTES,
  MAX_CSV_BYTES,
  MAX_MULTIPART_OVERHEAD_BYTES,
  SUBMIT_PATH,
  forwardSubmit,
  projectSubmitPayload,
  readPrivateCsvBlob,
  submitPrivateBlob,
  validBlobPathname,
};

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    json(res, 405, { ok: false, error: "method_not_allowed" });
    return;
  }

  const contentType = header(req, "content-type");
  const isPrivateBlobSubmit = contentType.includes("application/json");
  if (!contentType.includes("multipart/form-data") && !isPrivateBlobSubmit) {
    json(res, 415, { ok: false, error: "multipart_required" });
    return;
  }

  const { config: submitConfig, blobToken, errors } = isPrivateBlobSubmit
    ? privateBlobSubmitConfig()
    : publicSubmitConfig();
  if (errors.length > 0) {
    json(res, 503, { ok: false, error: "atlas_submit_not_configured" });
    return;
  }

  const accountId = header(req, "x-atlas-account-id");
  if (!accountId || !UUID_RE.test(accountId) || accountId !== submitConfig.accountId) {
    json(res, 400, { ok: false, error: "invalid_account_id" });
    return;
  }

  if (isPrivateBlobSubmit) {
    let payload;
    try {
      payload = await readJsonBody(req);
    } catch {
      json(res, 400, { ok: false, error: "invalid_blob_submit" });
      return;
    }

    const result = await submitPrivateBlob({
      config: submitConfig,
      blobToken,
      payload,
    });
    if (!result.ok) {
      json(res, result.statusCode, {
        ok: false,
        error: result.error,
        atlas_status: result.atlas_status,
      });
      return;
    }
    json(res, 200, result.payload);
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
