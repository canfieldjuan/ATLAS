import { timingSafeEqual } from "node:crypto";
import { atlasUrl, clean, configFromEnv } from "./atlas-report.js";
import {
  BLOB_UPLOAD_PATH_PREFIX,
  FULL_THREAD_IMPORTER_MODE,
  submitPrivateBlob,
} from "./submit.js";

const ZENDESK_EXPORT_PATH = "/api/v1/content-ops/zendesk-export/full-thread";
const DEFAULT_ZENDESK_EXPORT_LIMIT = 50;
const MAX_ZENDESK_EXPORT_LIMIT = 1000;
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const SAFE_EXPORT_ERROR_REASONS = new Set([
  "invalid_zendesk_export_submit",
  "zendesk_credentials_missing",
  "zendesk_credentials_unavailable",
  "zendesk_export_artifact_invalid",
  "zendesk_export_blob_contract_violation",
  "zendesk_export_blob_unavailable",
  "zendesk_export_contract_violation",
  "zendesk_export_failed",
  "zendesk_export_invalid_json",
  "zendesk_export_request_failed",
  "zendesk_export_unavailable",
]);

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

function zendeskCredentialFlowConfig(env = process.env) {
  const config = configFromEnv(env);
  const blobToken = clean(env.BLOB_READ_WRITE_TOKEN);
  const accessToken = clean(env.ATLAS_DEFLECTION_ZENDESK_EXPORT_ACCESS_TOKEN);
  const errors = [];
  if (!config.baseUrl) errors.push("atlas_base_url_missing");
  if (!config.token) errors.push("atlas_token_missing");
  if (!config.accountId || !UUID_RE.test(config.accountId)) {
    errors.push("atlas_account_missing");
  }
  if (!blobToken) errors.push("blob_token_missing");
  if (!accessToken) errors.push("zendesk_export_access_token_missing");
  return { config, blobToken, accessToken, errors };
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

function positiveInt(value, fallback, max = MAX_ZENDESK_EXPORT_LIMIT) {
  const raw = clean(value);
  if (!raw) return fallback;
  const parsed = Number.parseInt(raw, 10);
  if (!Number.isInteger(parsed) || parsed < 1 || parsed > max) return null;
  return parsed;
}

function nonNegativeInt(value, fallback = 0) {
  const raw = clean(value);
  if (!raw) return fallback;
  const parsed = Number.parseInt(raw, 10);
  if (!Number.isInteger(parsed) || parsed < 0) return null;
  return parsed;
}

function normalizePublicPayload(payload) {
  const companyName = clean(payload?.company_name);
  const contactEmail = clean(payload?.contact_email);
  const limit = positiveInt(payload?.limit, DEFAULT_ZENDESK_EXPORT_LIMIT);
  const startTime = nonNegativeInt(payload?.start_time, 0);
  const errors = [];
  if (!companyName) errors.push("company_name_required");
  if (!contactEmail) errors.push("contact_email_required");
  if (limit === null) errors.push("invalid_limit");
  if (startTime === null) errors.push("invalid_start_time");
  if (errors.length > 0) return { ok: false, errors };
  return {
    ok: true,
    companyName,
    contactEmail,
    limit,
    startTime,
  };
}

function safeExportErrorReason(payload) {
  const reason = clean(payload?.detail?.reason) || clean(payload?.reason) || clean(payload?.error);
  return SAFE_EXPORT_ERROR_REASONS.has(reason) ? reason : "zendesk_export_failed";
}

function safeSecretEqual(left, right) {
  const leftBuffer = Buffer.from(clean(left));
  const rightBuffer = Buffer.from(clean(right));
  if (leftBuffer.length === 0 || leftBuffer.length !== rightBuffer.length) return false;
  return timingSafeEqual(leftBuffer, rightBuffer);
}

function requestBearerToken(req) {
  const authorization = header(req, "authorization");
  const match = authorization.match(/^Bearer\s+(.+)$/i);
  return clean(match?.[1]);
}

function authorizeZendeskCredentialFlow(req, accessToken) {
  return safeSecretEqual(requestBearerToken(req), accessToken);
}

function validateZendeskExportPayload(payload) {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    return null;
  }
  if (
    payload.importer_mode !== FULL_THREAD_IMPORTER_MODE ||
    payload.support_platform !== "zendesk" ||
    !payload.artifact ||
    typeof payload.artifact !== "object" ||
    Array.isArray(payload.artifact) ||
    !Array.isArray(payload.artifact.tickets)
  ) {
    return null;
  }
  return payload.artifact;
}

async function exportZendeskArtifact({
  config,
  limit,
  startTime,
  fetchImpl = fetch,
}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), config.timeoutMs);
  let response;
  try {
    response = await fetchImpl(atlasUrl(config.baseUrl, ZENDESK_EXPORT_PATH), {
      method: "POST",
      headers: {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": `Bearer ${config.token}`,
      },
      body: JSON.stringify({ limit, start_time: startTime }),
      signal: controller.signal,
    });
  } catch {
    return { ok: false, statusCode: 502, error: "zendesk_export_unavailable" };
  } finally {
    clearTimeout(timeout);
  }

  const text = await response.text();
  let payload = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      return { ok: false, statusCode: 502, error: "zendesk_export_invalid_json" };
    }
  }
  if (!response.ok) {
    return {
      ok: false,
      statusCode: response.status >= 400 && response.status < 500 ? response.status : 502,
      error: safeExportErrorReason(payload),
      atlas_status: response.status,
    };
  }
  const artifact = validateZendeskExportPayload(payload);
  if (!artifact) {
    return { ok: false, statusCode: 502, error: "zendesk_export_contract_violation" };
  }
  return { ok: true, artifact };
}

async function putPrivateJsonArtifact(pathname, body, options) {
  const { put } = await import("@vercel/blob");
  return put(pathname, body, options);
}

function exportBlobPathname(nowMs = Date.now()) {
  return `${BLOB_UPLOAD_PATH_PREFIX}${nowMs}-zendesk-api-export.json`;
}

async function persistZendeskArtifact({
  artifact,
  blobToken,
  putBlobImpl = putPrivateJsonArtifact,
  nowMs = Date.now(),
}) {
  const pathname = exportBlobPathname(nowMs);
  const body = Buffer.from(JSON.stringify(artifact), "utf8");
  try {
    const result = await putBlobImpl(pathname, body, {
      access: "private",
      addRandomSuffix: true,
      contentType: "application/json",
      token: blobToken,
    });
    const resultPathname = clean(result?.pathname);
    if (!resultPathname) {
      return { ok: false, statusCode: 502, error: "zendesk_export_blob_contract_violation" };
    }
    return { ok: true, pathname: resultPathname };
  } catch {
    return { ok: false, statusCode: 502, error: "zendesk_export_blob_unavailable" };
  }
}

async function submitZendeskCredentialFlow({
  config,
  blobToken,
  payload,
  fetchImpl = fetch,
  putBlobImpl = putPrivateJsonArtifact,
  getBlobImpl,
  deleteBlobImpl,
  eventLogger = console.warn,
  nowMs = Date.now(),
}) {
  const publicPayload = normalizePublicPayload(payload);
  if (!publicPayload.ok) {
    return {
      ok: false,
      statusCode: 400,
      error: "invalid_zendesk_export_submit",
      details: publicPayload.errors,
    };
  }

  const exported = await exportZendeskArtifact({
    config,
    limit: publicPayload.limit,
    startTime: publicPayload.startTime,
    fetchImpl,
  });
  if (!exported.ok) return exported;

  const persisted = await persistZendeskArtifact({
    artifact: exported.artifact,
    blobToken,
    putBlobImpl,
    nowMs,
  });
  if (!persisted.ok) return persisted;

  return submitPrivateBlob({
    config,
    blobToken,
    payload: {
      blob_pathname: persisted.pathname,
      importer_mode: FULL_THREAD_IMPORTER_MODE,
      support_platform: "zendesk",
      company_name: publicPayload.companyName,
      contact_email: publicPayload.contactEmail,
      limit: String(publicPayload.limit),
    },
    fetchImpl,
    getBlobImpl,
    deleteBlobImpl,
    eventLogger,
  });
}

export {
  DEFAULT_ZENDESK_EXPORT_LIMIT,
  MAX_ZENDESK_EXPORT_LIMIT,
  ZENDESK_EXPORT_PATH,
  exportBlobPathname,
  exportZendeskArtifact,
  authorizeZendeskCredentialFlow,
  normalizePublicPayload,
  persistZendeskArtifact,
  submitZendeskCredentialFlow,
  validateZendeskExportPayload,
  zendeskCredentialFlowConfig,
};

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    json(res, 405, { ok: false, error: "method_not_allowed" });
    return;
  }

  const contentType = header(req, "content-type");
  if (!contentType.includes("application/json")) {
    json(res, 415, { ok: false, error: "zendesk_export_json_required" });
    return;
  }

  const { config, blobToken, accessToken, errors } = zendeskCredentialFlowConfig();
  if (errors.length > 0) {
    json(res, 503, { ok: false, error: "zendesk_export_not_configured" });
    return;
  }

  if (!authorizeZendeskCredentialFlow(req, accessToken)) {
    json(res, 401, { ok: false, error: "zendesk_export_auth_required" });
    return;
  }

  const requestedAccountId = header(req, "x-atlas-account-id");
  if (
    requestedAccountId &&
    (!UUID_RE.test(requestedAccountId) || requestedAccountId !== config.accountId)
  ) {
    json(res, 400, { ok: false, error: "invalid_account_id" });
    return;
  }

  let payload;
  try {
    payload = await readJsonBody(req);
  } catch {
    json(res, 400, { ok: false, error: "invalid_zendesk_export_submit" });
    return;
  }

  const result = await submitZendeskCredentialFlow({
    config,
    blobToken,
    payload,
  });
  if (!result.ok) {
    json(res, result.statusCode, {
      ok: false,
      error: result.error,
      atlas_status: result.atlas_status,
      details: result.details,
    });
    return;
  }
  json(res, 200, result.payload);
}
