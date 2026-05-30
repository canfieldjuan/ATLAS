import { clean } from "./atlas-report.js";

const BLOB_UPLOAD_PATH_PREFIX = "faq-deflection/uploads/";
const MAX_BLOB_CSV_BYTES = 50 * 1024 * 1024;
const CSV_CONTENT_TYPES = ["text/csv", "application/vnd.ms-excel", "application/csv"];
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

function json(res, statusCode, payload) {
  res.statusCode = statusCode;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.setHeader("Cache-Control", "no-store");
  res.end(JSON.stringify(payload));
}

async function readJsonBody(req) {
  if (req.body && typeof req.body === "object") return req.body;
  if (typeof req.body === "string") return JSON.parse(req.body);
  const chunks = [];
  for await (const chunk of req) chunks.push(Buffer.from(chunk));
  const raw = Buffer.concat(chunks).toString("utf8");
  return raw ? JSON.parse(raw) : {};
}

function parseClientPayload(raw) {
  if (!raw) return {};
  try {
    const payload = JSON.parse(raw);
    return payload && typeof payload === "object" ? payload : {};
  } catch {
    return {};
  }
}

function uploadConfigFromEnv(env = process.env) {
  const accountId = clean(env.ATLAS_ACCOUNT_ID);
  const token = clean(env.BLOB_READ_WRITE_TOKEN);
  const errors = [];
  if (!token) errors.push("blob_token_missing");
  if (!accountId || !UUID_RE.test(accountId)) errors.push("atlas_account_missing");
  return { accountId, token, errors };
}

function uploadTokenConfig(pathname, clientPayload, env = process.env) {
  const config = uploadConfigFromEnv(env);
  const payload = parseClientPayload(clientPayload);
  const accountId = clean(payload.account_id);
  const errors = [...config.errors];
  if (!pathname.startsWith(BLOB_UPLOAD_PATH_PREFIX) || !pathname.toLowerCase().endsWith(".csv")) {
    errors.push("invalid_blob_pathname");
  }
  if (!accountId || accountId !== config.accountId) {
    errors.push("invalid_account_id");
  }
  if (errors.length > 0) return { ok: false, errors };
  return {
    ok: true,
    token: config.token,
    options: {
      allowedContentTypes: CSV_CONTENT_TYPES,
      maximumSizeInBytes: MAX_BLOB_CSV_BYTES,
      addRandomSuffix: true,
      tokenPayload: JSON.stringify({ account_id: config.accountId }),
    },
  };
}

async function handleBlobUpload(options) {
  const { handleUpload } = await import("@vercel/blob/client");
  return handleUpload(options);
}

export {
  BLOB_UPLOAD_PATH_PREFIX,
  CSV_CONTENT_TYPES,
  MAX_BLOB_CSV_BYTES,
  handleBlobUpload,
  parseClientPayload,
  uploadTokenConfig,
};

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    json(res, 405, { ok: false, error: "method_not_allowed" });
    return;
  }

  let body;
  try {
    body = await readJsonBody(req);
  } catch {
    json(res, 400, { ok: false, error: "invalid_upload_request" });
    return;
  }

  const envConfig = uploadConfigFromEnv();
  if (envConfig.errors.length > 0) {
    json(res, 503, { ok: false, error: "deflection_blob_upload_not_configured" });
    return;
  }

  try {
    const response = await handleBlobUpload({
      body,
      request: req,
      token: envConfig.token,
      onBeforeGenerateToken: async (pathname, clientPayload) => {
        const result = uploadTokenConfig(pathname, clientPayload);
        if (!result.ok) throw new Error("deflection_blob_upload_not_allowed");
        return result.options;
      },
      onUploadCompleted: async () => {},
    });
    json(res, 200, response);
  } catch {
    json(res, 400, { ok: false, error: "deflection_blob_upload_not_allowed" });
  }
}
