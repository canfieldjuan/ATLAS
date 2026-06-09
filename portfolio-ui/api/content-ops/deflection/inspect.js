import { atlasUrl, clean, configFromEnv } from "./atlas-report.js";

const INSPECT_PATH = "/api/v1/content-ops/ingestion/files/inspect";
const MAX_INSPECT_MULTIPART_BYTES = 51 * 1024 * 1024;
const MAX_SAMPLE_COUNT = 3;
const MAX_SAMPLE_FIELD_CHARS = 240;
const EMAIL_RE = /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/gi;

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

function inspectConfig(env = process.env) {
  const config = configFromEnv(env);
  const errors = [];
  if (!config.baseUrl) errors.push("atlas_base_url_missing");
  if (!config.token) errors.push("atlas_token_missing");
  return { config, errors };
}

async function readRequestBody(req, maxBytes = MAX_INSPECT_MULTIPART_BYTES) {
  if (Buffer.isBuffer(req.body)) {
    if (req.body.length > maxBytes) return { ok: false, statusCode: 413 };
    return { ok: true, body: req.body };
  }
  if (typeof req.body === "string") {
    const body = Buffer.from(req.body);
    if (body.length > maxBytes) return { ok: false, statusCode: 413 };
    return { ok: true, body };
  }

  const chunks = [];
  let total = 0;
  for await (const chunk of req) {
    const buffer = Buffer.from(chunk);
    total += buffer.length;
    if (total > maxBytes) return { ok: false, statusCode: 413 };
    chunks.push(buffer);
  }
  return { ok: true, body: Buffer.concat(chunks) };
}

function finiteNumber(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function countMap(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const projected = {};
  for (const [key, rawValue] of Object.entries(value)) {
    const count = finiteNumber(rawValue);
    if (count === null) return null;
    projected[clean(key) || "unknown"] = count;
  }
  return projected;
}

function sanitizeSampleValue(value) {
  if (typeof value === "string") {
    return value
      .replace(EMAIL_RE, "[redacted-email]")
      .slice(0, MAX_SAMPLE_FIELD_CHARS);
  }
  if (typeof value === "number" || typeof value === "boolean" || value === null) {
    return value;
  }
  return "";
}

function sanitizeSample(row) {
  if (!row || typeof row !== "object" || Array.isArray(row)) return null;
  const projected = {};
  for (const [key, value] of Object.entries(row)) {
    const cleanKey = clean(key);
    if (!cleanKey) continue;
    projected[cleanKey] = sanitizeSampleValue(value);
  }
  return projected;
}

function projectInspectPayload(payload) {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) return null;
  if (typeof payload.ok !== "boolean") return null;
  const opportunityCount = finiteNumber(payload.opportunity_count);
  const warningCount = finiteNumber(payload.warning_count);
  const warningCounts = countMap(payload.warning_counts);
  const missingFieldCounts = countMap(payload.missing_field_counts);
  const sourceTypeCounts = countMap(payload.source_type_counts);
  if (
    opportunityCount === null ||
    warningCount === null ||
    warningCounts === null ||
    missingFieldCounts === null ||
    sourceTypeCounts === null ||
    !Array.isArray(payload.samples)
  ) {
    return null;
  }
  const samples = payload.samples
    .slice(0, MAX_SAMPLE_COUNT)
    .map(sanitizeSample)
    .filter(Boolean);
  return {
    ok: payload.ok,
    ingestion_path: clean(payload.ingestion_path),
    mode: clean(payload.mode),
    opportunity_count: opportunityCount,
    warning_count: warningCount,
    warning_counts: warningCounts,
    missing_field_counts: missingFieldCounts,
    source_type_counts: sourceTypeCounts,
    samples,
  };
}

async function forwardInspect({ config, contentType, body, fetchImpl = fetch }) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), config.timeoutMs);
  const headers = {
    "Accept": "application/json",
    "Authorization": `Bearer ${config.token}`,
  };
  if (contentType) headers["Content-Type"] = contentType;
  try {
    const response = await fetchImpl(atlasUrl(config.baseUrl, INSPECT_PATH), {
      method: "POST",
      headers,
      body,
      signal: controller.signal,
    });
    const text = await response.text();
    let payload = null;
    if (text) {
      try {
        payload = JSON.parse(text);
      } catch {
        return { ok: false, statusCode: 502, error: "atlas_inspect_invalid_json" };
      }
    }
    if (!response.ok) {
      return {
        ok: false,
        statusCode: response.status >= 400 && response.status < 500 ? response.status : 502,
        error: "atlas_inspect_failed",
        atlas_status: response.status,
      };
    }
    const projected = projectInspectPayload(payload);
    if (!projected) {
      return { ok: false, statusCode: 502, error: "atlas_inspect_contract_violation" };
    }
    return { ok: true, statusCode: 200, payload: projected };
  } catch (error) {
    return {
      ok: false,
      statusCode: error?.name === "AbortError" ? 504 : 502,
      error: "atlas_inspect_unreachable",
    };
  } finally {
    clearTimeout(timeout);
  }
}

export {
  INSPECT_PATH,
  MAX_INSPECT_MULTIPART_BYTES,
  forwardInspect,
  inspectConfig,
  projectInspectPayload,
  readRequestBody,
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
  const { config: atlasConfig, errors } = inspectConfig();
  if (errors.length > 0) {
    json(res, 503, { ok: false, error: "atlas_inspect_not_configured" });
    return;
  }

  const body = await readRequestBody(req);
  if (!body.ok) {
    json(res, body.statusCode, { ok: false, error: "deflection_inspect_csv_too_large" });
    return;
  }

  const result = await forwardInspect({
    config: atlasConfig,
    contentType,
    body: body.body,
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
}
