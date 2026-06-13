import { atlasUrl, clean, configFromEnv } from "./atlas-report.js";
import { resultPath } from "./checkout.js";
import { emitDeflectionServerEvent } from "./events.js";
import { forwardInspect } from "./inspect.js";

const REQUEST_ID_RE = /^[A-Za-z0-9][A-Za-z0-9_.:-]{5,160}$/;
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const SUBMIT_PATH = "/api/v1/content-ops/deflection-reports/submit";
const MAX_BLOB_CSV_BYTES = 50 * 1024 * 1024;
const BLOB_UPLOAD_PATH_PREFIX = "faq-deflection/uploads/";
const CSV_IMPORTER_MODE = "csv";
const FULL_THREAD_IMPORTER_MODE = "full_thread";

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

function normalizeImporterMode(value) {
  const mode = clean(value);
  if (!mode || mode === CSV_IMPORTER_MODE) return CSV_IMPORTER_MODE;
  if (mode === FULL_THREAD_IMPORTER_MODE) return FULL_THREAD_IMPORTER_MODE;
  return "";
}

function validBlobPathname(value, importerMode = CSV_IMPORTER_MODE) {
  const pathname = clean(value);
  const mode = normalizeImporterMode(importerMode);
  if (!mode) return "";
  if (!pathname.startsWith(BLOB_UPLOAD_PATH_PREFIX)) return "";
  const lowerPathname = pathname.toLowerCase();
  if (mode === FULL_THREAD_IMPORTER_MODE && !lowerPathname.endsWith(".json")) return "";
  if (mode === CSV_IMPORTER_MODE && !lowerPathname.endsWith(".csv")) return "";
  if (pathname.includes("..") || pathname.includes("\\") || pathname.startsWith("/")) return "";
  return pathname;
}

function fileNameFromPathname(pathname, importerMode = CSV_IMPORTER_MODE) {
  const mode = normalizeImporterMode(importerMode) || CSV_IMPORTER_MODE;
  const fallback = mode === FULL_THREAD_IMPORTER_MODE ? "zendesk-thread.json" : "tickets.csv";
  const extension = mode === FULL_THREAD_IMPORTER_MODE ? ".json" : ".csv";
  const name = pathname.split("/").filter(Boolean).pop() || fallback;
  return name.toLowerCase().endsWith(extension) ? name : fallback;
}

async function getPrivateBlob(pathname, options) {
  const { get } = await import("@vercel/blob");
  return get(pathname, options);
}

async function deletePrivateBlob(pathname, options) {
  const { del } = await import("@vercel/blob");
  return del(pathname, options);
}

async function readPrivateSubmitBlob({
  pathname,
  token,
  importerMode = CSV_IMPORTER_MODE,
  getBlobImpl = getPrivateBlob,
  maxBytes = MAX_BLOB_CSV_BYTES,
}) {
  const mode = normalizeImporterMode(importerMode);
  if (!mode) return { ok: false, statusCode: 400, error: "invalid_importer_mode" };
  const safePathname = validBlobPathname(pathname, mode);
  if (!safePathname) return { ok: false, statusCode: 400, error: "invalid_blob_reference" };
  const tooLargeError =
    mode === FULL_THREAD_IMPORTER_MODE
      ? "deflection_submit_full_thread_too_large"
      : "deflection_submit_csv_too_large";
  const fallbackContentType = mode === FULL_THREAD_IMPORTER_MODE ? "application/json" : "text/csv";
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
      return { ok: false, statusCode: 413, error: tooLargeError };
    }
    const body = await streamToBuffer(result.stream, maxBytes);
    if (!body.ok) {
      return { ok: false, statusCode: 413, error: tooLargeError };
    }
    return {
      ok: true,
      pathname: safePathname,
      body: body.body,
      contentType: clean(result.blob?.contentType) || fallbackContentType,
      fileName: fileNameFromPathname(safePathname, mode),
    };
  } catch {
    return { ok: false, statusCode: 502, error: "private_blob_unavailable" };
  }
}

async function readPrivateCsvBlob(options) {
  return readPrivateSubmitBlob({ ...options, importerMode: CSV_IMPORTER_MODE });
}

async function readPrivateJsonBlob(options) {
  return readPrivateSubmitBlob({ ...options, importerMode: FULL_THREAD_IMPORTER_MODE });
}

async function cleanupPrivateSubmitBlob({
  pathname,
  token,
  importerMode = CSV_IMPORTER_MODE,
  deleteBlobImpl = deletePrivateBlob,
  eventLogger = console.warn,
}) {
  const mode = normalizeImporterMode(importerMode);
  if (!mode) return { ok: false, skipped: true, error: "invalid_importer_mode" };
  const safePathname = validBlobPathname(pathname, mode);
  if (!safePathname) return { ok: false, skipped: true, error: "invalid_blob_reference" };
  try {
    await deleteBlobImpl(safePathname, { token });
    return { ok: true };
  } catch (error) {
    emitDeflectionServerEvent("faq_deflection_private_blob_cleanup_failed", {
      pathname: safePathname,
      error: "delete_failed",
    }, eventLogger);
    return { ok: false, error: "private_blob_cleanup_failed" };
  }
}

async function cleanupPrivateCsvBlob(options) {
  return cleanupPrivateSubmitBlob({ ...options, importerMode: CSV_IMPORTER_MODE });
}

async function cleanupPrivateJsonBlob(options) {
  return cleanupPrivateSubmitBlob({ ...options, importerMode: FULL_THREAD_IMPORTER_MODE });
}

function inspectFormFromBlob(blob) {
  const form = new FormData();
  form.set("file", new Blob([blob.body], { type: blob.contentType }), blob.fileName);
  form.set("source_rows", "true");
  form.set("source", "ticket-csv-upload");
  form.set("target_mode", "faq_deflection_report");
  form.set("file_format", "csv");
  form.set("sample_limit", "3");
  form.set("include_source_material", "false");
  return form;
}

async function inspectPrivateCsvBlob({ config, blob, fetchImpl = fetch }) {
  const result = await forwardInspect({
    config,
    contentType: "",
    body: inspectFormFromBlob(blob),
    fetchImpl,
  });
  if (!result.ok) return result;
  if (!result.payload.ok) {
    return { ok: false, statusCode: 400, error: "deflection_inspect_not_ready" };
  }
  return { ok: true, payload: result.payload };
}

async function submitPrivateBlob({
  config,
  blobToken,
  payload,
  fetchImpl = fetch,
  getBlobImpl = getPrivateBlob,
  deleteBlobImpl = deletePrivateBlob,
  eventLogger = console.warn,
}) {
  const importerMode = normalizeImporterMode(payload?.importer_mode);
  if (!importerMode) return { ok: false, statusCode: 400, error: "invalid_importer_mode" };
  const blob = await readPrivateSubmitBlob({
    pathname: payload?.blob_pathname,
    token: blobToken,
    importerMode,
    getBlobImpl,
  });
  if (!blob.ok) return blob;

  if (importerMode === CSV_IMPORTER_MODE) {
    const inspectResult = await inspectPrivateCsvBlob({ config, blob, fetchImpl });
    if (!inspectResult.ok) {
      await cleanupPrivateSubmitBlob({
        pathname: blob.pathname,
        token: blobToken,
        importerMode,
        deleteBlobImpl,
        eventLogger,
      });
      return inspectResult;
    }
  }

  const form = new FormData();
  if (importerMode === FULL_THREAD_IMPORTER_MODE) {
    form.set("json_file", new Blob([blob.body], { type: blob.contentType }), blob.fileName);
    form.set("importer_mode", FULL_THREAD_IMPORTER_MODE);
  } else {
    form.set("csv_file", new Blob([blob.body], { type: blob.contentType }), blob.fileName);
  }
  form.set(
    "support_platform",
    importerMode === FULL_THREAD_IMPORTER_MODE ? "zendesk" : clean(payload?.support_platform),
  );
  form.set("company_name", clean(payload?.company_name));
  form.set("contact_email", clean(payload?.contact_email));
  form.set("limit", clean(payload?.limit) || "1000");

  const result = await forwardSubmit({ config, contentType: "", body: form, fetchImpl });
  await cleanupPrivateSubmitBlob({
    pathname: blob.pathname,
    token: blobToken,
    importerMode,
    deleteBlobImpl,
    eventLogger,
  });
  return result;
}

export {
  BLOB_UPLOAD_PATH_PREFIX,
  CSV_IMPORTER_MODE,
  FULL_THREAD_IMPORTER_MODE,
  MAX_BLOB_CSV_BYTES,
  SUBMIT_PATH,
  cleanupPrivateCsvBlob,
  cleanupPrivateJsonBlob,
  forwardSubmit,
  inspectPrivateCsvBlob,
  normalizeImporterMode,
  projectSubmitPayload,
  readPrivateCsvBlob,
  readPrivateJsonBlob,
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
  if (contentType.includes("multipart/form-data")) {
    json(res, 410, { ok: false, error: "direct_multipart_deprecated" });
    return;
  }
  if (!isPrivateBlobSubmit) {
    json(res, 415, { ok: false, error: "private_blob_json_required" });
    return;
  }

  const { config: submitConfig, blobToken, errors } = privateBlobSubmitConfig();
  if (errors.length > 0) {
    json(res, 503, { ok: false, error: "atlas_submit_not_configured" });
    return;
  }

  const requestedAccountId = header(req, "x-atlas-account-id");
  if (
    requestedAccountId &&
    (!UUID_RE.test(requestedAccountId) || requestedAccountId !== submitConfig.accountId)
  ) {
    json(res, 400, { ok: false, error: "invalid_account_id" });
    return;
  }
  if (!submitConfig.accountId || !UUID_RE.test(submitConfig.accountId)) {
    json(res, 400, { ok: false, error: "invalid_account_id" });
    return;
  }

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
}
