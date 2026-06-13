#!/usr/bin/env node
import { readFile, writeFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import submitHandler, {
  BLOB_UPLOAD_PATH_PREFIX,
  MAX_BLOB_CSV_BYTES,
  submitPrivateBlob,
} from "../api/content-ops/deflection/submit.js";

const SUPPORT_PLATFORMS = new Set(["zendesk", "intercom", "help_scout", "other"]);
const IMPORTER_MODES = new Set(["csv", "full_thread"]);
const REQUEST_ID_RE = /^[A-Za-z0-9][A-Za-z0-9_.:-]{5,160}$/;
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const LOCAL_HOSTS = new Set(["localhost", "0.0.0.0", "::1"]);
const DEFAULT_TIMEOUT_MS = 30000;
const DEFAULT_FIXTURE_PATHNAME = `${BLOB_UPLOAD_PATH_PREFIX}live-smoke.csv`;
const DEFAULT_FULL_THREAD_PATHNAME = `${BLOB_UPLOAD_PATH_PREFIX}zendesk-live-smoke.json`;
const DEFAULT_ZENDESK_JSON_FIXTURE = new URL(
  "../../tests/fixtures/zendesk_full_thread_seed_sample.json",
  import.meta.url,
);
const SAMPLE_CSV = [
  "ticket_id,subject,message,resolution_text,pain_category",
  "ticket-1,Export reports,How do I export attribution reports?,Open Analytics and download the CSV.,exports",
  "ticket-2,Reset invite,How can I resend an invite?,Open team settings and resend the pending invite.,accounts",
  "",
].join("\n");

function clean(value) {
  return typeof value === "string" ? value.trim() : "";
}

function parseArgs(argv = process.argv.slice(2), sourceEnv = process.env) {
  const options = {
    baseUrl: clean(sourceEnv.ATLAS_API_BASE_URL),
    token: clean(sourceEnv.ATLAS_B2B_JWT || sourceEnv.ATLAS_TOKEN),
    accountId: clean(sourceEnv.ATLAS_ACCOUNT_ID || sourceEnv.ATLAS_FAQ_SEARCH_ACCOUNT_ID),
    blobPathname: clean(sourceEnv.ATLAS_DEFLECTION_BLOB_PATHNAME),
    blobToken: clean(sourceEnv.BLOB_READ_WRITE_TOKEN),
    csvFile: clean(sourceEnv.ATLAS_DEFLECTION_SUBMIT_CSV_FILE),
    jsonFile: clean(sourceEnv.ATLAS_DEFLECTION_SUBMIT_JSON_FILE),
    importerMode: clean(sourceEnv.ATLAS_DEFLECTION_IMPORTER_MODE) || "csv",
    supportPlatform: clean(sourceEnv.ATLAS_DEFLECTION_SUPPORT_PLATFORM) || "zendesk",
    companyName: clean(sourceEnv.ATLAS_DEFLECTION_COMPANY_NAME) || "Atlas Smoke Co.",
    contactEmail: clean(sourceEnv.ATLAS_DEFLECTION_CONTACT_EMAIL) || "smoke@example.com",
    limit: clean(sourceEnv.ATLAS_DEFLECTION_LIMIT) || "1000",
    timeoutMs: Number.parseInt(clean(sourceEnv.ATLAS_PROXY_TIMEOUT_MS), 10) || DEFAULT_TIMEOUT_MS,
    outputResult: clean(sourceEnv.ATLAS_DEFLECTION_SUBMIT_RESULT_FILE),
    json: false,
    preflightOnly: false,
    routeHandler: false,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--json") {
      options.json = true;
    } else if (arg === "--preflight-only") {
      options.preflightOnly = true;
    } else if (arg === "--route-handler") {
      options.routeHandler = true;
    } else if (arg.startsWith("--")) {
      const key = arg.slice(2).replace(/-([a-z])/g, (_match, letter) => letter.toUpperCase());
      const value = argv[index + 1];
      if (value === undefined || value.startsWith("--")) {
        throw new Error(`${arg} requires a value`);
      }
      options[key] = value;
      index += 1;
    }
  }

  options.timeoutMs = Number.parseInt(String(options.timeoutMs), 10);
  options.limit = String(options.limit);
  options.importerMode = clean(options.importerMode) || "csv";
  return options;
}

function validationErrors(options) {
  const errors = [];
  const importerMode = clean(options.importerMode) || "csv";
  if (!clean(options.baseUrl) || !URL.canParse(options.baseUrl)) {
    errors.push("ATLAS_API_BASE_URL or --base-url must be an absolute HTTPS URL");
  } else {
    const url = new URL(options.baseUrl);
    const host = url.hostname.toLowerCase();
    if (url.protocol !== "https:" || LOCAL_HOSTS.has(host) || host.startsWith("127.")) {
      errors.push("ATLAS_API_BASE_URL or --base-url must point to a deployed HTTPS host");
    }
  }
  if (!clean(options.token)) errors.push("ATLAS_B2B_JWT, ATLAS_TOKEN, or --token is required");
  if (!UUID_RE.test(clean(options.accountId))) {
    errors.push("ATLAS_ACCOUNT_ID or --account-id must be an ATLAS account UUID");
  }
  if (!SUPPORT_PLATFORMS.has(clean(options.supportPlatform))) {
    errors.push("--support-platform must be one of: help_scout, intercom, other, zendesk");
  }
  if (!IMPORTER_MODES.has(importerMode)) {
    errors.push("--importer-mode must be one of: csv, full_thread");
  }
  const limit = Number.parseInt(clean(options.limit), 10);
  if (!Number.isInteger(limit) || limit < 1 || limit > 1000) {
    errors.push("--limit must be between 1 and 1000");
  }
  if (!Number.isInteger(options.timeoutMs) || options.timeoutMs <= 0) {
    errors.push("--timeout-ms must be a positive integer");
  }
  if (clean(options.blobPathname) && !clean(options.blobToken)) {
    errors.push("BLOB_READ_WRITE_TOKEN or --blob-token is required with --blob-pathname");
  }
  if (options.routeHandler && !clean(options.blobPathname)) {
    errors.push("--route-handler requires --blob-pathname");
  }
  if (options.routeHandler && clean(options.csvFile)) {
    errors.push("--route-handler cannot use --csv-file");
  }
  if (options.routeHandler && clean(options.jsonFile)) {
    errors.push("--route-handler cannot use --json-file");
  }
  if (clean(options.blobPathname) && (clean(options.csvFile) || clean(options.jsonFile))) {
    errors.push("use either --blob-pathname or a local fixture file, not both");
  }
  if (importerMode === "full_thread" && clean(options.csvFile)) {
    errors.push("--csv-file cannot be used with --importer-mode full_thread");
  }
  if (importerMode === "csv" && clean(options.jsonFile)) {
    errors.push("--json-file requires --importer-mode full_thread");
  }
  if (clean(options.blobPathname) && !options.blobPathname.startsWith(BLOB_UPLOAD_PATH_PREFIX)) {
    errors.push("--blob-pathname must stay under faq-deflection/uploads/");
  }
  if (clean(options.blobPathname)) {
    const lowerPathname = options.blobPathname.toLowerCase();
    if (importerMode === "full_thread" && !lowerPathname.endsWith(".json")) {
      errors.push("--blob-pathname must end in .json with --importer-mode full_thread");
    }
    if (importerMode === "csv" && !lowerPathname.endsWith(".csv")) {
      errors.push("--blob-pathname must end in .csv with --importer-mode csv");
    }
  }
  return errors;
}

async function localCsvBlobReader(csvFile) {
  const body = csvFile ? await readFile(csvFile) : Buffer.from(SAMPLE_CSV);
  if (body.length > MAX_BLOB_CSV_BYTES) {
    return { statusCode: 200, stream: new Blob([body]).stream(), blob: { size: body.length } };
  }
  return {
    statusCode: 200,
    stream: new Blob([body], { type: "text/csv" }).stream(),
    blob: {
      size: body.length,
      contentType: "text/csv",
    },
  };
}

async function localJsonBlobReader(jsonFile) {
  const body = await readFile(jsonFile || DEFAULT_ZENDESK_JSON_FIXTURE);
  if (body.length > MAX_BLOB_CSV_BYTES) {
    return { statusCode: 200, stream: new Blob([body]).stream(), blob: { size: body.length } };
  }
  return {
    statusCode: 200,
    stream: new Blob([body], { type: "application/json" }).stream(),
    blob: {
      size: body.length,
      contentType: "application/json",
    },
  };
}

function resultPayload(result, options, sourceMode) {
  return {
    ok: result.ok,
    source_mode: sourceMode,
    statusCode: result.statusCode,
    request_id: result.payload?.request_id,
    account_id: result.payload?.account_id,
    result_path: result.payload?.result_path,
    error: result.error,
    atlas_status: result.atlas_status,
    base_host: URL.canParse(options.baseUrl) ? new URL(options.baseUrl).host : "",
  };
}

function routePayloadFromOptions(options) {
  const payload = {
    blob_pathname: clean(options.blobPathname),
    support_platform: clean(options.importerMode) === "full_thread"
      ? "zendesk"
      : clean(options.supportPlatform),
    company_name: clean(options.companyName),
    contact_email: clean(options.contactEmail),
    limit: clean(options.limit),
  };
  if (clean(options.importerMode) === "full_thread") {
    payload.importer_mode = "full_thread";
  }
  return payload;
}

function mockRouteResponse() {
  return {
    statusCode: 200,
    headers: {},
    body: "",
    setHeader(name, value) {
      this.headers[name] = value;
    },
    end(value) {
      this.body = value || "";
    },
  };
}

async function withRouteEnv(options, fn) {
  const nextEnv = {
    ATLAS_API_BASE_URL: clean(options.baseUrl),
    ATLAS_B2B_JWT: clean(options.token),
    ATLAS_TOKEN: "",
    ATLAS_ACCOUNT_ID: clean(options.accountId),
    BLOB_READ_WRITE_TOKEN: clean(options.blobToken),
    ATLAS_PROXY_TIMEOUT_MS: String(options.timeoutMs),
  };
  const previous = {};
  for (const key of Object.keys(nextEnv)) {
    previous[key] = process.env[key];
    if (nextEnv[key]) {
      process.env[key] = nextEnv[key];
    } else {
      delete process.env[key];
    }
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

function routeResultPayload(statusCode, payload, options) {
  const ok = statusCode >= 200 && statusCode < 300 && payload?.ok === true;
  return {
    ok,
    source_mode: "portfolio_submit_route",
    statusCode,
    request_id: payload?.request_id,
    account_id: payload?.account_id,
    result_path: payload?.result_path,
    error: ok ? undefined : clean(payload?.error) || "portfolio_submit_route_failed",
    atlas_status: payload?.atlas_status,
    base_host: URL.canParse(options.baseUrl) ? new URL(options.baseUrl).host : "",
  };
}

async function runRouteHandlerSmoke(options, handlerImpl = submitHandler) {
  const res = mockRouteResponse();
  const req = {
    method: "POST",
    headers: {
      "content-type": "application/json",
    },
    body: JSON.stringify(routePayloadFromOptions(options)),
  };
  await withRouteEnv(options, async () => {
    await handlerImpl(req, res);
  });
  let payload = {};
  try {
    payload = res.body ? JSON.parse(res.body) : {};
  } catch {
    return {
      ok: false,
      source_mode: "portfolio_submit_route",
      statusCode: res.statusCode,
      error: "portfolio_submit_route_invalid_json",
      base_host: URL.canParse(options.baseUrl) ? new URL(options.baseUrl).host : "",
    };
  }
  const projected = routeResultPayload(res.statusCode, payload, options);
  if (projected.ok && !REQUEST_ID_RE.test(clean(projected.request_id))) {
    return { ...projected, ok: false, error: "invalid_request_id" };
  }
  return projected;
}

async function runSubmitSmoke(options) {
  const errors = validationErrors(options);
  if (errors.length > 0) {
    return { ok: false, status: "preflight_failed", errors };
  }

  const isFullThread = clean(options.importerMode) === "full_thread";
  const sourceMode = options.routeHandler
    ? "portfolio_submit_route"
    : clean(options.blobPathname)
      ? "private_blob"
      : isFullThread
        ? "local_zendesk_full_thread_fixture"
        : "local_csv_fixture";
  if (options.preflightOnly) {
    return { ok: true, status: "preflight_ok", source_mode: sourceMode };
  }
  if (options.routeHandler) {
    return runRouteHandlerSmoke(options);
  }

  const payload = {
    blob_pathname: clean(options.blobPathname) ||
      (isFullThread ? DEFAULT_FULL_THREAD_PATHNAME : DEFAULT_FIXTURE_PATHNAME),
    support_platform: isFullThread ? "zendesk" : clean(options.supportPlatform),
    company_name: clean(options.companyName),
    contact_email: clean(options.contactEmail),
    limit: clean(options.limit),
  };
  if (isFullThread) {
    payload.importer_mode = "full_thread";
  }
  const usesLocalFixture =
    sourceMode === "local_csv_fixture" || sourceMode === "local_zendesk_full_thread_fixture";
  const result = await submitPrivateBlob({
    config: {
      baseUrl: clean(options.baseUrl),
      token: clean(options.token),
      accountId: clean(options.accountId),
      timeoutMs: options.timeoutMs,
    },
    blobToken: usesLocalFixture ? "local-fixture" : clean(options.blobToken),
    payload,
    getBlobImpl: usesLocalFixture
      ? async () => (
          isFullThread
            ? localJsonBlobReader(clean(options.jsonFile))
            : localCsvBlobReader(clean(options.csvFile))
        )
      : undefined,
    deleteBlobImpl: usesLocalFixture ? async () => {} : undefined,
  });
  const projected = resultPayload(result, options, sourceMode);
  if (result.ok && !REQUEST_ID_RE.test(clean(projected.request_id))) {
    return { ...projected, ok: false, error: "invalid_request_id" };
  }
  return projected;
}

function printResult(payload, asJson) {
  if (asJson) {
    console.log(JSON.stringify(payload, null, 2));
    return;
  }
  if (payload.ok) {
    console.log(`faq deflection submit smoke passed (${payload.source_mode})`);
    if (payload.request_id) console.log(`request_id=${payload.request_id}`);
    if (payload.result_path) console.log(`result_path=${payload.result_path}`);
    return;
  }
  console.error(`faq deflection submit smoke failed: ${payload.error || payload.status}`);
  if (Array.isArray(payload.errors)) {
    for (const error of payload.errors) console.error(`- ${error}`);
  }
}

async function main() {
  const options = parseArgs();
  const result = await runSubmitSmoke(options);
  if (clean(options.outputResult)) {
    await writeFile(options.outputResult, `${JSON.stringify(result, null, 2)}\n`);
  }
  printResult(result, options.json);
  return result.ok ? 0 : 1;
}

if (import.meta.url === pathToFileURL(process.argv[1] || "").href) {
  main()
    .then((code) => {
      process.exitCode = code;
    })
    .catch((error) => {
      console.error(`faq deflection submit smoke failed: ${error.message}`);
      process.exitCode = 1;
    });
}

export { parseArgs, runRouteHandlerSmoke, runSubmitSmoke, validationErrors };
