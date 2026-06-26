import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import submitHandler, {
  BLOB_UPLOAD_PATH_PREFIX,
  FULL_THREAD_IMPORTER_MODE,
  MAX_BLOB_CSV_BYTES,
  SUBMIT_PATH,
  cleanupPrivateCsvBlob,
  forwardSubmit,
  readPrivateCsvBlob,
  submitPrivateBlob,
} from "../api/content-ops/deflection/submit.js";
import zendeskExportSubmitHandler, {
  ZENDESK_EXPORT_PATH,
  submitZendeskCredentialFlow,
} from "../api/content-ops/deflection/zendesk-export-submit.js";
import inspectHandler, {
  INSPECT_PATH,
  MAX_INSPECT_MULTIPART_BYTES,
} from "../api/content-ops/deflection/inspect.js";
import {
  BLOB_UPLOAD_CONTENT_TYPES,
  CSV_CONTENT_TYPES,
  JSON_CONTENT_TYPES,
  MAX_BLOB_CSV_BYTES as MAX_UPLOAD_CSV_BYTES,
  uploadTokenConfig,
} from "../api/content-ops/deflection/upload.js";
import {
  emitDeflectionServerEvent,
  sanitizeDeflectionEventFields,
} from "../api/content-ops/deflection/events.js";
import {
  parseArgs as parseSubmitSmokeArgs,
  runRouteHandlerSmoke,
  runSubmitSmoke,
  runZendeskApiRouteHandlerSmoke,
  validationErrors as submitSmokeValidationErrors,
} from "./faq-deflection-submit-live-smoke.mjs";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const appSource = await readFile(resolve(root, "src/App.tsx"), "utf8");
const servicesSource = await readFile(resolve(root, "src/pages/Services.tsx"), "utf8");
const uploadSource = await readFile(resolve(root, "src/pages/FaqDeflectionUpload.tsx"), "utf8");
const submitSource = await readFile(resolve(root, "api/content-ops/deflection/submit.js"), "utf8");
const zendeskExportSubmitSource = await readFile(
  resolve(root, "api/content-ops/deflection/zendesk-export-submit.js"),
  "utf8",
);
const inspectSource = await readFile(resolve(root, "api/content-ops/deflection/inspect.js"), "utf8");
const submitSmokeSource = await readFile(
  resolve(root, "scripts/faq-deflection-submit-live-smoke.mjs"),
  "utf8",
);
const blobUploadRouteSource = await readFile(
  resolve(root, "api/content-ops/deflection/upload.js"),
  "utf8",
);
const packageJson = JSON.parse(await readFile(resolve(root, "package.json"), "utf8"));

const ACCOUNT_ID = "2b2b950d-f64b-4852-bc30-f92a34cdf169";
const ENV = {
  ATLAS_API_BASE_URL: "https://atlas.example.com",
  ATLAS_B2B_JWT: "secret-service-token",
  ATLAS_ACCOUNT_ID: ACCOUNT_ID,
  ATLAS_DEFLECTION_ZENDESK_EXPORT_ACCESS_TOKEN: "operator-export-access-token",
  BLOB_READ_WRITE_TOKEN: "vercel_blob_rw_token",
};
const READY_INSPECT_PAYLOAD = {
  ok: true,
  ingestion_path: "file_upload",
  mode: "source_rows",
  opportunity_count: 1,
  warning_count: 0,
  warning_counts: {},
  missing_field_counts: {},
  source_type_counts: { support_ticket: 1 },
  samples: [],
};
const NOT_READY_INSPECT_PAYLOAD = {
  ok: false,
  ingestion_path: "file_upload",
  mode: "source_rows",
  opportunity_count: 0,
  warning_count: 1,
  warning_counts: { no_rows: 1 },
  missing_field_counts: { target_id: 1 },
  source_type_counts: {},
  samples: [],
};
const MULTIPART_BODY = Buffer.from(
  [
    "--atlas",
    'Content-Disposition: form-data; name="support_platform"',
    "",
    "zendesk",
    "--atlas",
    'Content-Disposition: form-data; name="company_name"',
    "",
    "Acme Co.",
    "--atlas",
    'Content-Disposition: form-data; name="contact_email"',
    "",
    "lead@acme.example",
    "--atlas",
    'Content-Disposition: form-data; name="csv_file"; filename="tickets.csv"',
    "Content-Type: text/csv",
    "",
    "ticket_id,message",
    "ticket-1,How do I export reports?",
    "--atlas--",
    "",
  ].join("\r\n"),
);
const INSPECT_MULTIPART_BODY = Buffer.from(
  [
    "--atlas",
    'Content-Disposition: form-data; name="source_rows"',
    "",
    "true",
    "--atlas",
    'Content-Disposition: form-data; name="source"',
    "",
    "ticket-csv-upload",
    "--atlas",
    'Content-Disposition: form-data; name="target_mode"',
    "",
    "faq_deflection_report",
    "--atlas",
    'Content-Disposition: form-data; name="file_format"',
    "",
    "csv",
    "--atlas",
    'Content-Disposition: form-data; name="file"; filename="tickets.csv"',
    "Content-Type: text/csv",
    "",
    "ticket_id,message",
    "ticket-1,How do I export reports?",
    "--atlas--",
    "",
  ].join("\r\n"),
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

function request({ method = "POST", headers = {}, body = MULTIPART_BODY } = {}) {
  return {
    method,
    headers: {
      "content-type": "multipart/form-data; boundary=atlas",
      "content-length": String(body.length),
      "x-atlas-account-id": ACCOUNT_ID,
      ...headers,
    },
    body,
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

await test("upload shell route is wired into the portfolio app and services page", () => {
  assert.match(appSource, /FaqDeflectionUpload/);
  assert.match(appSource, /\/services\/faq-deflection/);
  assert.match(servicesSource, /\/services\/faq-deflection/);
  assert.match(servicesSource, /FAQ Deflection Upload/);
});

await test("upload shell exposes live submit markers and avoids browser credentials", () => {
  for (const marker of [
    "data-atlas-deflection-upload",
    "data-atlas-deflection-csv-file",
    "data-atlas-deflection-upload-file",
    "data-atlas-deflection-upload-mode",
    "data-atlas-deflection-upload-mode-option",
    "data-atlas-deflection-company",
    "data-atlas-deflection-contact-email",
    "data-atlas-deflection-support-platform",
    "data-atlas-deflection-submit",
    "data-atlas-deflection-inspect-endpoint",
    "data-atlas-deflection-upload-endpoint",
    "data-atlas-deflection-zendesk-export-submit-endpoint",
    "data-atlas-deflection-upload-progress",
    "data-atlas-deflection-zendesk-api-source",
    "data-atlas-deflection-zendesk-export-access-token",
    "data-atlas-deflection-zendesk-export-progress",
    "data-atlas-deflection-inspect-preview",
    "data-atlas-deflection-export-guidance",
    "data-atlas-deflection-retry",
  ]) {
    assert.match(uploadSource, new RegExp(marker));
  }
  assert.match(uploadSource, /MAX_CSV_BYTES = 50 \* 1024 \* 1024/);
  assert.match(uploadSource, /FULL_THREAD_IMPORTER_MODE = "full_thread"/);
  assert.match(uploadSource, /@vercel\/blob\/client/);
  assert.doesNotMatch(blobUploadRouteSource, /^import .*@vercel\/blob\/client/m);
  assert.match(blobUploadRouteSource, /import\("@vercel\/blob\/client"\)/);
  assert.match(uploadSource, /access: "private"/);
  assert.match(uploadSource, /accept=\{isFullThreadUpload \? "\.json,application\/json" : "\.csv,text\/csv"\}/);
  assert.match(uploadSource, /contentType: isFullThreadUpload \? "application\/json" : "text\/csv"/);
  assert.match(uploadSource, /onUploadProgress/);
  assert.match(uploadSource, /boundedProgress\(event\.percentage\)/);
  assert.match(uploadSource, /role="progressbar"/);
  assert.match(uploadSource, /aria-valuenow=\{submit\.percentage\}/);
  assert.match(uploadSource, /FAQ Deflection Intake/);
  assert.match(uploadSource, /Start a deterministic FAQ gap audit/);
  assert.match(uploadSource, /repeatable clustering/);
  assert.match(uploadSource, /not\s+chatbot interpretation/);
  assert.match(uploadSource, /100% Deterministic Engine/);
  assert.match(uploadSource, /This tool does not use LLMs or generative AI[\s\S]*deterministic clustering/);
  assert.match(uploadSource, /full ticket threads[\s\S]*customer questions[\s\S]*agent replies/);
  assert.match(uploadSource, /resolution text[\s\S]*resolved ticket notes/);
  assert.match(uploadSource, /Question-only exports[\s\S]*clustering[\s\S]*gap\s+list/);
  assert.match(uploadSource, /publishable answers require uploaded resolution\s+evidence/);
  assert.match(uploadSource, /value: "zendesk_full_thread"/);
  assert.match(uploadSource, /Zendesk JSON/);
  assert.match(uploadSource, /value: "zendesk_api"/);
  assert.match(uploadSource, /Zendesk API/);
  assert.match(uploadSource, /ZENDESK_EXPORT_SUBMIT_ENDPOINT = "\/api\/content-ops\/deflection\/zendesk-export-submit"/);
  assert.match(uploadSource, /fetch\(ZENDESK_EXPORT_SUBMIT_ENDPOINT/);
  assert.match(uploadSource, /Authorization": `Bearer \$\{zendeskExportAccessToken\.trim\(\)\}`/);
  assert.match(uploadSource, /setSubmit\(\{ status: "exporting" \}\)/);
  assert.match(uploadSource, /The browser receives only the locked report path/);
  assert.match(uploadSource, /ticket artifacts stay server-side/);
  assert.match(zendeskExportSubmitSource, /ZENDESK_EXPORT_PATH = "\/api\/v1\/content-ops\/zendesk-export\/full-thread"/);
  assert.match(zendeskExportSubmitSource, /ATLAS_DEFLECTION_ZENDESK_EXPORT_ACCESS_TOKEN/);
  assert.match(zendeskExportSubmitSource, /zendesk_export_auth_required/);
  assert.match(zendeskExportSubmitSource, /bodyParser: false/);
  assert.match(zendeskExportSubmitSource, /submitPrivateBlob/);
  assert.match(zendeskExportSubmitSource, /access: "private"/);
  assert.match(zendeskExportSubmitSource, /addRandomSuffix: true/);
  assert.match(uploadSource, /public requester comments[\s\S]*public agent replies/);
  assert.match(uploadSource, /Private notes are dropped during import/);
  assert.match(uploadSource, /ATLAS validates the thread shape during submit/);
  assert.doesNotMatch(uploadSource, /exact mathematical clustering/);
  assert.match(uploadSource, /Workspace routing is handled server-side/);
  assert.match(uploadSource, /Your browser never[\s\S]*receives ATLAS service tokens/);
  assert.match(uploadSource, /JSON import/);
  assert.match(uploadSource, /CSV inspection/);
  assert.match(uploadSource, /private storage both run server-side/);
  assert.match(uploadSource, /ATLAS checks the CSV first[\s\S]*stores it privately after validation/);
  assert.match(uploadSource, /private Zendesk JSON server-side[\s\S]*imports public thread evidence/);
  assert.match(uploadSource, /const INSPECT_ENDPOINT = "\/api\/content-ops\/deflection\/inspect"/);
  assert.match(uploadSource, /fetch\(INSPECT_ENDPOINT/);
  assert.match(uploadSource, /if \(!isFullThreadUpload\)/);
  assert.match(uploadSource, /source_rows", "true"/);
  assert.match(uploadSource, /target_mode", "faq_deflection_report"/);
  assert.match(uploadSource, /Checking CSV/);
  assert.match(uploadSource, /CSV pre-flight passed/);
  assert.match(uploadSource, /CSV needs a fuller export/);
  assert.ok(
    uploadSource.indexOf("await inspectDeflectionCsv") < uploadSource.indexOf("uploadBlob("),
    "inspect must run before private Blob upload",
  );
  assert.doesNotMatch(uploadSource, />\s*Support-ticket CSV upload\s*</);
  assert.doesNotMatch(uploadSource, /Bound server-side to the configured report workspace/);
  assert.doesNotMatch(uploadSource, /CSV bytes are first stored in private Vercel Blob/);
  assert.match(uploadSource, /Retry upload/);
  assert.match(uploadSource, /starts a new\s+private upload/);
  assert.match(uploadSource, /blob_pathname: blob\.pathname/);
  assert.match(uploadSource, /support_platform: isFullThreadUpload \? "zendesk" : supportPlatform/);
  assert.match(uploadSource, /importer_mode: FULL_THREAD_IMPORTER_MODE/);
  assert.match(uploadSource, /private_blob_persistence/);
  assert.match(uploadSource, /value: "help_scout"/);
  assert.match(uploadSource, /value: "other", label: "Freshdesk \/ other"/);
  assert.doesNotMatch(uploadSource, /value: "freshdesk"/);
  assert.doesNotMatch(uploadSource, /value: "help-scout"/);
  assert.match(uploadSource, /new FormData\(\)/);
  assert.match(uploadSource, /JSON\.stringify/);
  assert.doesNotMatch(uploadSource, /X-Atlas-Account-Id|account_id/);
  assert.doesNotMatch(uploadSource, /ATLAS_B2B_JWT|ATLAS_API_BASE_URL|ATLAS_TOKEN/);
  assert.doesNotMatch(uploadSource, /BLOB_READ_WRITE_TOKEN|ATLAS_DEFLECTION_ZENDESK_EXPORT_ACCESS_TOKEN/);
  assert.doesNotMatch(uploadSource, /\/paid\b/);
});

await test("portfolio inspect endpoint forwards multipart to ATLAS and redacts preview samples", async () => {
  const previousFetch = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push({ url, options });
    return {
      ok: true,
      status: 200,
      async text() {
        return JSON.stringify({
          ok: true,
          ingestion_path: "file_upload",
          mode: "source_rows",
          opportunity_count: 2,
          warning_count: 0,
          warning_counts: {},
          missing_field_counts: {},
          source_type_counts: { support_ticket: 2 },
          samples: [{
            target_id: "ticket-1",
            text: "Customer lead@example.com asks how to export reports.",
          }],
          source_material: [{
            contact_email: "lead@example.com",
          }],
        });
      },
    };
  };
  const res = mockResponse();
  try {
    await withEnv(ENV, async () => {
      await inspectHandler(
        request({ body: INSPECT_MULTIPART_BODY }),
        res,
      );
    });
  } finally {
    globalThis.fetch = previousFetch;
  }

  assert.equal(res.statusCode, 200);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, `${ENV.ATLAS_API_BASE_URL}${INSPECT_PATH}`);
  assert.equal(calls[0].options.method, "POST");
  assert.equal(calls[0].options.headers.Authorization, `Bearer ${ENV.ATLAS_B2B_JWT}`);
  assert.equal(calls[0].options.headers["Content-Type"], "multipart/form-data; boundary=atlas");
  assert.deepEqual(calls[0].options.body, INSPECT_MULTIPART_BODY);
  const payload = JSON.parse(res.body);
  assert.equal(payload.ok, true);
  assert.equal(payload.opportunity_count, 2);
  assert.deepEqual(payload.source_type_counts, { support_ticket: 2 });
  assert.equal(payload.samples[0].text.includes("[redacted-email]"), true);
  assert.equal("source_material" in payload, false);
  assert.equal(res.body.includes(ENV.ATLAS_B2B_JWT), false);
  assert.equal(res.body.includes("lead@example.com"), false);
});

await test("portfolio inspect endpoint returns not-ready diagnostics without failing transport", async () => {
  const previousFetch = globalThis.fetch;
  globalThis.fetch = async () => ({
    ok: true,
    status: 200,
    async text() {
      return JSON.stringify({
        ok: false,
        ingestion_path: "file_upload",
        mode: "source_rows",
        opportunity_count: 0,
        warning_count: 1,
        warning_counts: { no_rows: 1 },
        missing_field_counts: { target_id: 1 },
        source_type_counts: {},
        samples: [],
      });
    },
  });
  const res = mockResponse();
  try {
    await withEnv(ENV, async () => {
      await inspectHandler(request({ body: INSPECT_MULTIPART_BODY }), res);
    });
  } finally {
    globalThis.fetch = previousFetch;
  }

  assert.equal(res.statusCode, 200);
  assert.deepEqual(JSON.parse(res.body), {
    ok: false,
    ingestion_path: "file_upload",
    mode: "source_rows",
    opportunity_count: 0,
    warning_count: 1,
    warning_counts: { no_rows: 1 },
    missing_field_counts: { target_id: 1 },
    source_type_counts: {},
    samples: [],
  });
});

await test("portfolio inspect endpoint fails closed on malformed ATLAS envelope", async () => {
  const previousFetch = globalThis.fetch;
  globalThis.fetch = async () => ({
    ok: true,
    status: 200,
    async text() {
      return JSON.stringify({ ok: true, samples: [] });
    },
  });
  const res = mockResponse();
  try {
    await withEnv(ENV, async () => {
      await inspectHandler(request({ body: INSPECT_MULTIPART_BODY }), res);
    });
  } finally {
    globalThis.fetch = previousFetch;
  }

  assert.equal(res.statusCode, 502);
  assert.deepEqual(JSON.parse(res.body), {
    ok: false,
    error: "atlas_inspect_contract_violation",
  });
});

await test("portfolio inspect endpoint caps multipart bytes before ATLAS", async () => {
  const previousFetch = globalThis.fetch;
  let fetchCalled = false;
  globalThis.fetch = async () => {
    fetchCalled = true;
    throw new Error("must not call ATLAS for oversized inspect");
  };
  const res = mockResponse();
  try {
    await withEnv(ENV, async () => {
      await inspectHandler(
        request({ body: Buffer.alloc(MAX_INSPECT_MULTIPART_BYTES + 1) }),
        res,
      );
    });
  } finally {
    globalThis.fetch = previousFetch;
  }

  assert.equal(fetchCalled, false);
  assert.equal(res.statusCode, 413);
  assert.deepEqual(JSON.parse(res.body), {
    ok: false,
    error: "deflection_inspect_csv_too_large",
  });
});

await test("portfolio submit endpoint pins private Blob submit handling", () => {
  assert.match(submitSource, /bodyParser:\s*false/);
  assert.doesNotMatch(submitSource, /^import .*@vercel\/blob/m);
  assert.match(submitSource, /import\("@vercel\/blob"\)/);
  assert.match(submitSource, /const \{ del \} = await import\("@vercel\/blob"\)/);
  assert.match(submitSource, /direct_multipart_deprecated/);
  assert.doesNotMatch(submitSource, /readRawBody|MAX_MULTIPART_OVERHEAD_BYTES/);
  assert.match(submitSource, /FULL_THREAD_IMPORTER_MODE = "full_thread"/);
  assert.match(submitSource, /normalizeImporterMode/);
  assert.match(submitSource, /form\.set\("json_file"/);
  assert.match(submitSource, /form\.set\("importer_mode", FULL_THREAD_IMPORTER_MODE\)/);
  assert.match(submitSource, /importerMode === FULL_THREAD_IMPORTER_MODE \? "zendesk" : clean\(payload\?\.support_platform\)/);
  assert.match(submitSource, /inspectPrivateCsvBlob/);
  assert.match(submitSource, /deflection_inspect_not_ready/);
  assert.match(inspectSource, /bodyParser:\s*false/);
  assert.match(inspectSource, /projectInspectPayload/);
  assert.doesNotMatch(inspectSource, /ATLAS_B2B_JWT[^\\n]*console\\.log/);
});

await test("submit live smoke exercises the production private blob helper", async () => {
  assert.match(submitSmokeSource, /submitPrivateBlob/);
  assert.match(submitSmokeSource, /submitHandler/);
  assert.match(submitSmokeSource, /local_csv_fixture/);
  assert.match(submitSmokeSource, /local_zendesk_full_thread_fixture/);
  assert.match(submitSmokeSource, /ATLAS_DEFLECTION_IMPORTER_MODE/);
  assert.match(submitSmokeSource, /ATLAS_DEFLECTION_SUBMIT_JSON_FILE/);
  assert.match(submitSmokeSource, /zendesk_full_thread_seed_sample\.json/);
  assert.match(submitSmokeSource, /portfolio_submit_route/);
  assert.match(submitSmokeSource, /portfolio_zendesk_api_route/);
  assert.match(submitSmokeSource, /ATLAS_DEFLECTION_ZENDESK_API_SMOKE/);
  assert.match(submitSmokeSource, /ATLAS_DEFLECTION_ZENDESK_EXPORT_ACCESS_TOKEN/);
  assert.match(submitSmokeSource, /zendeskExportSubmitHandler/);
  assert.doesNotMatch(submitSmokeSource, /ATLAS_B2B_JWT[^\\n]*console\\.log/);
  assert.doesNotMatch(submitSmokeSource, /ATLAS_DEFLECTION_ZENDESK_EXPORT_ACCESS_TOKEN[^\\n]*console\\.log/);
  assert.deepEqual(
    submitSmokeValidationErrors({
      baseUrl: "http://localhost:8000",
      token: ENV.ATLAS_B2B_JWT,
      accountId: ACCOUNT_ID,
      supportPlatform: "zendesk",
      limit: "1000",
      timeoutMs: 1000,
    }),
    ["ATLAS_API_BASE_URL or --base-url must point to a deployed HTTPS host"],
  );
  assert.equal(
    submitSmokeValidationErrors({
      baseUrl: ENV.ATLAS_API_BASE_URL,
      token: ENV.ATLAS_B2B_JWT,
      accountId: ACCOUNT_ID,
      supportPlatform: "zendesk",
      limit: "1000",
      timeoutMs: 1000,
      routeHandler: true,
    }).includes("--route-handler requires --blob-pathname"),
    true,
  );
  assert.deepEqual(
    submitSmokeValidationErrors({
      baseUrl: ENV.ATLAS_API_BASE_URL,
      token: ENV.ATLAS_B2B_JWT,
      accountId: ACCOUNT_ID,
      blobToken: ENV.BLOB_READ_WRITE_TOKEN,
      supportPlatform: "zendesk",
      limit: "1000",
      timeoutMs: 1000,
      zendeskApi: true,
    }),
    [
      "ATLAS_DEFLECTION_ZENDESK_EXPORT_ACCESS_TOKEN or --zendesk-export-access-token is required with --zendesk-api",
    ],
  );
  assert.equal(
    submitSmokeValidationErrors({
      baseUrl: ENV.ATLAS_API_BASE_URL,
      token: ENV.ATLAS_B2B_JWT,
      accountId: ACCOUNT_ID,
      blobPathname: `${BLOB_UPLOAD_PATH_PREFIX}zendesk-thread.json`,
      blobToken: ENV.BLOB_READ_WRITE_TOKEN,
      zendeskExportAccessToken: ENV.ATLAS_DEFLECTION_ZENDESK_EXPORT_ACCESS_TOKEN,
      supportPlatform: "zendesk",
      limit: "1000",
      timeoutMs: 1000,
      zendeskApi: true,
    }).includes("--zendesk-api cannot use --blob-pathname"),
    true,
  );
  assert.equal(
    submitSmokeValidationErrors({
      baseUrl: ENV.ATLAS_API_BASE_URL,
      token: ENV.ATLAS_B2B_JWT,
      accountId: ACCOUNT_ID,
      blobToken: ENV.BLOB_READ_WRITE_TOKEN,
      zendeskExportAccessToken: ENV.ATLAS_DEFLECTION_ZENDESK_EXPORT_ACCESS_TOKEN,
      supportPlatform: "zendesk",
      limit: "1000",
      startTime: "-1",
      timeoutMs: 1000,
      zendeskApi: true,
    }).includes("--start-time must be a non-negative integer"),
    true,
  );
  assert.equal(
    submitSmokeValidationErrors({
      baseUrl: ENV.ATLAS_API_BASE_URL,
      token: ENV.ATLAS_B2B_JWT,
      accountId: ACCOUNT_ID,
      blobToken: ENV.BLOB_READ_WRITE_TOKEN,
      zendeskExportAccessToken: ENV.ATLAS_DEFLECTION_ZENDESK_EXPORT_ACCESS_TOKEN,
      supportPlatform: "zendesk",
      limit: "1000",
      startTime: "1710000000abc",
      timeoutMs: 1000,
      zendeskApi: true,
    }).includes("--start-time must be a non-negative integer"),
    true,
  );
  assert.equal(
    submitSmokeValidationErrors({
      baseUrl: ENV.ATLAS_API_BASE_URL,
      token: ENV.ATLAS_B2B_JWT,
      accountId: ACCOUNT_ID,
      supportPlatform: "zendesk",
      limit: "1000",
      startTime: "not-used-outside-zendesk-api",
      timeoutMs: 1000,
    }).includes("--start-time must be a non-negative integer"),
    false,
  );
  assert.equal(
    submitSmokeValidationErrors({
      baseUrl: ENV.ATLAS_API_BASE_URL,
      token: ENV.ATLAS_B2B_JWT,
      accountId: ACCOUNT_ID,
      blobPathname: `${BLOB_UPLOAD_PATH_PREFIX}zendesk-thread.json`,
      blobToken: ENV.BLOB_READ_WRITE_TOKEN,
      jsonFile: "zendesk-thread.json",
      importerMode: "full_thread",
      supportPlatform: "zendesk",
      limit: "1000",
      timeoutMs: 1000,
      routeHandler: true,
    }).includes("--route-handler cannot use --json-file"),
    true,
  );
  assert.equal(
    submitSmokeValidationErrors({
      baseUrl: ENV.ATLAS_API_BASE_URL,
      token: ENV.ATLAS_B2B_JWT,
      accountId: ACCOUNT_ID,
      csvFile: "tickets.csv",
      importerMode: "full_thread",
      supportPlatform: "zendesk",
      limit: "1000",
      timeoutMs: 1000,
    }).includes("--csv-file cannot be used with --importer-mode full_thread"),
    true,
  );
  assert.equal(
    submitSmokeValidationErrors({
      baseUrl: ENV.ATLAS_API_BASE_URL,
      token: ENV.ATLAS_B2B_JWT,
      accountId: ACCOUNT_ID,
      blobPathname: `${BLOB_UPLOAD_PATH_PREFIX}tickets.csv`,
      blobToken: ENV.BLOB_READ_WRITE_TOKEN,
      csvFile: "tickets.csv",
      importerMode: "csv",
      supportPlatform: "zendesk",
      limit: "1000",
      timeoutMs: 1000,
      routeHandler: true,
    }).includes("--route-handler cannot use --csv-file"),
    true,
  );
  assert.equal(
    submitSmokeValidationErrors({
      baseUrl: ENV.ATLAS_API_BASE_URL,
      token: ENV.ATLAS_B2B_JWT,
      accountId: ACCOUNT_ID,
      blobPathname: `${BLOB_UPLOAD_PATH_PREFIX}tickets.csv`,
      blobToken: ENV.BLOB_READ_WRITE_TOKEN,
      importerMode: "full_thread",
      supportPlatform: "zendesk",
      limit: "1000",
      timeoutMs: 1000,
    }).includes("--blob-pathname must end in .json with --importer-mode full_thread"),
    true,
  );
  assert.equal(
    submitSmokeValidationErrors({
      baseUrl: ENV.ATLAS_API_BASE_URL,
      token: ENV.ATLAS_B2B_JWT,
      accountId: ACCOUNT_ID,
      blobPathname: `${BLOB_UPLOAD_PATH_PREFIX}zendesk-thread.json`,
      blobToken: ENV.BLOB_READ_WRITE_TOKEN,
      importerMode: "csv",
      supportPlatform: "zendesk",
      limit: "1000",
      timeoutMs: 1000,
    }).includes("--blob-pathname must end in .csv with --importer-mode csv"),
    true,
  );
  assert.equal(
    submitSmokeValidationErrors({
      baseUrl: ENV.ATLAS_API_BASE_URL,
      token: ENV.ATLAS_B2B_JWT,
      accountId: ACCOUNT_ID,
      importerMode: "xml",
      supportPlatform: "zendesk",
      limit: "1000",
      timeoutMs: 1000,
    }).includes("--importer-mode must be one of: csv, full_thread"),
    true,
  );

  const options = parseSubmitSmokeArgs(["--preflight-only"], {
    ATLAS_API_BASE_URL: ENV.ATLAS_API_BASE_URL,
    ATLAS_B2B_JWT: ENV.ATLAS_B2B_JWT,
    ATLAS_ACCOUNT_ID: ACCOUNT_ID,
  });
  assert.deepEqual(await runSubmitSmoke(options), {
    ok: true,
    status: "preflight_ok",
    source_mode: "local_csv_fixture",
  });
  const fullThreadOptions = parseSubmitSmokeArgs([
    "--preflight-only",
    "--importer-mode",
    "full_thread",
  ], {
    ATLAS_API_BASE_URL: ENV.ATLAS_API_BASE_URL,
    ATLAS_B2B_JWT: ENV.ATLAS_B2B_JWT,
    ATLAS_ACCOUNT_ID: ACCOUNT_ID,
  });
  assert.deepEqual(await runSubmitSmoke(fullThreadOptions), {
    ok: true,
    status: "preflight_ok",
    source_mode: "local_zendesk_full_thread_fixture",
  });

  const routeOptions = parseSubmitSmokeArgs([
    "--route-handler",
    "--preflight-only",
    "--blob-pathname",
    `${BLOB_UPLOAD_PATH_PREFIX}tickets.csv`,
  ], {
    ...ENV,
  });
  assert.deepEqual(await runSubmitSmoke(routeOptions), {
    ok: true,
    status: "preflight_ok",
    source_mode: "portfolio_submit_route",
  });
  const fullThreadRouteOptions = parseSubmitSmokeArgs([
    "--route-handler",
    "--preflight-only",
    "--importer-mode",
    "full_thread",
    "--blob-pathname",
    `${BLOB_UPLOAD_PATH_PREFIX}zendesk-thread.json`,
  ], {
    ...ENV,
  });
  assert.deepEqual(await runSubmitSmoke(fullThreadRouteOptions), {
    ok: true,
    status: "preflight_ok",
    source_mode: "portfolio_submit_route",
  });

  const zendeskApiOptions = parseSubmitSmokeArgs([
    "--zendesk-api",
    "--preflight-only",
    "--start-time",
    "1710000000",
  ], {
    ...ENV,
  });
  assert.deepEqual(await runSubmitSmoke(zendeskApiOptions), {
    ok: true,
    status: "preflight_ok",
    source_mode: "portfolio_zendesk_api_route",
  });
});

await test("submit live smoke route-handler mode omits buyer account header", async () => {
  const options = parseSubmitSmokeArgs([
    "--route-handler",
    "--blob-pathname",
    `${BLOB_UPLOAD_PATH_PREFIX}tickets.csv`,
  ], {
    ...ENV,
  });
  const result = await runRouteHandlerSmoke(options, async (req, res) => {
    assert.equal(req.method, "POST");
    assert.equal(req.headers["content-type"], "application/json");
    assert.equal("x-atlas-account-id" in req.headers, false);
    assert.equal(process.env.ATLAS_API_BASE_URL, ENV.ATLAS_API_BASE_URL);
    assert.equal(process.env.ATLAS_B2B_JWT, ENV.ATLAS_B2B_JWT);
    assert.equal(process.env.ATLAS_ACCOUNT_ID, ACCOUNT_ID);
    assert.equal(process.env.BLOB_READ_WRITE_TOKEN, ENV.BLOB_READ_WRITE_TOKEN);
    assert.deepEqual(JSON.parse(req.body), {
      blob_pathname: `${BLOB_UPLOAD_PATH_PREFIX}tickets.csv`,
      support_platform: "zendesk",
      company_name: "Atlas Smoke Co.",
      contact_email: "smoke@example.com",
      limit: "1000",
    });
    res.statusCode = 200;
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.end(JSON.stringify({
      ok: true,
      request_id: "content-ops-route123",
      account_id: ACCOUNT_ID,
      result_path: "/services/faq-deflection/results/content-ops-route123",
    }));
  });
  assert.deepEqual(result, {
    ok: true,
    source_mode: "portfolio_submit_route",
    statusCode: 200,
    request_id: "content-ops-route123",
    account_id: ACCOUNT_ID,
    result_path: "/services/faq-deflection/results/content-ops-route123",
    error: undefined,
    atlas_status: undefined,
    base_host: "atlas.example.com",
  });
});

await test("submit live smoke route-handler mode carries full-thread importer mode", async () => {
  const options = parseSubmitSmokeArgs([
    "--route-handler",
    "--importer-mode",
    "full_thread",
    "--blob-pathname",
    `${BLOB_UPLOAD_PATH_PREFIX}zendesk-thread.json`,
    "--support-platform",
    "intercom",
  ], {
    ...ENV,
  });
  const result = await runRouteHandlerSmoke(options, async (req, res) => {
    assert.equal(req.method, "POST");
    assert.equal(req.headers["content-type"], "application/json");
    assert.equal("x-atlas-account-id" in req.headers, false);
    assert.deepEqual(JSON.parse(req.body), {
      blob_pathname: `${BLOB_UPLOAD_PATH_PREFIX}zendesk-thread.json`,
      support_platform: "zendesk",
      company_name: "Atlas Smoke Co.",
      contact_email: "smoke@example.com",
      limit: "1000",
      importer_mode: "full_thread",
    });
    res.statusCode = 200;
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.end(JSON.stringify({
      ok: true,
      request_id: "content-ops-json123",
      account_id: ACCOUNT_ID,
      result_path: "/services/faq-deflection/results/content-ops-json123",
    }));
  });
  assert.deepEqual(result, {
    ok: true,
    source_mode: "portfolio_submit_route",
    statusCode: 200,
    request_id: "content-ops-json123",
    account_id: ACCOUNT_ID,
    result_path: "/services/faq-deflection/results/content-ops-json123",
    error: undefined,
    atlas_status: undefined,
    base_host: "atlas.example.com",
  });
});

await test("submit live smoke Zendesk API mode calls credential route with operator token", async () => {
  const options = parseSubmitSmokeArgs([
    "--zendesk-api",
    "--limit",
    "49",
    "--start-time",
    "1710000000",
  ], {
    ...ENV,
  });
  const result = await runZendeskApiRouteHandlerSmoke(options, async (req, res) => {
    assert.equal(req.method, "POST");
    assert.equal(req.headers["content-type"], "application/json");
    assert.equal(
      req.headers.authorization,
      `Bearer ${ENV.ATLAS_DEFLECTION_ZENDESK_EXPORT_ACCESS_TOKEN}`,
    );
    assert.equal("x-atlas-account-id" in req.headers, false);
    assert.equal(process.env.ATLAS_API_BASE_URL, ENV.ATLAS_API_BASE_URL);
    assert.equal(process.env.ATLAS_B2B_JWT, ENV.ATLAS_B2B_JWT);
    assert.equal(process.env.ATLAS_ACCOUNT_ID, ACCOUNT_ID);
    assert.equal(process.env.BLOB_READ_WRITE_TOKEN, ENV.BLOB_READ_WRITE_TOKEN);
    assert.equal(
      process.env.ATLAS_DEFLECTION_ZENDESK_EXPORT_ACCESS_TOKEN,
      ENV.ATLAS_DEFLECTION_ZENDESK_EXPORT_ACCESS_TOKEN,
    );
    assert.deepEqual(JSON.parse(req.body), {
      company_name: "Atlas Smoke Co.",
      contact_email: "smoke@example.com",
      limit: "49",
      start_time: "1710000000",
    });
    res.statusCode = 200;
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.end(JSON.stringify({
      request_id: "content-ops-zendeskapi456",
      account_id: ACCOUNT_ID,
      result_path: "/services/faq-deflection/results/content-ops-zendeskapi456",
    }));
  });
  assert.deepEqual(result, {
    ok: true,
    source_mode: "portfolio_zendesk_api_route",
    statusCode: 200,
    request_id: "content-ops-zendeskapi456",
    account_id: ACCOUNT_ID,
    result_path: "/services/faq-deflection/results/content-ops-zendeskapi456",
    error: undefined,
    atlas_status: undefined,
    base_host: "atlas.example.com",
  });
});

await test("submit live smoke Zendesk API mode sanitizes route failures", async () => {
  const options = parseSubmitSmokeArgs(["--zendesk-api"], {
    ...ENV,
  });
  const result = await runZendeskApiRouteHandlerSmoke(options, async (_req, res) => {
    res.statusCode = 502;
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.end(JSON.stringify({
      ok: false,
      error: "zendesk_export_unavailable",
      detail: {
        token: ENV.ATLAS_B2B_JWT,
        access_token: ENV.ATLAS_DEFLECTION_ZENDESK_EXPORT_ACCESS_TOKEN,
        blob_token: ENV.BLOB_READ_WRITE_TOKEN,
        tickets: [{ id: 1, subject: "Private ticket" }],
      },
    }));
  });
  assert.deepEqual(result, {
    ok: false,
    source_mode: "portfolio_zendesk_api_route",
    statusCode: 502,
    request_id: undefined,
    account_id: undefined,
    result_path: undefined,
    error: "zendesk_export_unavailable",
    atlas_status: undefined,
    base_host: "atlas.example.com",
  });
  const serialized = JSON.stringify(result);
  assert.equal(serialized.includes(ENV.ATLAS_B2B_JWT), false);
  assert.equal(serialized.includes(ENV.ATLAS_DEFLECTION_ZENDESK_EXPORT_ACCESS_TOKEN), false);
  assert.equal(serialized.includes(ENV.BLOB_READ_WRITE_TOKEN), false);
  assert.equal(serialized.includes("Private ticket"), false);
});

await test("submit live smoke Zendesk API mode fails closed on invalid route JSON", async () => {
  const options = parseSubmitSmokeArgs(["--zendesk-api"], {
    ...ENV,
  });
  const result = await runZendeskApiRouteHandlerSmoke(options, async (_req, res) => {
    res.statusCode = 502;
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.end("{not-json");
  });
  assert.deepEqual(result, {
    ok: false,
    source_mode: "portfolio_zendesk_api_route",
    statusCode: 502,
    error: "portfolio_zendesk_api_route_invalid_json",
    base_host: "atlas.example.com",
  });
});

await test("submit live smoke full-thread local fixture forwards JSON submit", async () => {
  const previousFetch = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push({ url, options });
    assert.equal(url.endsWith(INSPECT_PATH), false);
    assert.equal(url.endsWith(SUBMIT_PATH), true);
    assert.equal(options.body instanceof FormData, true);
    assert.equal(options.body.get("support_platform"), "zendesk");
    assert.equal(options.body.get("importer_mode"), "full_thread");
    assert.equal(options.body.get("csv_file"), null);
    const jsonFile = options.body.get("json_file");
    assert.equal(await jsonFile.text().then((text) => text.includes("Internal note")), true);
    return {
      ok: true,
      status: 200,
      async text() {
        return JSON.stringify({ request_id: "content-ops-localjson123" });
      },
    };
  };
  try {
    const options = parseSubmitSmokeArgs([
      "--importer-mode",
      "full_thread",
    ], {
      ATLAS_API_BASE_URL: ENV.ATLAS_API_BASE_URL,
      ATLAS_B2B_JWT: ENV.ATLAS_B2B_JWT,
      ATLAS_ACCOUNT_ID: ACCOUNT_ID,
    });
    assert.deepEqual(await runSubmitSmoke(options), {
      ok: true,
      source_mode: "local_zendesk_full_thread_fixture",
      statusCode: 200,
      request_id: "content-ops-localjson123",
      account_id: ACCOUNT_ID,
      result_path: "/services/faq-deflection/results/content-ops-localjson123",
      error: undefined,
      atlas_status: undefined,
      base_host: "atlas.example.com",
    });
  } finally {
    globalThis.fetch = previousFetch;
  }
  assert.equal(calls.length, 1);
});

await test("portfolio submit endpoint rejects direct multipart before ATLAS", async () => {
  const previousFetch = globalThis.fetch;
  let fetchCalled = false;
  globalThis.fetch = async (url, options) => {
    fetchCalled = true;
    throw new Error(`must not call ATLAS: ${url} ${options?.method}`);
  };
  const res = mockResponse();
  try {
    await withEnv({ ATLAS_API_BASE_URL: "", ATLAS_B2B_JWT: "", ATLAS_ACCOUNT_ID: "" }, async () => {
      await submitHandler(
        request({
          headers: { "content-length": "999999999" },
        }),
        res,
      );
    });
  } finally {
    globalThis.fetch = previousFetch;
  }

  assert.equal(fetchCalled, false);
  assert.equal(res.statusCode, 410);
  assert.equal(res.headers["Cache-Control"], "no-store");
  assert.deepEqual(JSON.parse(res.body), { ok: false, error: "direct_multipart_deprecated" });
  assert.equal(res.body.includes(ENV.ATLAS_B2B_JWT), false);
});

await test("private blob upload token config fails closed on path and account binding", () => {
  const ok = uploadTokenConfig(
    `${BLOB_UPLOAD_PATH_PREFIX}tickets.csv`,
    "",
    ENV,
  );
  assert.equal(ok.ok, true);
  assert.equal(ok.token, ENV.BLOB_READ_WRITE_TOKEN);
  assert.equal(ok.options.maximumSizeInBytes, MAX_UPLOAD_CSV_BYTES);
  assert.deepEqual(ok.options.allowedContentTypes, BLOB_UPLOAD_CONTENT_TYPES);
  assert.deepEqual(BLOB_UPLOAD_CONTENT_TYPES, [...CSV_CONTENT_TYPES, ...JSON_CONTENT_TYPES]);
  assert.equal(ok.options.addRandomSuffix, true);

  const jsonOk = uploadTokenConfig(
    `${BLOB_UPLOAD_PATH_PREFIX}zendesk-thread.json`,
    "",
    ENV,
  );
  assert.equal(jsonOk.ok, true);
  assert.deepEqual(jsonOk.options.allowedContentTypes, BLOB_UPLOAD_CONTENT_TYPES);

  assert.deepEqual(
    uploadTokenConfig("other/tickets.csv", JSON.stringify({ account_id: ACCOUNT_ID }), ENV),
    { ok: false, errors: ["invalid_blob_pathname"] },
  );
  assert.deepEqual(
    uploadTokenConfig(`${BLOB_UPLOAD_PATH_PREFIX}tickets.txt`, "", ENV),
    { ok: false, errors: ["invalid_blob_pathname"] },
  );
  assert.deepEqual(
    uploadTokenConfig(`${BLOB_UPLOAD_PATH_PREFIX}../tickets.csv`, "", ENV),
    { ok: false, errors: ["invalid_blob_pathname"] },
  );
  assert.deepEqual(
    uploadTokenConfig(
      `${BLOB_UPLOAD_PATH_PREFIX}tickets.csv`,
      JSON.stringify({ account_id: "3b2b950d-f64b-4852-bc30-f92a34cdf169" }),
      ENV,
    ),
    { ok: false, errors: ["invalid_account_id"] },
  );
  assert.deepEqual(uploadTokenConfig(`${BLOB_UPLOAD_PATH_PREFIX}tickets.csv`, "{}", {
    ...ENV,
    ATLAS_ACCOUNT_ID: "",
  }), {
    ok: false,
    errors: ["atlas_account_missing"],
  });
});

await test("private blob submit reads server-side blob and forwards ATLAS multipart", async () => {
  const calls = [];
  const deleteCalls = [];
  const csv = "ticket_id,message\nticket-1,How do I export reports?";
  const result = await submitPrivateBlob({
    config: {
      baseUrl: ENV.ATLAS_API_BASE_URL,
      token: ENV.ATLAS_B2B_JWT,
      accountId: ACCOUNT_ID,
      timeoutMs: 1000,
    },
    blobToken: ENV.BLOB_READ_WRITE_TOKEN,
    payload: {
      blob_pathname: `${BLOB_UPLOAD_PATH_PREFIX}tickets.csv`,
      support_platform: "zendesk",
      company_name: "Acme Co.",
      contact_email: "lead@acme.example",
      limit: "1000",
    },
    getBlobImpl: async (pathname, options) => {
      assert.equal(pathname, `${BLOB_UPLOAD_PATH_PREFIX}tickets.csv`);
      assert.deepEqual(options, {
        access: "private",
        token: ENV.BLOB_READ_WRITE_TOKEN,
        useCache: false,
      });
      return {
        statusCode: 200,
        stream: new Blob([csv], { type: "text/csv" }).stream(),
        blob: {
          size: Buffer.byteLength(csv),
          contentType: "text/csv",
        },
      };
    },
    fetchImpl: async (url, options) => {
      calls.push({ url, options });
      if (url.endsWith(INSPECT_PATH)) {
        return {
          ok: true,
          status: 200,
          async text() {
            return JSON.stringify(READY_INSPECT_PAYLOAD);
          },
        };
      }
      assert.equal(url.endsWith(SUBMIT_PATH), true);
      return {
        ok: true,
        status: 200,
        async text() {
          return JSON.stringify({ request_id: "content-ops-private123" });
        },
      };
    },
    deleteBlobImpl: async (pathname, options) => {
      deleteCalls.push({ pathname, options });
    },
  });

  assert.equal(result.ok, true);
  assert.equal(result.payload.result_path.includes("content-ops-private123"), true);
  assert.equal(calls.length, 2);
  assert.equal(calls[0].url, `${ENV.ATLAS_API_BASE_URL}${INSPECT_PATH}`);
  assert.equal(calls[0].options.headers.Authorization, `Bearer ${ENV.ATLAS_B2B_JWT}`);
  assert.equal("Content-Type" in calls[0].options.headers, false);
  assert.equal(calls[0].options.body instanceof FormData, true);
  assert.equal(calls[0].options.body.get("source_rows"), "true");
  assert.equal(calls[0].options.body.get("target_mode"), "faq_deflection_report");
  assert.equal(await calls[0].options.body.get("file").text(), csv);
  assert.equal(calls[1].url, `${ENV.ATLAS_API_BASE_URL}${SUBMIT_PATH}`);
  assert.equal(calls[1].options.headers.Authorization, `Bearer ${ENV.ATLAS_B2B_JWT}`);
  assert.equal("Content-Type" in calls[1].options.headers, false);
  assert.equal(calls[1].options.body instanceof FormData, true);
  assert.equal(calls[1].options.body.get("support_platform"), "zendesk");
  assert.equal(calls[1].options.body.get("company_name"), "Acme Co.");
  assert.equal(calls[1].options.body.get("limit"), "1000");
  assert.equal(await calls[1].options.body.get("csv_file").text(), csv);
  assert.deepEqual(deleteCalls, [
    {
      pathname: `${BLOB_UPLOAD_PATH_PREFIX}tickets.csv`,
      options: { token: ENV.BLOB_READ_WRITE_TOKEN },
    },
  ]);
});

await test("private blob submit forwards Zendesk full-thread JSON without CSV inspect", async () => {
  const calls = [];
  const deleteCalls = [];
  const threadJson = JSON.stringify({
    ticket: { id: 123, status: "solved", satisfaction_rating: { score: "good" } },
    comments: [
      { id: 1, public: true, author_role: "end_user", body: "How do I export reports?" },
      { id: 2, public: false, author_role: "agent", body: "Private triage note" },
      { id: 3, public: true, author_role: "agent", body: "Open Reports, then Export." },
    ],
  });
  const result = await submitPrivateBlob({
    config: {
      baseUrl: ENV.ATLAS_API_BASE_URL,
      token: ENV.ATLAS_B2B_JWT,
      accountId: ACCOUNT_ID,
      timeoutMs: 1000,
    },
    blobToken: ENV.BLOB_READ_WRITE_TOKEN,
    payload: {
      blob_pathname: `${BLOB_UPLOAD_PATH_PREFIX}zendesk-thread.json`,
      importer_mode: FULL_THREAD_IMPORTER_MODE,
      support_platform: "intercom",
      company_name: "Acme Co.",
      contact_email: "lead@acme.example",
      limit: "1000",
    },
    getBlobImpl: async (pathname, options) => {
      assert.equal(pathname, `${BLOB_UPLOAD_PATH_PREFIX}zendesk-thread.json`);
      assert.deepEqual(options, {
        access: "private",
        token: ENV.BLOB_READ_WRITE_TOKEN,
        useCache: false,
      });
      return {
        statusCode: 200,
        stream: new Blob([threadJson], { type: "application/json" }).stream(),
        blob: {
          size: Buffer.byteLength(threadJson),
          contentType: "application/json",
        },
      };
    },
    fetchImpl: async (url, options) => {
      calls.push({ url, options });
      assert.equal(url.endsWith(INSPECT_PATH), false);
      assert.equal(url.endsWith(SUBMIT_PATH), true);
      return {
        ok: true,
        status: 200,
        async text() {
          return JSON.stringify({ request_id: "content-ops-json123" });
        },
      };
    },
    deleteBlobImpl: async (pathname, options) => {
      deleteCalls.push({ pathname, options });
    },
  });

  assert.equal(result.ok, true);
  assert.equal(result.payload.result_path.includes("content-ops-json123"), true);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, `${ENV.ATLAS_API_BASE_URL}${SUBMIT_PATH}`);
  assert.equal(calls[0].options.headers.Authorization, `Bearer ${ENV.ATLAS_B2B_JWT}`);
  assert.equal("Content-Type" in calls[0].options.headers, false);
  assert.equal(calls[0].options.body instanceof FormData, true);
  assert.equal(calls[0].options.body.get("support_platform"), "zendesk");
  assert.equal(calls[0].options.body.get("company_name"), "Acme Co.");
  assert.equal(calls[0].options.body.get("limit"), "1000");
  assert.equal(calls[0].options.body.get("importer_mode"), FULL_THREAD_IMPORTER_MODE);
  assert.equal(await calls[0].options.body.get("json_file").text(), threadJson);
  assert.equal(calls[0].options.body.get("csv_file"), null);
  assert.deepEqual(deleteCalls, [
    {
      pathname: `${BLOB_UPLOAD_PATH_PREFIX}zendesk-thread.json`,
      options: { token: ENV.BLOB_READ_WRITE_TOKEN },
    },
  ]);
});

await test("Zendesk credential flow exports server-side, stores private JSON, then submits", async () => {
  const calls = [];
  const putCalls = [];
  const deleteCalls = [];
  const artifact = {
    tickets: [
      {
        ticket: { id: 123, status: "solved", satisfaction_rating: { score: "good" } },
        comments: [
          { id: 1, public: true, author_role: "end_user", body: "How do I export reports?" },
          { id: 2, public: true, author_role: "agent", body: "Open Reports, then Export." },
        ],
      },
    ],
  };
  let storedBody = null;
  const result = await submitZendeskCredentialFlow({
    config: {
      baseUrl: ENV.ATLAS_API_BASE_URL,
      token: ENV.ATLAS_B2B_JWT,
      accountId: ACCOUNT_ID,
      timeoutMs: 1000,
    },
    blobToken: ENV.BLOB_READ_WRITE_TOKEN,
    payload: {
      account_id: "3b2b950d-f64b-4852-bc30-f92a34cdf169",
      company_name: "Acme Co.",
      contact_email: "lead@acme.example",
      limit: "1000",
      start_time: "0",
    },
    nowMs: 123456,
    fetchImpl: async (url, options) => {
      calls.push({ url, options });
      if (url.endsWith(ZENDESK_EXPORT_PATH)) {
        return {
          ok: true,
          status: 200,
          async text() {
            return JSON.stringify({
              importer_mode: FULL_THREAD_IMPORTER_MODE,
              support_platform: "zendesk",
              ticket_count: 1,
              limit: 1000,
              start_time: 0,
              artifact,
            });
          },
        };
      }
      assert.equal(url.endsWith(SUBMIT_PATH), true);
      return {
        ok: true,
        status: 200,
        async text() {
          return JSON.stringify({ request_id: "content-ops-zendeskapi123" });
        },
      };
    },
    putBlobImpl: async (pathname, body, options) => {
      putCalls.push({ pathname, body, options });
      storedBody = body;
      return {
        pathname: `${BLOB_UPLOAD_PATH_PREFIX}123456-zendesk-api-export-random.json`,
        url: `https://blob.example.com/private/zendesk-api-export.json?token=${ENV.BLOB_READ_WRITE_TOKEN}`,
      };
    },
    getBlobImpl: async (pathname, options) => {
      assert.equal(pathname, `${BLOB_UPLOAD_PATH_PREFIX}123456-zendesk-api-export-random.json`);
      assert.deepEqual(options, {
        access: "private",
        token: ENV.BLOB_READ_WRITE_TOKEN,
        useCache: false,
      });
      return {
        statusCode: 200,
        stream: new Blob([storedBody], { type: "application/json" }).stream(),
        blob: {
          size: storedBody.length,
          contentType: "application/json",
        },
      };
    },
    deleteBlobImpl: async (pathname, options) => {
      deleteCalls.push({ pathname, options });
    },
  });

  assert.equal(result.ok, true);
  assert.equal(result.payload.result_path.includes("content-ops-zendeskapi123"), true);
  const resultJson = JSON.stringify(result);
  assert.equal(resultJson.includes("tickets"), false);
  assert.equal(resultJson.includes("https://blob.example.com/private"), false);
  assert.equal(resultJson.includes(ENV.BLOB_READ_WRITE_TOKEN), false);
  assert.equal(resultJson.includes(ENV.ATLAS_B2B_JWT), false);
  assert.equal(calls.length, 2);
  assert.equal(calls[0].url, `${ENV.ATLAS_API_BASE_URL}${ZENDESK_EXPORT_PATH}`);
  assert.equal(calls[0].options.headers.Authorization, `Bearer ${ENV.ATLAS_B2B_JWT}`);
  assert.equal(calls[0].options.headers["Content-Type"], "application/json");
  assert.deepEqual(JSON.parse(calls[0].options.body), { limit: 1000, start_time: 0 });
  assert.equal(calls[0].options.body.includes("account_id"), false);
  assert.equal(calls[0].options.body.includes(ENV.BLOB_READ_WRITE_TOKEN), false);
  assert.equal(putCalls.length, 1);
  assert.equal(putCalls[0].pathname, `${BLOB_UPLOAD_PATH_PREFIX}123456-zendesk-api-export.json`);
  assert.deepEqual(putCalls[0].options, {
    access: "private",
    addRandomSuffix: true,
    contentType: "application/json",
    token: ENV.BLOB_READ_WRITE_TOKEN,
  });
  assert.equal(putCalls[0].body.toString("utf8"), JSON.stringify(artifact));
  assert.equal(calls[1].url, `${ENV.ATLAS_API_BASE_URL}${SUBMIT_PATH}`);
  assert.equal(calls[1].options.body.get("support_platform"), "zendesk");
  assert.equal(calls[1].options.body.get("importer_mode"), FULL_THREAD_IMPORTER_MODE);
  assert.equal(await calls[1].options.body.get("json_file").text(), JSON.stringify(artifact));
  assert.deepEqual(deleteCalls, [
    {
      pathname: `${BLOB_UPLOAD_PATH_PREFIX}123456-zendesk-api-export-random.json`,
      options: { token: ENV.BLOB_READ_WRITE_TOKEN },
    },
  ]);
});

await test("Zendesk credential flow fails closed when private Blob write fails", async () => {
  const calls = [];
  const artifact = {
    tickets: [
      {
        ticket: { id: 123, status: "solved" },
        comments: [{ id: 1, public: true, author_role: "agent", body: "Use Reports." }],
      },
    ],
  };

  const result = await submitZendeskCredentialFlow({
    config: {
      baseUrl: ENV.ATLAS_API_BASE_URL,
      token: ENV.ATLAS_B2B_JWT,
      accountId: ACCOUNT_ID,
      timeoutMs: 1000,
    },
    blobToken: ENV.BLOB_READ_WRITE_TOKEN,
    payload: {
      company_name: "Acme Co.",
      contact_email: "lead@acme.example",
      limit: "1000",
    },
    fetchImpl: async (url, options) => {
      calls.push({ url, options });
      if (url.endsWith(ZENDESK_EXPORT_PATH)) {
        return {
          ok: true,
          status: 200,
          async text() {
            return JSON.stringify({
              importer_mode: FULL_THREAD_IMPORTER_MODE,
              support_platform: "zendesk",
              artifact,
            });
          },
        };
      }
      throw new Error("must not submit after failed Blob write");
    },
    putBlobImpl: async () => {
      throw new Error(`blob failed with ${ENV.BLOB_READ_WRITE_TOKEN}`);
    },
  });

  assert.deepEqual(result, {
    ok: false,
    statusCode: 502,
    error: "zendesk_export_blob_unavailable",
  });
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, `${ENV.ATLAS_API_BASE_URL}${ZENDESK_EXPORT_PATH}`);
  assert.equal(JSON.stringify(result).includes(ENV.BLOB_READ_WRITE_TOKEN), false);
  assert.equal(JSON.stringify(result).includes("tickets"), false);
});

await test("Zendesk credential flow fails closed on malformed Blob write envelope", async () => {
  const result = await submitZendeskCredentialFlow({
    config: {
      baseUrl: ENV.ATLAS_API_BASE_URL,
      token: ENV.ATLAS_B2B_JWT,
      accountId: ACCOUNT_ID,
      timeoutMs: 1000,
    },
    blobToken: ENV.BLOB_READ_WRITE_TOKEN,
    payload: {
      company_name: "Acme Co.",
      contact_email: "lead@acme.example",
      limit: "1000",
    },
    fetchImpl: async (url) => {
      assert.equal(url.endsWith(ZENDESK_EXPORT_PATH), true);
      return {
        ok: true,
        status: 200,
        async text() {
          return JSON.stringify({
            importer_mode: FULL_THREAD_IMPORTER_MODE,
            support_platform: "zendesk",
            artifact: { tickets: [{ ticket: { id: 123 }, comments: [] }] },
          });
        },
      };
    },
    putBlobImpl: async () => ({ url: "https://blob.example.com/private/no-pathname.json" }),
  });

  assert.deepEqual(result, {
    ok: false,
    statusCode: 502,
    error: "zendesk_export_blob_contract_violation",
  });
  assert.equal(JSON.stringify(result).includes("https://blob.example.com/private"), false);
});

await test("Zendesk credential flow preserves submit failure after export and Blob success", async () => {
  const calls = [];
  const deleteCalls = [];
  let storedBody = null;
  const result = await submitZendeskCredentialFlow({
    config: {
      baseUrl: ENV.ATLAS_API_BASE_URL,
      token: ENV.ATLAS_B2B_JWT,
      accountId: ACCOUNT_ID,
      timeoutMs: 1000,
    },
    blobToken: ENV.BLOB_READ_WRITE_TOKEN,
    payload: {
      company_name: "Acme Co.",
      contact_email: "lead@acme.example",
      limit: "1000",
    },
    nowMs: 222,
    fetchImpl: async (url, options) => {
      calls.push({ url, options });
      if (url.endsWith(ZENDESK_EXPORT_PATH)) {
        return {
          ok: true,
          status: 200,
          async text() {
            return JSON.stringify({
              importer_mode: FULL_THREAD_IMPORTER_MODE,
              support_platform: "zendesk",
              artifact: { tickets: [{ ticket: { id: 123 }, comments: [] }] },
            });
          },
        };
      }
      assert.equal(url.endsWith(SUBMIT_PATH), true);
      return {
        ok: false,
        status: 503,
        async text() {
          return JSON.stringify({
            error: "upstream failed",
            detail: ENV.ATLAS_B2B_JWT,
          });
        },
      };
    },
    putBlobImpl: async (_pathname, body) => {
      storedBody = body;
      return { pathname: `${BLOB_UPLOAD_PATH_PREFIX}222-zendesk-api-export-random.json` };
    },
    getBlobImpl: async (pathname, options) => {
      assert.equal(pathname, `${BLOB_UPLOAD_PATH_PREFIX}222-zendesk-api-export-random.json`);
      assert.deepEqual(options, {
        access: "private",
        token: ENV.BLOB_READ_WRITE_TOKEN,
        useCache: false,
      });
      return {
        statusCode: 200,
        stream: new Blob([storedBody], { type: "application/json" }).stream(),
        blob: {
          size: storedBody.length,
          contentType: "application/json",
        },
      };
    },
    deleteBlobImpl: async (pathname, options) => {
      deleteCalls.push({ pathname, options });
    },
  });

  assert.deepEqual(result, {
    ok: false,
    statusCode: 502,
    error: "atlas_submit_failed",
    atlas_status: 503,
  });
  assert.equal(calls.length, 2);
  assert.equal(calls[0].url, `${ENV.ATLAS_API_BASE_URL}${ZENDESK_EXPORT_PATH}`);
  assert.equal(calls[1].url, `${ENV.ATLAS_API_BASE_URL}${SUBMIT_PATH}`);
  assert.deepEqual(deleteCalls, [
    {
      pathname: `${BLOB_UPLOAD_PATH_PREFIX}222-zendesk-api-export-random.json`,
      options: { token: ENV.BLOB_READ_WRITE_TOKEN },
    },
  ]);
  assert.equal(JSON.stringify(result).includes(ENV.ATLAS_B2B_JWT), false);
  assert.equal(JSON.stringify(result).includes("tickets"), false);
});

await test("Zendesk credential flow fails closed on malformed export envelope", async () => {
  let putCalled = false;
  const result = await submitZendeskCredentialFlow({
    config: {
      baseUrl: ENV.ATLAS_API_BASE_URL,
      token: ENV.ATLAS_B2B_JWT,
      accountId: ACCOUNT_ID,
      timeoutMs: 1000,
    },
    blobToken: ENV.BLOB_READ_WRITE_TOKEN,
    payload: {
      company_name: "Acme Co.",
      contact_email: "lead@acme.example",
      limit: "1000",
    },
    fetchImpl: async () => ({
      ok: true,
      status: 200,
      async text() {
        return JSON.stringify({
          importer_mode: FULL_THREAD_IMPORTER_MODE,
          support_platform: "zendesk",
          artifact: { ticket: [] },
        });
      },
    }),
    putBlobImpl: async () => {
      putCalled = true;
      throw new Error("must not persist malformed export");
    },
  });

  assert.deepEqual(result, {
    ok: false,
    statusCode: 502,
    error: "zendesk_export_contract_violation",
  });
  assert.equal(putCalled, false);
});

await test("Zendesk credential flow reports missing credentials without leaking upstream body", async () => {
  let putCalled = false;
  const result = await submitZendeskCredentialFlow({
    config: {
      baseUrl: ENV.ATLAS_API_BASE_URL,
      token: ENV.ATLAS_B2B_JWT,
      accountId: ACCOUNT_ID,
      timeoutMs: 1000,
    },
    blobToken: ENV.BLOB_READ_WRITE_TOKEN,
    payload: {
      company_name: "Acme Co.",
      contact_email: "lead@acme.example",
      limit: "1000",
    },
    fetchImpl: async () => ({
      ok: false,
      status: 404,
      async text() {
        return JSON.stringify({
          detail: {
            reason: "zendesk_credentials_missing",
            message: `missing ${ENV.ATLAS_B2B_JWT}`,
          },
        });
      },
    }),
    putBlobImpl: async () => {
      putCalled = true;
      throw new Error("must not persist failed export");
    },
  });

  assert.deepEqual(result, {
    ok: false,
    statusCode: 404,
    error: "zendesk_credentials_missing",
    atlas_status: 404,
  });
  assert.equal(JSON.stringify(result).includes(ENV.ATLAS_B2B_JWT), false);
  assert.equal(putCalled, false);
});

await test("Zendesk credential flow endpoint gates method, content type, and access token", async () => {
  const previousFetch = globalThis.fetch;
  let fetchCalled = false;
  globalThis.fetch = async () => {
    fetchCalled = true;
    throw new Error("must not call ATLAS before route auth passes");
  };
  try {
    const methodRes = mockResponse();
    await zendeskExportSubmitHandler(
      request({
        method: "GET",
        headers: {
          "content-type": "application/json",
          authorization: `Bearer ${ENV.ATLAS_DEFLECTION_ZENDESK_EXPORT_ACCESS_TOKEN}`,
        },
        body: Buffer.from("{}"),
      }),
      methodRes,
    );
    assert.equal(methodRes.statusCode, 405);
    assert.deepEqual(JSON.parse(methodRes.body), { ok: false, error: "method_not_allowed" });

    const contentTypeRes = mockResponse();
    await zendeskExportSubmitHandler(
      request({
        headers: {
          "content-type": "text/plain",
          authorization: `Bearer ${ENV.ATLAS_DEFLECTION_ZENDESK_EXPORT_ACCESS_TOKEN}`,
        },
        body: Buffer.from("{}"),
      }),
      contentTypeRes,
    );
    assert.equal(contentTypeRes.statusCode, 415);
    assert.deepEqual(JSON.parse(contentTypeRes.body), {
      ok: false,
      error: "zendesk_export_json_required",
    });

    const authRes = mockResponse();
    await withEnv(ENV, async () => {
      await zendeskExportSubmitHandler(
        request({
          headers: { "content-type": "application/json" },
          body: Buffer.from(JSON.stringify({
            company_name: "Acme Co.",
            contact_email: "lead@acme.example",
          })),
        }),
        authRes,
      );
    });
    assert.equal(authRes.statusCode, 401);
    assert.deepEqual(JSON.parse(authRes.body), {
      ok: false,
      error: "zendesk_export_auth_required",
    });
    assert.equal(authRes.body.includes(ENV.ATLAS_DEFLECTION_ZENDESK_EXPORT_ACCESS_TOKEN), false);
    assert.equal(authRes.body.includes(ENV.ATLAS_B2B_JWT), false);
    assert.equal(fetchCalled, false);
  } finally {
    globalThis.fetch = previousFetch;
  }
});

await test("Zendesk credential flow endpoint rejects account spoofing before ATLAS", async () => {
  const previousFetch = globalThis.fetch;
  let fetchCalled = false;
  globalThis.fetch = async (url, options) => {
    fetchCalled = true;
    throw new Error(`must not call ATLAS: ${url} ${options?.method}`);
  };
  const res = mockResponse();
  try {
    await withEnv(ENV, async () => {
      await zendeskExportSubmitHandler(
        request({
          headers: {
            "content-type": "application/json",
            "authorization": `Bearer ${ENV.ATLAS_DEFLECTION_ZENDESK_EXPORT_ACCESS_TOKEN}`,
            "x-atlas-account-id": "3b2b950d-f64b-4852-bc30-f92a34cdf169",
          },
          body: Buffer.from(JSON.stringify({
            company_name: "Acme Co.",
            contact_email: "lead@acme.example",
          })),
        }),
        res,
      );
    });
  } finally {
    globalThis.fetch = previousFetch;
  }

  assert.equal(fetchCalled, false);
  assert.equal(res.statusCode, 400);
  assert.equal(res.headers["Cache-Control"], "no-store");
  assert.deepEqual(JSON.parse(res.body), { ok: false, error: "invalid_account_id" });
  assert.equal(res.body.includes(ENV.ATLAS_B2B_JWT), false);
});

await test("private blob submit blocks not-ready inspect before report submit", async () => {
  const calls = [];
  const deleteCalls = [];
  const csv = "ticket_id,message\nticket-1,How do I export reports?";
  const result = await submitPrivateBlob({
    config: {
      baseUrl: ENV.ATLAS_API_BASE_URL,
      token: ENV.ATLAS_B2B_JWT,
      accountId: ACCOUNT_ID,
      timeoutMs: 1000,
    },
    blobToken: ENV.BLOB_READ_WRITE_TOKEN,
    payload: {
      blob_pathname: `${BLOB_UPLOAD_PATH_PREFIX}tickets.csv`,
      support_platform: "zendesk",
      company_name: "Acme Co.",
      contact_email: "lead@acme.example",
      limit: "1000",
    },
    getBlobImpl: async () => ({
      statusCode: 200,
      stream: new Blob([csv], { type: "text/csv" }).stream(),
      blob: {
        size: Buffer.byteLength(csv),
        contentType: "text/csv",
      },
    }),
    fetchImpl: async (url, options) => {
      calls.push({ url, options });
      if (url.endsWith(SUBMIT_PATH)) {
        throw new Error("not-ready inspect must block report submit");
      }
      return {
        ok: true,
        status: 200,
        async text() {
          return JSON.stringify(NOT_READY_INSPECT_PAYLOAD);
        },
      };
    },
    deleteBlobImpl: async (pathname, options) => {
      deleteCalls.push({ pathname, options });
    },
  });

  assert.deepEqual(result, {
    ok: false,
    statusCode: 400,
    error: "deflection_inspect_not_ready",
  });
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, `${ENV.ATLAS_API_BASE_URL}${INSPECT_PATH}`);
  assert.deepEqual(deleteCalls, [
    {
      pathname: `${BLOB_UPLOAD_PATH_PREFIX}tickets.csv`,
      options: { token: ENV.BLOB_READ_WRITE_TOKEN },
    },
  ]);
});

await test("private blob cleanup preserves ATLAS failure result", async () => {
  const deleteCalls = [];
  const csv = "ticket_id,message\nticket-1,How do I export reports?";
  const result = await submitPrivateBlob({
    config: {
      baseUrl: ENV.ATLAS_API_BASE_URL,
      token: ENV.ATLAS_B2B_JWT,
      accountId: ACCOUNT_ID,
      timeoutMs: 1000,
    },
    blobToken: ENV.BLOB_READ_WRITE_TOKEN,
    payload: {
      blob_pathname: `${BLOB_UPLOAD_PATH_PREFIX}tickets.csv`,
      support_platform: "zendesk",
      company_name: "Acme Co.",
      contact_email: "lead@acme.example",
      limit: "1000",
    },
    getBlobImpl: async () => ({
      statusCode: 200,
      stream: new Blob([csv], { type: "text/csv" }).stream(),
      blob: {
        size: Buffer.byteLength(csv),
        contentType: "text/csv",
      },
    }),
    fetchImpl: async (url) => {
      if (url.endsWith(INSPECT_PATH)) {
        return {
          ok: true,
          status: 200,
          async text() {
            return JSON.stringify(READY_INSPECT_PAYLOAD);
          },
        };
      }
      return {
        ok: false,
        status: 502,
        async text() {
          return JSON.stringify({ error: "upstream failed" });
        },
      };
    },
    deleteBlobImpl: async (pathname, options) => {
      deleteCalls.push({ pathname, options });
    },
  });

  assert.deepEqual(result, {
    ok: false,
    statusCode: 502,
    error: "atlas_submit_failed",
    atlas_status: 502,
  });
  assert.deepEqual(deleteCalls, [
    {
      pathname: `${BLOB_UPLOAD_PATH_PREFIX}tickets.csv`,
      options: { token: ENV.BLOB_READ_WRITE_TOKEN },
    },
  ]);
});

await test("private blob cleanup failure does not mask submit success", async () => {
  const csv = "ticket_id,message\nticket-1,How do I export reports?";
  const events = [];
  const result = await submitPrivateBlob({
    config: {
      baseUrl: ENV.ATLAS_API_BASE_URL,
      token: ENV.ATLAS_B2B_JWT,
      accountId: ACCOUNT_ID,
      timeoutMs: 1000,
    },
    blobToken: ENV.BLOB_READ_WRITE_TOKEN,
    payload: {
      blob_pathname: `${BLOB_UPLOAD_PATH_PREFIX}tickets.csv`,
      support_platform: "zendesk",
      company_name: "Acme Co.",
      contact_email: "lead@acme.example",
      limit: "1000",
    },
    getBlobImpl: async () => ({
      statusCode: 200,
      stream: new Blob([csv], { type: "text/csv" }).stream(),
      blob: {
        size: Buffer.byteLength(csv),
        contentType: "text/csv",
      },
    }),
    fetchImpl: async (url) => {
      if (url.endsWith(INSPECT_PATH)) {
        return {
          ok: true,
          status: 200,
          async text() {
            return JSON.stringify(READY_INSPECT_PAYLOAD);
          },
        };
      }
      return {
        ok: true,
        status: 200,
        async text() {
          return JSON.stringify({ request_id: "content-ops-cleanup123" });
        },
      };
    },
    deleteBlobImpl: async () => {
      throw new Error("delete unavailable");
    },
    eventLogger: (event, fields) => {
      events.push({ event, fields });
    },
  });

  assert.equal(result.ok, true);
  assert.equal(result.payload.request_id, "content-ops-cleanup123");
  assert.deepEqual(events, [
    {
      event: "faq_deflection_private_blob_cleanup_failed",
      fields: {
        pathname: `${BLOB_UPLOAD_PATH_PREFIX}tickets.csv`,
        error: "delete_failed",
      },
    },
  ]);
});

await test("private blob cleanup event logging is secondary", async () => {
  const result = await cleanupPrivateCsvBlob({
    pathname: `${BLOB_UPLOAD_PATH_PREFIX}tickets.csv`,
    token: ENV.BLOB_READ_WRITE_TOKEN,
    deleteBlobImpl: async () => {
      throw new Error("delete unavailable");
    },
    eventLogger: () => {
      throw new Error("logger unavailable");
    },
  });

  assert.deepEqual(result, { ok: false, error: "private_blob_cleanup_failed" });
});

await test("deflection server events redact secret-shaped fields", () => {
  assert.deepEqual(
    sanitizeDeflectionEventFields({
      pathname: `${BLOB_UPLOAD_PATH_PREFIX}tickets.csv`,
      token: ENV.BLOB_READ_WRITE_TOKEN,
      Authorization: "Bearer secret",
      contact_email: "lead@acme.example",
      error: "x".repeat(300),
      detail: `delete failed for https://blob.example.com/file.csv?token=${ENV.BLOB_READ_WRITE_TOKEN}`,
      ok: false,
      count: 3,
    }),
    {
      pathname: `${BLOB_UPLOAD_PATH_PREFIX}tickets.csv`,
      error: "x".repeat(240),
      detail: "[redacted]",
      ok: false,
      count: 3,
    },
  );

  const emitted = emitDeflectionServerEvent(
    "not a valid event name",
    { token: "secret", outcome: "cleanup_failed" },
    () => {},
  );
  assert.deepEqual(emitted, {
    ok: true,
    event: "faq_deflection_server_event",
    fields: { outcome: "cleanup_failed" },
  });
});

await test("private blob cleanup rejects unsafe references before deleting", async () => {
  let deleteCalled = false;
  assert.deepEqual(
    await cleanupPrivateCsvBlob({
      pathname: "../tickets.csv",
      token: ENV.BLOB_READ_WRITE_TOKEN,
      deleteBlobImpl: async () => {
        deleteCalled = true;
      },
    }),
    { ok: false, skipped: true, error: "invalid_blob_reference" },
  );
  assert.equal(deleteCalled, false);
});

await test("private blob reader rejects unsafe or oversized blob references", async () => {
  assert.deepEqual(
    await readPrivateCsvBlob({
      pathname: "../tickets.csv",
      token: ENV.BLOB_READ_WRITE_TOKEN,
      getBlobImpl: async () => {
        throw new Error("must not call blob store");
      },
    }),
    { ok: false, statusCode: 400, error: "invalid_blob_reference" },
  );

  assert.deepEqual(
    await readPrivateCsvBlob({
      pathname: `${BLOB_UPLOAD_PATH_PREFIX}tickets.csv`,
      token: ENV.BLOB_READ_WRITE_TOKEN,
      getBlobImpl: async () => ({
        statusCode: 200,
        stream: new Blob(["x"]).stream(),
        blob: { size: MAX_BLOB_CSV_BYTES + 1, contentType: "text/csv" },
      }),
    }),
    { ok: false, statusCode: 413, error: "deflection_submit_csv_too_large" },
  );
});

await test("private blob submit fails closed on importer mode and blob extension mismatch", async () => {
  let externalCalled = false;
  const common = {
    config: {
      baseUrl: ENV.ATLAS_API_BASE_URL,
      token: ENV.ATLAS_B2B_JWT,
      accountId: ACCOUNT_ID,
      timeoutMs: 1000,
    },
    blobToken: ENV.BLOB_READ_WRITE_TOKEN,
    getBlobImpl: async () => {
      externalCalled = true;
      throw new Error("must not call blob store");
    },
    fetchImpl: async () => {
      externalCalled = true;
      throw new Error("must not call ATLAS");
    },
    deleteBlobImpl: async () => {
      externalCalled = true;
    },
  };

  assert.deepEqual(
    await submitPrivateBlob({
      ...common,
      payload: {
        blob_pathname: `${BLOB_UPLOAD_PATH_PREFIX}tickets.csv`,
        importer_mode: FULL_THREAD_IMPORTER_MODE,
      },
    }),
    { ok: false, statusCode: 400, error: "invalid_blob_reference" },
  );
  assert.deepEqual(
    await submitPrivateBlob({
      ...common,
      payload: {
        blob_pathname: `${BLOB_UPLOAD_PATH_PREFIX}zendesk-thread.json`,
      },
    }),
    { ok: false, statusCode: 400, error: "invalid_blob_reference" },
  );
  assert.deepEqual(
    await submitPrivateBlob({
      ...common,
      payload: {
        blob_pathname: `${BLOB_UPLOAD_PATH_PREFIX}zendesk-thread.json`,
        importer_mode: "raw_json",
      },
    }),
    { ok: false, statusCode: 400, error: "invalid_importer_mode" },
  );
  assert.equal(externalCalled, false);
});

await test("portfolio submit forwarding times out hung ATLAS calls", async () => {
  const result = await forwardSubmit({
    config: {
      baseUrl: ENV.ATLAS_API_BASE_URL,
      token: ENV.ATLAS_B2B_JWT,
      accountId: ACCOUNT_ID,
      timeoutMs: 1,
    },
    contentType: "multipart/form-data; boundary=atlas",
    body: MULTIPART_BODY,
    fetchImpl: async (_url, options) =>
      new Promise((_resolve, reject) => {
        assert.equal(options.signal instanceof AbortSignal, true);
        options.signal.addEventListener("abort", () => {
          const error = new Error("aborted");
          error.name = "AbortError";
          reject(error);
        });
      }),
  });

  assert.deepEqual(result, {
    ok: false,
    statusCode: 504,
    error: "atlas_submit_unreachable",
  });
});

await test("portfolio submit endpoint rejects account mismatch before ATLAS", async () => {
  const previousFetch = globalThis.fetch;
  let fetchCalled = false;
  globalThis.fetch = async () => {
    fetchCalled = true;
    throw new Error("must not call ATLAS");
  };
  const res = mockResponse();
  try {
    await withEnv(ENV, async () => {
      await submitHandler(
        request({
          headers: {
            "content-type": "application/json",
            "x-atlas-account-id": "3b2b950d-f64b-4852-bc30-f92a34cdf169",
          },
          body: JSON.stringify({ blob_pathname: `${BLOB_UPLOAD_PATH_PREFIX}tickets.csv` }),
        }),
        res,
      );
    });
  } finally {
    globalThis.fetch = previousFetch;
  }
  assert.equal(fetchCalled, false);
  assert.equal(res.statusCode, 400);
  assert.deepEqual(JSON.parse(res.body), { ok: false, error: "invalid_account_id" });

  const blobRes = mockResponse();
  await withEnv(ENV, async () => {
    await submitHandler(
      request({
        headers: {
          "content-type": "application/json",
          "x-atlas-account-id": "3b2b950d-f64b-4852-bc30-f92a34cdf169",
        },
        body: JSON.stringify({ blob_pathname: `${BLOB_UPLOAD_PATH_PREFIX}tickets.csv` }),
      }),
      blobRes,
    );
  });
  assert.equal(blobRes.statusCode, 400);
  assert.deepEqual(JSON.parse(blobRes.body), { ok: false, error: "invalid_account_id" });
});

await test("private blob submit uses configured account when browser omits account header", async () => {
  const res = mockResponse();
  await withEnv(ENV, async () => {
    await submitHandler(
      request({
        headers: {
          "content-type": "application/json",
          "x-atlas-account-id": undefined,
        },
        body: JSON.stringify({ blob_pathname: "../tickets.csv" }),
      }),
      res,
    );
  });
  assert.equal(res.statusCode, 400);
  assert.deepEqual(JSON.parse(res.body), { ok: false, error: "invalid_blob_reference" });
});

await test("portfolio submit endpoint hides missing config details", async () => {
  const res = mockResponse();
  await withEnv({ ATLAS_API_BASE_URL: "", ATLAS_B2B_JWT: "", ATLAS_ACCOUNT_ID: "" }, async () => {
    await submitHandler(
      request({
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ blob_pathname: `${BLOB_UPLOAD_PATH_PREFIX}tickets.csv` }),
      }),
      res,
    );
  });
  assert.equal(res.statusCode, 503);
  assert.deepEqual(JSON.parse(res.body), {
    ok: false,
    error: "atlas_submit_not_configured",
  });
  assert.equal(res.body.includes("ATLAS_B2B_JWT"), false);
});

await test("portfolio submit endpoint only accepts POST JSON blob requests", async () => {
  const getRes = mockResponse();
  await submitHandler(request({ method: "GET" }), getRes);
  assert.equal(getRes.statusCode, 405);
  assert.equal(getRes.headers.Allow, "POST");
  assert.deepEqual(JSON.parse(getRes.body), {
    ok: false,
    error: "method_not_allowed",
  });

  const multipartRes = mockResponse();
  await submitHandler(request(), multipartRes);
  assert.equal(multipartRes.statusCode, 410);
  assert.deepEqual(JSON.parse(multipartRes.body), {
    ok: false,
    error: "direct_multipart_deprecated",
  });

  const textRes = mockResponse();
  await submitHandler(request({ headers: { "content-type": "text/plain" } }), textRes);
  assert.equal(textRes.statusCode, 415);
  assert.deepEqual(JSON.parse(textRes.body), {
    ok: false,
    error: "private_blob_json_required",
  });
});

await test("upload shell test is enrolled in package scripts", () => {
  assert.equal(packageJson.dependencies["@vercel/blob"], "^2.4.1");
  assert.equal(
    packageJson.scripts["test:deflection-upload-shell"],
    "node scripts/faq-deflection-upload-shell.test.mjs",
  );
  assert.equal(
    packageJson.scripts["smoke:deflection-submit-live"],
    "node scripts/faq-deflection-submit-live-smoke.mjs",
  );
});
