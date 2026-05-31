import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import submitHandler, {
  BLOB_UPLOAD_PATH_PREFIX,
  MAX_BLOB_CSV_BYTES,
  MAX_CSV_BYTES,
  MAX_MULTIPART_OVERHEAD_BYTES,
  SUBMIT_PATH,
  cleanupPrivateCsvBlob,
  forwardSubmit,
  readPrivateCsvBlob,
  submitPrivateBlob,
} from "../api/content-ops/deflection/submit.js";
import {
  CSV_CONTENT_TYPES,
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
  validationErrors as submitSmokeValidationErrors,
} from "./faq-deflection-submit-live-smoke.mjs";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const appSource = await readFile(resolve(root, "src/App.tsx"), "utf8");
const servicesSource = await readFile(resolve(root, "src/pages/Services.tsx"), "utf8");
const uploadSource = await readFile(resolve(root, "src/pages/FaqDeflectionUpload.tsx"), "utf8");
const submitSource = await readFile(resolve(root, "api/content-ops/deflection/submit.js"), "utf8");
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
  BLOB_READ_WRITE_TOKEN: "vercel_blob_rw_token",
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
    "data-atlas-deflection-company",
    "data-atlas-deflection-contact-email",
    "data-atlas-deflection-support-platform",
    "data-atlas-deflection-submit",
    "data-atlas-deflection-upload-endpoint",
    "data-atlas-deflection-upload-progress",
    "data-atlas-deflection-retry",
  ]) {
    assert.match(uploadSource, new RegExp(marker));
  }
  assert.match(uploadSource, /MAX_CSV_BYTES = 50 \* 1024 \* 1024/);
  assert.match(uploadSource, /@vercel\/blob\/client/);
  assert.doesNotMatch(blobUploadRouteSource, /^import .*@vercel\/blob\/client/m);
  assert.match(blobUploadRouteSource, /import\("@vercel\/blob\/client"\)/);
  assert.match(uploadSource, /access: "private"/);
  assert.match(uploadSource, /onUploadProgress/);
  assert.match(uploadSource, /boundedProgress\(event\.percentage\)/);
  assert.match(uploadSource, /role="progressbar"/);
  assert.match(uploadSource, /aria-valuenow=\{submit\.percentage\}/);
  assert.match(uploadSource, /Retry upload/);
  assert.match(uploadSource, /starts a new private upload/);
  assert.match(uploadSource, /blob_pathname: blob\.pathname/);
  assert.match(uploadSource, /private_blob_persistence/);
  assert.match(uploadSource, /Bound server-side to the configured report workspace/);
  assert.match(uploadSource, /value: "help_scout"/);
  assert.match(uploadSource, /value: "other", label: "Freshdesk \/ other"/);
  assert.doesNotMatch(uploadSource, /value: "freshdesk"/);
  assert.doesNotMatch(uploadSource, /value: "help-scout"/);
  assert.doesNotMatch(uploadSource, /new FormData/);
  assert.match(uploadSource, /JSON\.stringify/);
  assert.doesNotMatch(uploadSource, /X-Atlas-Account-Id|account_id/);
  assert.doesNotMatch(uploadSource, /ATLAS_B2B_JWT|ATLAS_API_BASE_URL|ATLAS_TOKEN/);
  assert.doesNotMatch(uploadSource, /Authorization/);
  assert.doesNotMatch(uploadSource, /\/paid\b/);
});

await test("portfolio submit endpoint pins raw multipart body handling", () => {
  assert.match(submitSource, /bodyParser:\s*false/);
  assert.doesNotMatch(submitSource, /^import .*@vercel\/blob/m);
  assert.match(submitSource, /import\("@vercel\/blob"\)/);
  assert.match(submitSource, /const \{ del \} = await import\("@vercel\/blob"\)/);
});

await test("submit live smoke exercises the production private blob helper", async () => {
  assert.match(submitSmokeSource, /submitPrivateBlob/);
  assert.match(submitSmokeSource, /submitHandler/);
  assert.match(submitSmokeSource, /local_csv_fixture/);
  assert.match(submitSmokeSource, /portfolio_submit_route/);
  assert.doesNotMatch(submitSmokeSource, /ATLAS_B2B_JWT[^\\n]*console\\.log/);
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
  assert.equal(
    submitSmokeValidationErrors({
      baseUrl: ENV.ATLAS_API_BASE_URL,
      token: ENV.ATLAS_B2B_JWT,
      accountId: ACCOUNT_ID,
      blobPathname: `${BLOB_UPLOAD_PATH_PREFIX}tickets.csv`,
      blobToken: ENV.BLOB_READ_WRITE_TOKEN,
      csvFile: "tickets.csv",
      supportPlatform: "zendesk",
      limit: "1000",
      timeoutMs: 1000,
      routeHandler: true,
    }).includes("--route-handler cannot use --csv-file"),
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

await test("portfolio submit endpoint forwards raw multipart bytes to ATLAS", async () => {
  const previousFetch = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push({ url, options });
    return {
      ok: true,
      status: 200,
      async text() {
        return JSON.stringify({ request_id: "content-ops-abc123", status: "completed" });
      },
    };
  };
  const res = mockResponse();
  try {
    await withEnv(ENV, async () => {
      await submitHandler(request(), res);
    });
  } finally {
    globalThis.fetch = previousFetch;
  }

  assert.equal(res.statusCode, 200);
  assert.equal(res.headers["Cache-Control"], "no-store");
  assert.deepEqual(JSON.parse(res.body), {
    ok: true,
    request_id: "content-ops-abc123",
    account_id: ACCOUNT_ID,
    result_path: "/services/faq-deflection/results/content-ops-abc123",
  });
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, `${ENV.ATLAS_API_BASE_URL}${SUBMIT_PATH}`);
  assert.equal(calls[0].options.method, "POST");
  assert.equal(calls[0].options.headers.Authorization, `Bearer ${ENV.ATLAS_B2B_JWT}`);
  assert.equal(calls[0].options.headers["Content-Type"], "multipart/form-data; boundary=atlas");
  assert.equal(Buffer.compare(calls[0].options.body, MULTIPART_BODY), 0);
  assert.equal(calls[0].options.signal instanceof AbortSignal, true);
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
  assert.deepEqual(ok.options.allowedContentTypes, CSV_CONTENT_TYPES);
  assert.equal(ok.options.addRandomSuffix, true);

  assert.deepEqual(
    uploadTokenConfig("other/tickets.csv", JSON.stringify({ account_id: ACCOUNT_ID }), ENV),
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
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, `${ENV.ATLAS_API_BASE_URL}${SUBMIT_PATH}`);
  assert.equal(calls[0].options.headers.Authorization, `Bearer ${ENV.ATLAS_B2B_JWT}`);
  assert.equal("Content-Type" in calls[0].options.headers, false);
  assert.equal(calls[0].options.body instanceof FormData, true);
  assert.equal(calls[0].options.body.get("support_platform"), "zendesk");
  assert.equal(calls[0].options.body.get("company_name"), "Acme Co.");
  assert.equal(await calls[0].options.body.get("csv_file").text(), csv);
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
    fetchImpl: async () => ({
      ok: false,
      status: 502,
      async text() {
        return JSON.stringify({ error: "upstream failed" });
      },
    }),
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
    fetchImpl: async () => ({
      ok: true,
      status: 200,
      async text() {
        return JSON.stringify({ request_id: "content-ops-cleanup123" });
      },
    }),
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
        request({ headers: { "x-atlas-account-id": "3b2b950d-f64b-4852-bc30-f92a34cdf169" } }),
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

await test("portfolio submit endpoint rejects oversize bodies before ATLAS", async () => {
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
            "content-length": String(MAX_CSV_BYTES + MAX_MULTIPART_OVERHEAD_BYTES + 1),
          },
        }),
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
    error: "deflection_submit_csv_too_large",
  });
});

await test("portfolio submit endpoint hides missing config details", async () => {
  const res = mockResponse();
  await withEnv({ ATLAS_API_BASE_URL: "", ATLAS_B2B_JWT: "", ATLAS_ACCOUNT_ID: "" }, async () => {
    await submitHandler(request(), res);
  });
  assert.equal(res.statusCode, 503);
  assert.deepEqual(JSON.parse(res.body), {
    ok: false,
    error: "atlas_submit_not_configured",
  });
  assert.equal(res.body.includes("ATLAS_B2B_JWT"), false);
});

await test("portfolio submit endpoint only accepts POST multipart or JSON blob requests", async () => {
  const getRes = mockResponse();
  await submitHandler(request({ method: "GET" }), getRes);
  assert.equal(getRes.statusCode, 405);
  assert.equal(getRes.headers.Allow, "POST");
  assert.deepEqual(JSON.parse(getRes.body), {
    ok: false,
    error: "method_not_allowed",
  });

  const textRes = mockResponse();
  await submitHandler(request({ headers: { "content-type": "text/plain" } }), textRes);
  assert.equal(textRes.statusCode, 415);
  assert.deepEqual(JSON.parse(textRes.body), {
    ok: false,
    error: "multipart_required",
  });
});

await test("upload shell test is enrolled in package scripts", () => {
  assert.equal(packageJson.dependencies["@vercel/blob"], "^2.4.0");
  assert.equal(
    packageJson.scripts["test:deflection-upload-shell"],
    "node scripts/faq-deflection-upload-shell.test.mjs",
  );
  assert.equal(
    packageJson.scripts["smoke:deflection-submit-live"],
    "node scripts/faq-deflection-submit-live-smoke.mjs",
  );
});
