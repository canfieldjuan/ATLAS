import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import submitHandler, {
  MAX_CSV_BYTES,
  MAX_MULTIPART_OVERHEAD_BYTES,
  SUBMIT_PATH,
  forwardSubmit,
} from "../api/content-ops/deflection/submit.js";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const appSource = await readFile(resolve(root, "src/App.tsx"), "utf8");
const servicesSource = await readFile(resolve(root, "src/pages/Services.tsx"), "utf8");
const uploadSource = await readFile(resolve(root, "src/pages/FaqDeflectionUpload.tsx"), "utf8");
const submitSource = await readFile(resolve(root, "api/content-ops/deflection/submit.js"), "utf8");
const packageJson = JSON.parse(await readFile(resolve(root, "package.json"), "utf8"));

const ACCOUNT_ID = "2b2b950d-f64b-4852-bc30-f92a34cdf169";
const ENV = {
  ATLAS_API_BASE_URL: "https://atlas.example.com",
  ATLAS_B2B_JWT: "secret-service-token",
  ATLAS_ACCOUNT_ID: ACCOUNT_ID,
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
    "data-atlas-deflection-account-id-input",
    "data-atlas-deflection-submit",
  ]) {
    assert.match(uploadSource, new RegExp(marker));
  }
  assert.match(uploadSource, /MAX_CSV_BYTES = 4 \* 1024 \* 1024/);
  assert.match(uploadSource, /value: "help_scout"/);
  assert.match(uploadSource, /value: "other", label: "Freshdesk \/ other"/);
  assert.doesNotMatch(uploadSource, /value: "freshdesk"/);
  assert.doesNotMatch(uploadSource, /value: "help-scout"/);
  assert.match(uploadSource, /new FormData/);
  assert.match(uploadSource, /X-Atlas-Account-Id/);
  assert.doesNotMatch(uploadSource, /ATLAS_B2B_JWT|ATLAS_API_BASE_URL|ATLAS_TOKEN/);
  assert.doesNotMatch(uploadSource, /Authorization/);
  assert.doesNotMatch(uploadSource, /\/paid\b/);
});

await test("portfolio submit endpoint pins raw multipart body handling", () => {
  assert.match(submitSource, /bodyParser:\s*false/);
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
    result_path: `/services/faq-deflection/results/content-ops-abc123?account_id=${ACCOUNT_ID}`,
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

await test("portfolio submit endpoint only accepts POST multipart requests", async () => {
  const getRes = mockResponse();
  await submitHandler(request({ method: "GET" }), getRes);
  assert.equal(getRes.statusCode, 405);
  assert.equal(getRes.headers.Allow, "POST");
  assert.deepEqual(JSON.parse(getRes.body), {
    ok: false,
    error: "method_not_allowed",
  });

  const jsonRes = mockResponse();
  await submitHandler(request({ headers: { "content-type": "application/json" } }), jsonRes);
  assert.equal(jsonRes.statusCode, 415);
  assert.deepEqual(JSON.parse(jsonRes.body), {
    ok: false,
    error: "multipart_required",
  });
});

await test("upload shell test is enrolled in package scripts", () => {
  assert.equal(
    packageJson.scripts["test:deflection-upload-shell"],
    "node scripts/faq-deflection-upload-shell.test.mjs",
  );
});
