import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import submitHandler, {
  SUBMIT_PENDING_ERROR,
} from "../api/content-ops/deflection/submit.js";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const appSource = await readFile(resolve(root, "src/App.tsx"), "utf8");
const servicesSource = await readFile(resolve(root, "src/pages/Services.tsx"), "utf8");
const uploadSource = await readFile(resolve(root, "src/pages/FaqDeflectionUpload.tsx"), "utf8");
const submitSource = await readFile(resolve(root, "api/content-ops/deflection/submit.js"), "utf8");
const packageJson = JSON.parse(await readFile(resolve(root, "package.json"), "utf8"));

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

await test("upload shell route is wired into the portfolio app and services page", () => {
  assert.match(appSource, /FaqDeflectionUpload/);
  assert.match(appSource, /\/services\/faq-deflection/);
  assert.match(servicesSource, /\/services\/faq-deflection/);
  assert.match(servicesSource, /FAQ Deflection Upload/);
});

await test("upload shell exposes stable validation and submit guard markers", () => {
  for (const marker of [
    "data-atlas-deflection-upload",
    "data-atlas-deflection-csv-file",
    "data-atlas-deflection-company",
    "data-atlas-deflection-contact-email",
    "data-atlas-deflection-support-platform",
    "data-atlas-deflection-account-id-input",
    "data-atlas-deflection-submit-guard",
  ]) {
    assert.match(uploadSource, new RegExp(marker));
  }
  assert.match(uploadSource, /MAX_CSV_BYTES = 50 \* 1024 \* 1024/);
  assert.match(uploadSource, /Submit pending backend contract/);
});

await test("upload shell and submit guard do not expose ATLAS service credentials", () => {
  for (const source of [uploadSource, submitSource]) {
    assert.doesNotMatch(source, /ATLAS_B2B_JWT|ATLAS_API_BASE_URL|ATLAS_TOKEN/);
    assert.doesNotMatch(source, /Authorization/);
    assert.doesNotMatch(source, /\/paid\b/);
  }
});

await test("portfolio submit endpoint fails closed until live forwarding lands", async () => {
  const previousFetch = globalThis.fetch;
  let fetchCalled = false;
  globalThis.fetch = async () => {
    fetchCalled = true;
    throw new Error("submit endpoint must not call ATLAS while guarded");
  };
  const res = mockResponse();
  try {
    await submitHandler({ method: "POST" }, res);
  } finally {
    globalThis.fetch = previousFetch;
  }

  assert.equal(fetchCalled, false);
  assert.equal(res.statusCode, 503);
  assert.equal(res.headers["Cache-Control"], "no-store");
  assert.deepEqual(JSON.parse(res.body), {
    ok: false,
    error: SUBMIT_PENDING_ERROR,
  });
});

await test("portfolio submit endpoint only accepts POST", async () => {
  const res = mockResponse();
  await submitHandler({ method: "GET" }, res);
  assert.equal(res.statusCode, 405);
  assert.equal(res.headers.Allow, "POST");
  assert.deepEqual(JSON.parse(res.body), {
    ok: false,
    error: "method_not_allowed",
  });
});

await test("upload shell test is enrolled in package scripts", () => {
  assert.equal(
    packageJson.scripts["test:deflection-upload-shell"],
    "node scripts/faq-deflection-upload-shell.test.mjs",
  );
});
