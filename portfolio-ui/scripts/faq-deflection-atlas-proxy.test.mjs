import assert from "node:assert/strict";
import reportHandler from "../api/content-ops/deflection/report.js";
import {
  atlasPath,
  fetchAtlasJson,
  loadDeflectionReport,
  projectSnapshot,
  snapshotErrors,
} from "../api/content-ops/deflection/atlas-report.js";
import { renderResultPage } from "../api/content-ops/deflection/result-page.js";

const ACCOUNT_ID = "2b2b950d-f64b-4852-bc30-f92a34cdf169";
const REQUEST_ID = "content-ops-abc123";
const ENV = {
  ATLAS_API_BASE_URL: "https://atlas.example.com",
  ATLAS_B2B_JWT: "secret-service-token",
  ATLAS_ACCOUNT_ID: ACCOUNT_ID,
};

const SNAPSHOT = {
  summary: {
    generated: 3,
    drafted_answer_count: 2,
    no_proven_answer_count: 1,
  },
  top_questions: [
    {
      rank: 1,
      question: "How do I reset billing access?",
      weighted_frequency: 12,
      customer_wording: "billing reset access",
    },
  ],
};

async function test(name, fn) {
  try {
    await fn();
    console.log(`ok - ${name}`);
  } catch (error) {
    console.error(`not ok - ${name}`);
    throw error;
  }
}

function response(payload, status = 200) {
  return {
    status,
    async text() {
      return JSON.stringify(payload);
    },
  };
}

function mockFetch(results) {
  const calls = [];
  const fetchImpl = async (url, options = {}) => {
    calls.push({ url, options });
    const next = results.shift();
    if (!next) throw new Error(`unexpected fetch: ${url}`);
    return next;
  };
  return { calls, fetchImpl };
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

function withEnv(nextEnv, fn) {
  const previous = {};
  for (const key of Object.keys(nextEnv)) {
    previous[key] = process.env[key];
    process.env[key] = nextEnv[key];
  }
  return Promise.resolve()
    .then(fn)
    .finally(() => {
      for (const key of Object.keys(nextEnv)) {
        if (previous[key] === undefined) {
          delete process.env[key];
        } else {
          process.env[key] = previous[key];
        }
      }
    });
}

await test("proxy rejects account mismatch before calling ATLAS", async () => {
  const { calls, fetchImpl } = mockFetch([]);
  const result = await loadDeflectionReport({
    requestId: REQUEST_ID,
    accountId: "3b2b950d-f64b-4852-bc30-f92a34cdf169",
    env: ENV,
    fetchImpl,
  });
  assert.equal(result.ok, false);
  assert.equal(result.statusCode, 400);
  assert.deepEqual(result.details, ["account_id does not match the configured ATLAS account"]);
  assert.equal(calls.length, 0);
});

await test("proxy returns locked snapshot envelope without artifact payload", async () => {
  const leakySnapshot = {
    ...SNAPSHOT,
    answer_steps: ["paid step"],
    faq_markdown: "# paid",
    top_questions: [{ ...SNAPSHOT.top_questions[0], source_ids: ["ticket-1"] }],
  };
  const { calls, fetchImpl } = mockFetch([
    response(leakySnapshot, 200),
    response({ detail: "payment required" }, 403),
  ]);
  const result = await loadDeflectionReport({
    requestId: REQUEST_ID,
    accountId: ACCOUNT_ID,
    env: ENV,
    fetchImpl,
  });
  assert.equal(result.ok, true);
  assert.equal(result.artifact_status, "locked");
  assert.deepEqual(result.snapshot, SNAPSHOT);
  assert.equal(JSON.stringify(result).includes("answer_steps"), false);
  assert.equal(JSON.stringify(result).includes("faq_markdown"), false);
  assert.equal(JSON.stringify(result).includes("source_ids"), false);
  assert.equal("artifact" in result, false);
  assert.equal(calls[0].url, `${ENV.ATLAS_API_BASE_URL}${atlasPath(REQUEST_ID, "snapshot")}`);
  assert.equal(calls[1].url, `${ENV.ATLAS_API_BASE_URL}${atlasPath(REQUEST_ID, "artifact")}`);
  assert.equal(calls[0].options.headers.Authorization, `Bearer ${ENV.ATLAS_B2B_JWT}`);
});

await test("proxy returns unlocked artifact only after ATLAS returns 200", async () => {
  const artifact = { markdown: "# Paid report", summary: { generated: 3 } };
  const { fetchImpl } = mockFetch([response(SNAPSHOT, 200), response(artifact, 200)]);
  const result = await loadDeflectionReport({
    requestId: REQUEST_ID,
    accountId: ACCOUNT_ID,
    env: ENV,
    fetchImpl,
  });
  assert.equal(result.ok, true);
  assert.equal(result.artifact_status, "unlocked");
  assert.deepEqual(result.artifact, artifact);
});

await test("snapshot projection only returns known safe fields", async () => {
  const result = projectSnapshot({
    ...SNAPSHOT,
    markdown: "# paid",
    answer_steps: ["paid step"],
    top_questions: [{ ...SNAPSHOT.top_questions[0], faq_markdown: "# paid" }],
  });
  assert.equal(result.ok, true);
  assert.deepEqual(result.snapshot, SNAPSHOT);
  assert.equal(JSON.stringify(result).includes("markdown"), false);
  assert.equal(JSON.stringify(result).includes("answer_steps"), false);
  assert.deepEqual(snapshotErrors({ summary: {}, top_questions: [] }), [
    "snapshot.summary metrics must be finite numbers",
  ]);
});

await test("ATLAS fetches carry abort signals and timeout failures return errors", async () => {
  const fetchImpl = (url, options = {}) =>
    new Promise((resolve) => {
      assert.equal(url, `${ENV.ATLAS_API_BASE_URL}${atlasPath(REQUEST_ID, "snapshot")}`);
      assert.equal(options.signal instanceof AbortSignal, true);
      options.signal.addEventListener("abort", () => {
        resolve({
          status: 599,
          async text() {
            throw Object.assign(new Error("aborted"), { name: "AbortError" });
          },
        });
      });
    });
  const result = await fetchAtlasJson({
    config: { ...ENV, baseUrl: ENV.ATLAS_API_BASE_URL, token: ENV.ATLAS_B2B_JWT, timeoutMs: 1 },
    path: atlasPath(REQUEST_ID, "snapshot"),
    fetchImpl,
  });
  assert.equal(result.networkError, "AbortError");
});

await test("unresponsive ATLAS returns graceful snapshot failure", async () => {
  const fetchImpl = async (url, options = {}) => {
    assert.equal(options.signal instanceof AbortSignal, true);
    throw Object.assign(new Error(`timeout ${url}`), { name: "AbortError" });
  };
  const result = await loadDeflectionReport({
    requestId: REQUEST_ID,
    accountId: ACCOUNT_ID,
    env: { ...ENV, ATLAS_PROXY_TIMEOUT_MS: "1" },
    fetchImpl,
  });
  assert.equal(result.ok, false);
  assert.equal(result.statusCode, 502);
  assert.equal(result.error, "atlas_snapshot_failed");
});

await test("public report API hides config details from browser responses", async () => {
  await withEnv({ ATLAS_API_BASE_URL: "", ATLAS_B2B_JWT: "", ATLAS_ACCOUNT_ID: "" }, async () => {
    const req = {
      method: "GET",
      headers: { host: "portfolio.example.com", "x-forwarded-proto": "https" },
      query: { request_id: REQUEST_ID, account_id: ACCOUNT_ID },
      url: `/api/content-ops/deflection/report?request_id=${REQUEST_ID}&account_id=${ACCOUNT_ID}`,
    };
    const res = mockResponse();
    await reportHandler(req, res);
    assert.equal(res.statusCode, 503);
    const payload = JSON.parse(res.body);
    assert.deepEqual(payload, {
      ok: false,
      statusCode: 503,
      error: "atlas_proxy_not_configured",
    });
    assert.equal(JSON.stringify(payload).includes("ATLAS_B2B_JWT"), false);
  });
});

await test("public report API returns sanitized locked envelope and never the token", async () => {
  const previousFetch = globalThis.fetch;
  const { fetchImpl } = mockFetch([response(SNAPSHOT, 200), response({}, 403)]);
  globalThis.fetch = fetchImpl;
  await withEnv(ENV, async () => {
    const req = {
      method: "GET",
      headers: { host: "portfolio.example.com", "x-forwarded-proto": "https" },
      query: { request_id: REQUEST_ID, account_id: ACCOUNT_ID },
      url: `/api/content-ops/deflection/report?request_id=${REQUEST_ID}&account_id=${ACCOUNT_ID}`,
    };
    const res = mockResponse();
    try {
      await reportHandler(req, res);
    } finally {
      globalThis.fetch = previousFetch;
    }
    assert.equal(res.statusCode, 200);
    assert.equal(res.headers["Cache-Control"], "no-store");
    const payload = JSON.parse(res.body);
    assert.equal(payload.artifact_status, "locked");
    assert.equal(JSON.stringify(payload).includes(ENV.ATLAS_B2B_JWT), false);
  });
});

await test("hosted result page renders real snapshot metrics from the proxy envelope", () => {
  const html = renderResultPage({
    requestId: REQUEST_ID,
    accountId: ACCOUNT_ID,
    report: {
      ok: true,
      snapshot: SNAPSHOT,
      artifact_status: "locked",
    },
  });
  assert.match(html, /Questions found/);
  assert.match(html, />3</);
  assert.match(html, /How do I reset billing access\?/);
  assert.match(html, /artifact_status/);
  assert.match(html, /locked/);
  assert.doesNotMatch(html, /# Paid report/);
});
