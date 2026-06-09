const REQUEST_ID_RE = /^[A-Za-z0-9][A-Za-z0-9_.:-]{5,160}$/;
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const DEFAULT_ATLAS_TIMEOUT_MS = 5000;

function clean(value) {
  if (Array.isArray(value)) return clean(value[0]);
  return typeof value === "string" ? value.trim() : "";
}

function configFromEnv(env = process.env) {
  const timeoutMs = Number.parseInt(clean(env.ATLAS_PROXY_TIMEOUT_MS), 10);
  return {
    baseUrl: clean(env.ATLAS_API_BASE_URL),
    token: clean(env.ATLAS_B2B_JWT || env.ATLAS_TOKEN),
    accountId: clean(env.ATLAS_ACCOUNT_ID),
    timeoutMs:
      Number.isFinite(timeoutMs) && timeoutMs > 0 ? timeoutMs : DEFAULT_ATLAS_TIMEOUT_MS,
  };
}

function reportRequestErrors({ requestId, accountId }, config) {
  const errors = [];
  if (!requestId || !REQUEST_ID_RE.test(requestId)) {
    errors.push("request_id must be a valid Content Ops request id");
  }
  if (accountId && !UUID_RE.test(accountId)) {
    errors.push("account_id must be a valid ATLAS account UUID");
  }
  if (!config.baseUrl) errors.push("ATLAS_API_BASE_URL is not configured");
  if (!config.token) errors.push("ATLAS_B2B_JWT is not configured");
  if (!config.accountId || !UUID_RE.test(config.accountId)) {
    errors.push("ATLAS_ACCOUNT_ID is not configured");
  } else if (accountId && UUID_RE.test(accountId) && accountId !== config.accountId) {
    errors.push("account_id does not match the configured ATLAS account");
  }
  return errors;
}

function atlasUrl(baseUrl, path) {
  return `${baseUrl.replace(/\/+$/, "")}/${path.replace(/^\/+/, "")}`;
}

function atlasPath(requestId, suffix) {
  return `/api/v1/content-ops/deflection-reports/${encodeURIComponent(requestId)}/${suffix}`;
}

async function fetchAtlasJson({ config, path, fetchImpl = fetch }) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), config.timeoutMs);
  try {
    const response = await fetchImpl(atlasUrl(config.baseUrl, path), {
      headers: {
        "Accept": "application/json",
        "Authorization": `Bearer ${config.token}`,
      },
      signal: controller.signal,
    });
    const text = await response.text();
    let payload = null;
    if (text) {
      try {
        payload = JSON.parse(text);
      } catch {
        return { status: response.status, payload: null, parseError: true };
      }
    }
    return { status: response.status, payload, parseError: false };
  } catch (error) {
    return {
      status: null,
      payload: null,
      parseError: false,
      networkError: error && typeof error === "object" ? error.name || "fetch_error" : "fetch_error",
    };
  } finally {
    clearTimeout(timeout);
  }
}

function finiteNumber(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function cleanString(value) {
  return typeof value === "string" ? value.trim() : "";
}

function projectSnapshot(snapshot) {
  const errors = [];
  if (!snapshot || typeof snapshot !== "object" || Array.isArray(snapshot)) {
    return { ok: false, errors: ["snapshot response must be an object"] };
  }
  if (!snapshot.summary || typeof snapshot.summary !== "object") {
    errors.push("snapshot.summary must be an object");
  }
  if (!Array.isArray(snapshot.top_questions)) {
    errors.push("snapshot.top_questions must be an array");
  }
  if (errors.length > 0) return { ok: false, errors };

  const generated = finiteNumber(snapshot.summary.generated);
  const draftedAnswerCount = finiteNumber(snapshot.summary.drafted_answer_count);
  const noProvenAnswerCount = finiteNumber(snapshot.summary.no_proven_answer_count);
  const resolutionEvidencePresent = snapshot.summary.support_ticket_resolution_evidence_present;
  const resolutionEvidenceCount = finiteNumber(
    snapshot.summary.support_ticket_resolution_evidence_count,
  );
  if (
    generated === null ||
    draftedAnswerCount === null ||
    noProvenAnswerCount === null ||
    typeof resolutionEvidencePresent !== "boolean" ||
    resolutionEvidenceCount === null
  ) {
    errors.push("snapshot.summary metrics must include finite counts and resolution evidence");
  }

  const topQuestions = snapshot.top_questions.map((item) => {
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      errors.push("snapshot.top_questions entries must be objects");
      return null;
    }
    const rank = finiteNumber(item.rank);
    const weightedFrequency = finiteNumber(item.weighted_frequency);
    const question = cleanString(item.question);
    const customerWording = cleanString(item.customer_wording);
    if (rank === null || weightedFrequency === null || !question || !customerWording) {
      errors.push("snapshot.top_questions entries must include safe projected fields");
      return null;
    }
    return {
      rank,
      question,
      weighted_frequency: weightedFrequency,
      customer_wording: customerWording,
    };
  });
  if (errors.length > 0) return { ok: false, errors };

  return {
    ok: true,
    snapshot: {
      summary: {
        generated,
        drafted_answer_count: draftedAnswerCount,
        no_proven_answer_count: noProvenAnswerCount,
        support_ticket_resolution_evidence_present: resolutionEvidencePresent,
        support_ticket_resolution_evidence_count: resolutionEvidenceCount,
      },
      top_questions: topQuestions,
    },
  };
}

function snapshotErrors(snapshot) {
  const projection = projectSnapshot(snapshot);
  return projection.ok ? [] : projection.errors;
}

function snapshotFailed(snapshot) {
  return snapshot.status !== 200 || snapshot.parseError || snapshot.networkError;
}

function artifactFailed(artifact) {
  return artifact.parseError || artifact.networkError;
}

function proxyErrorPublicPayload(report) {
  if (report.error === "atlas_proxy_not_configured") {
    return {
      ok: false,
      statusCode: report.statusCode,
      error: report.error,
    };
  }
  return report;
}

async function loadDeflectionReport({
  requestId,
  accountId,
  env = process.env,
  fetchImpl = fetch,
}) {
  const config = configFromEnv(env);
  const cleanedRequestId = clean(requestId);
  const cleanedAccountId = clean(accountId) || config.accountId;
  const errors = reportRequestErrors(
    { requestId: cleanedRequestId, accountId: cleanedAccountId },
    config,
  );
  if (errors.length > 0) {
    const configError = errors.some((error) => error.endsWith("is not configured"));
    return {
      ok: false,
      statusCode: configError ? 503 : 400,
      error: configError ? "atlas_proxy_not_configured" : "invalid_report_request",
      details: errors,
    };
  }

  const snapshot = await fetchAtlasJson({
    config,
    path: atlasPath(cleanedRequestId, "snapshot"),
    fetchImpl,
  });
  if (snapshot.status === 404) {
    return {
      ok: false,
      statusCode: 404,
      error: "report_not_found",
      request_id: cleanedRequestId,
      account_id: cleanedAccountId,
    };
  }
  if (snapshotFailed(snapshot)) {
    return {
      ok: false,
      statusCode: 502,
      error: "atlas_snapshot_failed",
      request_id: cleanedRequestId,
      account_id: cleanedAccountId,
    };
  }

  const projectedSnapshot = projectSnapshot(snapshot.payload);
  if (!projectedSnapshot.ok) {
    return {
      ok: false,
      statusCode: 502,
      error: "atlas_snapshot_contract_violation",
      details: projectedSnapshot.errors,
      request_id: cleanedRequestId,
      account_id: cleanedAccountId,
    };
  }

  const artifact = await fetchAtlasJson({
    config,
    path: atlasPath(cleanedRequestId, "artifact"),
    fetchImpl,
  });
  if (artifact.status === 200 && !artifactFailed(artifact)) {
    return {
      ok: true,
      request_id: cleanedRequestId,
      account_id: cleanedAccountId,
      snapshot: projectedSnapshot.snapshot,
      artifact_status: "unlocked",
      artifact: artifact.payload,
    };
  }
  if (artifact.status === 403) {
    return {
      ok: true,
      request_id: cleanedRequestId,
      account_id: cleanedAccountId,
      snapshot: projectedSnapshot.snapshot,
      artifact_status: "locked",
    };
  }
  if (artifact.status === 404) {
    return {
      ok: true,
      request_id: cleanedRequestId,
      account_id: cleanedAccountId,
      snapshot: projectedSnapshot.snapshot,
      artifact_status: "missing",
    };
  }
  return {
    ok: false,
    statusCode: 502,
    error: "atlas_artifact_failed",
    request_id: cleanedRequestId,
    account_id: cleanedAccountId,
  };
}

export {
  atlasPath,
  atlasUrl,
  clean,
  configFromEnv,
  fetchAtlasJson,
  loadDeflectionReport,
  projectSnapshot,
  proxyErrorPublicPayload,
  reportRequestErrors,
  snapshotErrors,
};
