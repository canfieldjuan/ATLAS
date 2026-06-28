import {
  DEFLECTION_SNAPSHOT_SUMMARY_FIELDS,
  DEFLECTION_SNAPSHOT_SUMMARY_OPTIONAL_FIELDS,
  DEFLECTION_SNAPSHOT_TOP_BLIND_SPOT_FIELDS,
  DEFLECTION_SNAPSHOT_TOP_QUESTION_FIELDS,
} from "./snapshot-contract.js";

const REQUEST_ID_RE = /^[A-Za-z0-9][A-Za-z0-9_.:-]{5,160}$/;
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const DEFAULT_ATLAS_TIMEOUT_MS = 5000;
const INVALID_FIELD = Symbol("invalid-field");

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

function requiredNumber(value) {
  const number = finiteNumber(value);
  return number === null ? INVALID_FIELD : number;
}

function requiredString(value) {
  const string = cleanString(value);
  return string ? string : INVALID_FIELD;
}

function requiredBoolean(value) {
  return typeof value === "boolean" ? value : INVALID_FIELD;
}

function optionalNullableString(value) {
  if (value === undefined || value === null) return null;
  if (typeof value !== "string") return INVALID_FIELD;
  return value.trim() || null;
}

function optionalNullableNumber(value) {
  if (value === undefined || value === null) return null;
  const number = finiteNumber(value);
  return number === null ? INVALID_FIELD : number;
}

const SUMMARY_FIELD_READERS = {
  generated: requiredNumber,
  drafted_answer_count: requiredNumber,
  no_proven_answer_count: requiredNumber,
  support_ticket_resolution_evidence_present: requiredBoolean,
  support_ticket_resolution_evidence_count: requiredNumber,
  repeat_ticket_count: requiredNumber,
  non_repeat_ticket_count: requiredNumber,
  source_date_start: optionalNullableString,
  source_date_end: optionalNullableString,
  source_window_days: optionalNullableNumber,
};

const QUESTION_FIELD_READERS = {
  rank: requiredNumber,
  question: requiredString,
  ticket_count: requiredNumber,
  weighted_frequency: requiredNumber,
  customer_wording: requiredString,
  owner_lane: requiredString,
  action_label: requiredString,
  estimated_support_cost: requiredNumber,
};

const BLIND_SPOT_FIELD_READERS = {
  rank: requiredNumber,
  question: requiredString,
  ticket_count: requiredNumber,
  owner_lane: requiredString,
  action_label: requiredString,
  estimated_support_cost: requiredNumber,
};

function projectFields(record, fields, readers, errorMessage, errors) {
  const projected = {};
  for (const field of fields) {
    const reader = readers[field];
    if (!reader) {
      errors.push(`snapshot contract field has no runtime reader: ${field}`);
      return null;
    }
    const value = reader(record[field]);
    if (value === INVALID_FIELD) {
      errors.push(errorMessage);
      return null;
    }
    projected[field] = value;
  }
  return projected;
}

function assertContractReaders(fields, readers, errors, label) {
  for (const field of fields) {
    if (!readers[field]) {
      errors.push(`snapshot ${label} contract field has no runtime reader: ${field}`);
    }
  }
}

function projectSnapshot(snapshot) {
  const errors = [];
  if (!snapshot || typeof snapshot !== "object" || Array.isArray(snapshot)) {
    return { ok: false, errors: ["snapshot response must be an object"] };
  }
  if (
    !snapshot.summary ||
    typeof snapshot.summary !== "object" ||
    Array.isArray(snapshot.summary)
  ) {
    errors.push("snapshot.summary must be an object");
  }
  if (!Array.isArray(snapshot.top_questions)) {
    errors.push("snapshot.top_questions must be an array");
  }
  if (!Array.isArray(snapshot.top_blind_spots)) {
    errors.push("snapshot.top_blind_spots must be an array");
  }
  if (errors.length > 0) return { ok: false, errors };

  assertContractReaders(
    DEFLECTION_SNAPSHOT_SUMMARY_FIELDS,
    SUMMARY_FIELD_READERS,
    errors,
    "summary",
  );
  assertContractReaders(
    DEFLECTION_SNAPSHOT_TOP_QUESTION_FIELDS,
    QUESTION_FIELD_READERS,
    errors,
    "top_questions",
  );
  assertContractReaders(
    DEFLECTION_SNAPSHOT_TOP_BLIND_SPOT_FIELDS,
    BLIND_SPOT_FIELD_READERS,
    errors,
    "top_blind_spots",
  );
  const optionalSummaryFields = new Set(DEFLECTION_SNAPSHOT_SUMMARY_OPTIONAL_FIELDS);
  const summary = projectFields(
    snapshot.summary,
    DEFLECTION_SNAPSHOT_SUMMARY_FIELDS,
    SUMMARY_FIELD_READERS,
    "snapshot.summary metrics must include finite counts and resolution evidence",
    errors,
  );

  const topQuestions = snapshot.top_questions.map((item) => {
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      errors.push("snapshot.top_questions entries must be objects");
      return null;
    }
    return projectFields(
      item,
      DEFLECTION_SNAPSHOT_TOP_QUESTION_FIELDS,
      QUESTION_FIELD_READERS,
      "snapshot.top_questions entries must include safe projected fields",
      errors,
    );
  });
  const topBlindSpots = snapshot.top_blind_spots.map((item) => {
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      errors.push("snapshot.top_blind_spots entries must be objects");
      return null;
    }
    return projectFields(
      item,
      DEFLECTION_SNAPSHOT_TOP_BLIND_SPOT_FIELDS,
      BLIND_SPOT_FIELD_READERS,
      "snapshot.top_blind_spots entries must include safe projected fields",
      errors,
    );
  });
  if (errors.length > 0) return { ok: false, errors };

  return {
    ok: true,
    snapshot: {
      summary: Object.fromEntries(
        Object.entries(summary).filter(
          ([field, value]) => value !== null || optionalSummaryFields.has(field),
        ),
      ),
      top_questions: topQuestions,
      top_blind_spots: topBlindSpots,
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
