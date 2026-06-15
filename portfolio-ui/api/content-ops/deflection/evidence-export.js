import { clean, loadDeflectionReport, proxyErrorPublicPayload } from "./atlas-report.js";

const EVIDENCE_EXPORT_SCHEMA_VERSION = "deflection_evidence.v1";

function json(res, statusCode, payload) {
  res.statusCode = statusCode;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.setHeader("Cache-Control", "no-store");
  res.end(JSON.stringify(payload));
}

function requestUrl(req) {
  const proto = clean(req.headers?.["x-forwarded-proto"]) || "https";
  const host = clean(req.headers?.["x-forwarded-host"]) || clean(req.headers?.host) || "localhost";
  return new URL(req.url || "/", `${proto}://${host}`);
}

function isObjectRecord(value) {
  return value && typeof value === "object" && !Array.isArray(value);
}

function evidenceExportFromReport(report) {
  if (!report || !report.ok) {
    const payload = proxyErrorPublicPayload(report || {
      ok: false,
      statusCode: 502,
      error: "atlas_report_failed",
    });
    return {
      ok: false,
      statusCode: payload.statusCode || 502,
      payload: {
        ok: false,
        statusCode: payload.statusCode || 502,
        error: payload.error || "atlas_report_failed",
        ...(Array.isArray(payload.details) ? { details: payload.details } : {}),
      },
    };
  }

  if (report.artifact_status === "locked") {
    return {
      ok: false,
      statusCode: 403,
      payload: {
        ok: false,
        statusCode: 403,
        error: "evidence_export_locked",
      },
    };
  }
  if (report.artifact_status !== "unlocked") {
    return {
      ok: false,
      statusCode: 404,
      payload: {
        ok: false,
        statusCode: 404,
        error: "evidence_export_unavailable",
      },
    };
  }

  const evidenceExport = isObjectRecord(report.artifact)
    ? report.artifact.evidence_export
    : null;
  if (
    !isObjectRecord(evidenceExport) ||
    clean(evidenceExport.schema_version) !== EVIDENCE_EXPORT_SCHEMA_VERSION
  ) {
    return {
      ok: false,
      statusCode: 502,
      payload: {
        ok: false,
        statusCode: 502,
        error: "evidence_export_contract_violation",
      },
    };
  }

  return { ok: true, export: evidenceExport };
}

function safeFilenamePart(requestId) {
  return clean(requestId).replace(/[^A-Za-z0-9._-]/g, "_") || "deflection-report";
}

export { EVIDENCE_EXPORT_SCHEMA_VERSION, evidenceExportFromReport };

export default async function handler(req, res) {
  if (req.method && req.method !== "GET") {
    json(res, 405, { ok: false, statusCode: 405, error: "method_not_allowed" });
    return;
  }

  const url = requestUrl(req);
  const requestId = clean(req.query?.request_id) || clean(url.searchParams.get("request_id"));
  const accountId = clean(req.query?.account_id) || clean(url.searchParams.get("account_id"));
  const report = await loadDeflectionReport({ requestId, accountId });
  const projection = evidenceExportFromReport(report);
  if (!projection.ok) {
    json(res, projection.statusCode, projection.payload);
    return;
  }

  const body = `${JSON.stringify(projection.export, null, 2)}\n`;
  res.statusCode = 200;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.setHeader(
    "Content-Disposition",
    `attachment; filename="deflection-evidence-${safeFilenamePart(requestId)}.json"`,
  );
  res.setHeader("Cache-Control", "no-store");
  res.end(body);
}
