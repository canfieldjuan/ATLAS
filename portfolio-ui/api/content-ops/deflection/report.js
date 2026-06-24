import { clean, loadDeflectionReport, proxyErrorPublicPayload } from "./atlas-report.js";
import * as reportModelContract from "./report-model-contract.js";

const DEFLECTION_REPORT_SECTION_ID_SET = new Set(reportModelContract.DEFLECTION_REPORT_SECTION_IDS);

function isObjectRecord(value) {
  return value && typeof value === "object" && !Array.isArray(value);
}

function fieldConstToken(field) {
  return String(field).toUpperCase().replace(/[^A-Z0-9]+/g, "_").replace(/^_+|_+$/g, "");
}

function sectionConstPrefix(sectionId) {
  return `DEFLECTION_REPORT_${fieldConstToken(sectionId)}`;
}

function generatedFields(name) {
  const fields = reportModelContract[name];
  return Array.isArray(fields) ? fields : [];
}

function cloneScalar(value) {
  if (
    value === null ||
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return value;
  }
  return undefined;
}

function cloneScalarArray(value) {
  if (!Array.isArray(value)) return undefined;
  const cloned = [];
  for (const item of value) {
    const scalar = cloneScalar(item);
    if (scalar === undefined) return undefined;
    cloned.push(scalar);
  }
  return cloned;
}

function projectHostedFields(record, fields, prefix) {
  if (!isObjectRecord(record)) return {};
  const projected = {};
  for (const field of fields) {
    if (!Object.prototype.hasOwnProperty.call(record, field)) continue;
    const value = record[field];
    const nestedPrefix = `${prefix}_${fieldConstToken(field)}`;
    const nestedFields = generatedFields(`${nestedPrefix}_HOSTED_CONSUMER_SAFE_FIELDS`);
    if (nestedFields.length > 0) {
      if (Array.isArray(value)) {
        projected[field] = value
          .filter(isObjectRecord)
          .map((item) => projectHostedFields(item, nestedFields, nestedPrefix));
      } else if (isObjectRecord(value)) {
        projected[field] = projectHostedFields(value, nestedFields, nestedPrefix);
      }
      continue;
    }

    const scalar = cloneScalar(value);
    if (scalar !== undefined) {
      projected[field] = scalar;
      continue;
    }
    const scalarArray = cloneScalarArray(value);
    if (scalarArray !== undefined) {
      projected[field] = scalarArray;
    }
  }
  return projected;
}

function publicHostedReportModel(model) {
  if (!isObjectRecord(model) || !Array.isArray(model.sections)) return null;
  const sections = [];
  for (const section of model.sections) {
    if (!isObjectRecord(section)) continue;
    const sectionId = clean(section.id);
    if (!DEFLECTION_REPORT_SECTION_ID_SET.has(sectionId)) continue;
    const prefix = sectionConstPrefix(sectionId);
    const hostedFields = generatedFields(`${prefix}_HOSTED_CONSUMER_SAFE_FIELDS`);
    sections.push({
      id: sectionId,
      title: clean(section.title),
      priority: typeof section.priority === "number" && Number.isFinite(section.priority)
        ? section.priority
        : 0,
      surfaces: cloneScalarArray(section.surfaces) || [],
      default_limit: typeof section.default_limit === "number" && Number.isFinite(section.default_limit)
        ? section.default_limit
        : null,
      required_data: cloneScalarArray(section.required_data) || [],
      snapshot_safe_fields: cloneScalarArray(section.snapshot_safe_fields) || [],
      data: projectHostedFields(section.data, hostedFields, prefix),
    });
  }
  return {
    schema_version: clean(model.schema_version),
    title: clean(model.title),
    summary: projectHostedFields(model.summary, Object.keys(model.summary || {}), "DEFLECTION_REPORT_SUMMARY"),
    sections,
  };
}

function publicReportPayload(report) {
  const payload = {
    ok: true,
    request_id: clean(report.request_id),
    snapshot: report.snapshot,
    artifact_status: clean(report.artifact_status),
  };
  if (payload.artifact_status !== "unlocked") return payload;

  const reportModel = publicHostedReportModel(report.artifact?.report_model);
  if (reportModel) {
    payload.artifact = { report_model: reportModel };
  }
  return payload;
}

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

export default async function handler(req, res) {
  if (req.method !== "GET") {
    res.setHeader("Allow", "GET");
    json(res, 405, { error: "method_not_allowed" });
    return;
  }

  const url = requestUrl(req);
  const report = await loadDeflectionReport({
    requestId: clean(req.query?.request_id) || clean(url.searchParams.get("request_id")),
    accountId: clean(req.query?.account_id) || clean(url.searchParams.get("account_id")),
  });
  if (!report.ok) {
    json(res, report.statusCode || 502, proxyErrorPublicPayload(report));
    return;
  }
  json(res, 200, publicReportPayload(report));
}

export { publicHostedReportModel, publicReportPayload };
