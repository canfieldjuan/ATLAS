import { clean, loadDeflectionReport, proxyErrorPublicPayload } from "./atlas-report.js";

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
  json(res, 200, report);
}
