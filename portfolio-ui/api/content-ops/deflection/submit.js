const SUBMIT_PENDING_ERROR = "deflection_submit_backend_pending";

function json(res, statusCode, payload) {
  res.statusCode = statusCode;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.setHeader("Cache-Control", "no-store");
  res.end(JSON.stringify(payload));
}

export { SUBMIT_PENDING_ERROR };

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    json(res, 405, { ok: false, error: "method_not_allowed" });
    return;
  }

  json(res, 503, {
    ok: false,
    error: SUBMIT_PENDING_ERROR,
  });
}
