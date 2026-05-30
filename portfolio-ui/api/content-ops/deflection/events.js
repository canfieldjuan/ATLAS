const EVENT_NAME_RE = /^[a-z][a-z0-9_]{2,100}$/;
const SECRET_FIELD_RE = /(authorization|jwt|secret|token|credential|password|email|csv|body|content)/i;
const SECRET_VALUE_RE = /(authorization|bearer\s+[A-Za-z0-9._~+/=-]+|jwt|secret|token|credential|password|[?&][A-Za-z0-9_.-]*(token|secret|credential|password|jwt)=)/i;
const MAX_FIELD_LENGTH = 240;

function safeEventName(value) {
  return typeof value === "string" && EVENT_NAME_RE.test(value)
    ? value
    : "faq_deflection_server_event";
}

function sanitizeEventValue(value) {
  if (value === null || value === undefined) return undefined;
  if (typeof value === "boolean" || typeof value === "number") return value;
  if (typeof value !== "string") return String(value).slice(0, MAX_FIELD_LENGTH);
  const trimmed = value.trim();
  if (!trimmed) return undefined;
  if (SECRET_VALUE_RE.test(trimmed)) return "[redacted]";
  return trimmed.slice(0, MAX_FIELD_LENGTH);
}

function sanitizeDeflectionEventFields(fields = {}) {
  const safe = {};
  for (const [key, value] of Object.entries(fields)) {
    if (!key || SECRET_FIELD_RE.test(key)) continue;
    const sanitized = sanitizeEventValue(value);
    if (sanitized !== undefined) safe[key] = sanitized;
  }
  return safe;
}

function emitDeflectionServerEvent(eventName, fields = {}, logger = console.warn) {
  const event = safeEventName(eventName);
  const safeFields = sanitizeDeflectionEventFields(fields);
  try {
    logger(event, safeFields);
    return { ok: true, event, fields: safeFields };
  } catch {
    return { ok: false, event, fields: safeFields };
  }
}

export { emitDeflectionServerEvent, sanitizeDeflectionEventFields };
