import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { upload as uploadBlob } from "@vercel/blob/client";
import {
  AlertTriangle,
  ArrowLeft,
  CheckCircle2,
  FileSpreadsheet,
  Loader2,
  Send,
  Upload,
} from "lucide-react";
import { SeoHead } from "@/components/seo/SeoHead";

const SUBMIT_ENDPOINT = "/api/content-ops/deflection/submit";
const BLOB_UPLOAD_ENDPOINT = "/api/content-ops/deflection/upload";
const BLOB_UPLOAD_PATH_PREFIX = "faq-deflection/uploads/";
const MAX_CSV_BYTES = 50 * 1024 * 1024;
const SUPPORT_PLATFORMS = [
  { value: "zendesk", label: "Zendesk" },
  { value: "intercom", label: "Intercom" },
  { value: "help_scout", label: "Help Scout" },
  { value: "other", label: "Freshdesk / other" },
] as const;

type UploadState =
  | { status: "empty" }
  | { status: "ready"; file: File; fileName: string; fileSize: number }
  | { status: "invalid"; message: string };

type SubmitState =
  | { status: "idle" }
  | { status: "uploading" }
  | { status: "submitting" }
  | { status: "error"; message: string };

function formatBytes(bytes: number) {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${bytes} bytes`;
}

function fileState(file: File | undefined): UploadState {
  if (!file) return { status: "empty" };
  const lowerName = file.name.toLowerCase();
  if (!lowerName.endsWith(".csv")) {
    return { status: "invalid", message: "Select a CSV export." };
  }
  if (file.size <= 0) {
    return { status: "invalid", message: "The selected CSV is empty." };
  }
  if (file.size > MAX_CSV_BYTES) {
    return { status: "invalid", message: "CSV exports must be 50 MB or smaller." };
  }
  return { status: "ready", file, fileName: file.name, fileSize: file.size };
}

function blobPathname(fileName: string) {
  const cleaned = fileName
    .toLowerCase()
    .replace(/[^a-z0-9_.-]+/g, "-")
    .replace(/^-+|-+$/g, "");
  const csvName = cleaned.endsWith(".csv") ? cleaned : "tickets.csv";
  return `${BLOB_UPLOAD_PATH_PREFIX}${Date.now()}-${csvName}`;
}

export default function FaqDeflectionUpload() {
  const [companyName, setCompanyName] = useState("");
  const [contactEmail, setContactEmail] = useState("");
  const [accountId, setAccountId] = useState("");
  const [supportPlatform, setSupportPlatform] = useState<string>(SUPPORT_PLATFORMS[0].value);
  const [upload, setUpload] = useState<UploadState>({ status: "empty" });
  const [submit, setSubmit] = useState<SubmitState>({ status: "idle" });

  const fieldsReady = useMemo(
    () =>
      Boolean(
        companyName.trim() &&
          contactEmail.trim() &&
          accountId.trim() &&
          supportPlatform &&
          upload.status === "ready",
      ),
    [accountId, companyName, contactEmail, supportPlatform, upload.status],
  );
  const canSubmit = fieldsReady && submit.status !== "uploading" && submit.status !== "submitting";

  const startSubmit = async () => {
    if (upload.status !== "ready" || !fieldsReady) return;
    const trimmedAccountId = accountId.trim();

    try {
      setSubmit({ status: "uploading" });
      const blob = await uploadBlob(blobPathname(upload.fileName), upload.file, {
        access: "private",
        contentType: "text/csv",
        handleUploadUrl: BLOB_UPLOAD_ENDPOINT,
        clientPayload: JSON.stringify({ account_id: trimmedAccountId }),
        multipart: upload.fileSize > 4 * 1024 * 1024,
      });

      setSubmit({ status: "submitting" });
      const response = await fetch(SUBMIT_ENDPOINT, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Atlas-Account-Id": trimmedAccountId,
        },
        body: JSON.stringify({
          blob_pathname: blob.pathname,
          support_platform: supportPlatform,
          company_name: companyName.trim(),
          contact_email: contactEmail.trim(),
          limit: "1000",
        }),
      });
      const payload = (await response.json().catch(() => null)) as {
        result_path?: unknown;
        error?: unknown;
      } | null;
      if (!response.ok || !payload || typeof payload.result_path !== "string") {
        const message =
          payload && typeof payload.error === "string"
            ? payload.error
            : "FAQ deflection submit could not be started.";
        setSubmit({ status: "error", message });
        return;
      }
      window.location.assign(payload.result_path);
    } catch {
      setSubmit({ status: "error", message: "FAQ deflection upload could not be completed." });
    }
  };

  return (
    <>
      <SeoHead
        meta={{
          title: "FAQ Deflection Upload",
          description:
            "Prepare a support-ticket CSV for the FAQ deflection report handoff.",
          canonicalPath: "/services/faq-deflection",
          noindex: true,
        }}
      />

      <section
        className="mx-auto max-w-6xl px-6 py-10 md:py-14"
        data-atlas-deflection-upload
        data-atlas-deflection-submit-endpoint={SUBMIT_ENDPOINT}
        data-atlas-deflection-upload-endpoint={BLOB_UPLOAD_ENDPOINT}
      >
        <Link
          to="/services"
          className="mb-8 inline-flex items-center gap-2 text-sm font-medium text-surface-200 transition-colors hover:text-white"
        >
          <ArrowLeft size={16} />
          Services
        </Link>

        <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_340px]">
          <form
            className="rounded-lg border border-surface-700/60 bg-surface-800/35 p-6"
            onSubmit={(event) => {
              event.preventDefault();
              void startSubmit();
            }}
          >
            <div className="flex flex-wrap items-center gap-3">
              <div className="flex h-11 w-11 items-center justify-center rounded-lg bg-primary-500/15 text-primary-300">
                <FileSpreadsheet size={22} />
              </div>
              <div>
                <p className="text-sm font-semibold uppercase tracking-[0.16em] text-primary-300">
                  FAQ deflection
                </p>
                <h1 className="text-3xl font-bold tracking-normal text-white md:text-4xl">
                  Support-ticket CSV upload
                </h1>
              </div>
            </div>

            <div className="mt-8 grid gap-5 md:grid-cols-2">
              <label className="block">
                <span className="text-sm font-medium text-surface-100">Company</span>
                <input
                  className="mt-2 w-full rounded-lg border border-surface-700 bg-surface-900 px-3 py-3 text-sm text-white outline-none transition focus:border-primary-400"
                  data-atlas-deflection-company
                  value={companyName}
                  onChange={(event) => setCompanyName(event.target.value)}
                  placeholder="Acme Co."
                />
              </label>

              <label className="block">
                <span className="text-sm font-medium text-surface-100">Contact email</span>
                <input
                  className="mt-2 w-full rounded-lg border border-surface-700 bg-surface-900 px-3 py-3 text-sm text-white outline-none transition focus:border-primary-400"
                  data-atlas-deflection-contact-email
                  type="email"
                  value={contactEmail}
                  onChange={(event) => setContactEmail(event.target.value)}
                  placeholder="lead@example.com"
                />
              </label>

              <label className="block">
                <span className="text-sm font-medium text-surface-100">Support platform</span>
                <select
                  className="mt-2 w-full rounded-lg border border-surface-700 bg-surface-900 px-3 py-3 text-sm text-white outline-none transition focus:border-primary-400"
                  data-atlas-deflection-support-platform
                  value={supportPlatform}
                  onChange={(event) => setSupportPlatform(event.target.value)}
                >
                  {SUPPORT_PLATFORMS.map((platform) => (
                    <option key={platform.value} value={platform.value}>
                      {platform.label}
                    </option>
                  ))}
                </select>
              </label>

              <label className="block">
                <span className="text-sm font-medium text-surface-100">ATLAS account ID</span>
                <input
                  className="mt-2 w-full rounded-lg border border-surface-700 bg-surface-900 px-3 py-3 font-mono text-xs text-white outline-none transition focus:border-primary-400"
                  data-atlas-deflection-account-id-input
                  value={accountId}
                  onChange={(event) => setAccountId(event.target.value)}
                  placeholder="2b2b950d-f64b-4852-bc30-f92a34cdf169"
                />
              </label>
            </div>

            <label className="mt-6 block rounded-lg border border-dashed border-surface-600 bg-surface-900/55 p-5 transition focus-within:border-primary-400">
              <span className="flex items-center gap-3 text-sm font-medium text-surface-100">
                <Upload size={18} className="text-primary-300" />
                CSV export
              </span>
              <input
                className="mt-4 block w-full cursor-pointer rounded-lg border border-surface-700 bg-surface-900 text-sm text-surface-100 file:mr-4 file:border-0 file:bg-primary-500 file:px-4 file:py-3 file:text-sm file:font-semibold file:text-surface-900"
                data-atlas-deflection-csv-file
                type="file"
                accept=".csv,text/csv"
                onChange={(event) => setUpload(fileState(event.target.files?.[0]))}
              />
              <span className="mt-3 block text-xs text-surface-200/60">
                50 MB cap. CSV bytes are first stored in private Vercel Blob,
                then forwarded to ATLAS server-side; service tokens never reach
                the browser.
              </span>
            </label>

            <div className="mt-5 min-h-12">
              {upload.status === "ready" && (
                <div className="flex items-start gap-3 rounded-lg border border-primary-500/30 bg-primary-500/10 p-4 text-sm text-primary-100">
                  <CheckCircle2 className="mt-0.5 h-5 w-5 flex-none text-primary-300" />
                  <p>
                    {upload.fileName} selected ({formatBytes(upload.fileSize)}).
                  </p>
                </div>
              )}
              {upload.status === "invalid" && (
                <div className="flex items-start gap-3 rounded-lg border border-amber-400/30 bg-amber-400/10 p-4 text-sm text-amber-100">
                  <AlertTriangle className="mt-0.5 h-5 w-5 flex-none text-amber-300" />
                  <p>{upload.message}</p>
                </div>
              )}
            </div>

            <button
              type="submit"
              className="mt-6 inline-flex w-full items-center justify-center gap-2 rounded-lg bg-primary-500 px-4 py-3 text-sm font-semibold text-surface-900 transition hover:brightness-110 disabled:cursor-not-allowed disabled:bg-surface-700 disabled:text-surface-200/65"
              data-atlas-deflection-submit
              disabled={!canSubmit}
            >
              {submit.status === "uploading" ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Uploading CSV
                </>
              ) : submit.status === "submitting" ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Creating report
                </>
              ) : (
                <>
                  <Send size={16} />
                  Create locked report
                </>
              )}
            </button>
            {submit.status === "error" && (
              <p className="mt-3 text-xs leading-5 text-amber-200">{submit.message}</p>
            )}
          </form>

          <aside className="h-fit rounded-lg border border-surface-700/60 bg-surface-800/45 p-6">
            <h2 className="text-lg font-semibold text-white">Handoff state</h2>
            <dl className="mt-5 space-y-4 text-sm">
              <div>
                <dt className="text-surface-200/60">Upload fields</dt>
                <dd className="mt-1 text-surface-100">
                  {fieldsReady ? "Ready" : "Waiting for required values"}
                </dd>
              </div>
              <div>
                <dt className="text-surface-200/60">Server endpoint</dt>
                <dd className="mt-1 break-all font-mono text-xs text-surface-100">
                  {SUBMIT_ENDPOINT}
                </dd>
              </div>
              <div>
                <dt className="text-surface-200/60">Submit mode</dt>
                <dd className="mt-1 font-mono text-xs text-surface-100">
                  private_blob_persistence
                </dd>
              </div>
            </dl>
            <p className="mt-5 text-sm leading-6 text-surface-200/70">
              The portfolio persists the CSV to private Blob, reads it back on
              the server, then sends the browser to the hosted result page for
              snapshot hydration and Checkout.
            </p>
          </aside>
        </div>
      </section>
    </>
  );
}
