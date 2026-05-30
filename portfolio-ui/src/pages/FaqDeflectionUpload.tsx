import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import {
  AlertTriangle,
  ArrowLeft,
  CheckCircle2,
  FileSpreadsheet,
  LockKeyhole,
  Upload,
} from "lucide-react";
import { SeoHead } from "@/components/seo/SeoHead";

const SUBMIT_ENDPOINT = "/api/content-ops/deflection/submit";
const MAX_CSV_BYTES = 50 * 1024 * 1024;
const SUPPORT_PLATFORMS = [
  { value: "zendesk", label: "Zendesk" },
  { value: "intercom", label: "Intercom" },
  { value: "help-scout", label: "Help Scout" },
  { value: "freshdesk", label: "Freshdesk" },
] as const;

type UploadState =
  | { status: "empty" }
  | { status: "ready"; fileName: string; fileSize: number }
  | { status: "invalid"; message: string };

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
  return { status: "ready", fileName: file.name, fileSize: file.size };
}

export default function FaqDeflectionUpload() {
  const [companyName, setCompanyName] = useState("");
  const [contactEmail, setContactEmail] = useState("");
  const [accountId, setAccountId] = useState("");
  const [supportPlatform, setSupportPlatform] = useState<string>(SUPPORT_PLATFORMS[0].value);
  const [upload, setUpload] = useState<UploadState>({ status: "empty" });

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
            onSubmit={(event) => event.preventDefault()}
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
                50 MB cap. CSV bytes stay on the portfolio side until the server
                submit contract is enabled.
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
              className="mt-6 inline-flex w-full items-center justify-center gap-2 rounded-lg bg-surface-700 px-4 py-3 text-sm font-semibold text-surface-200/65"
              data-atlas-deflection-submit-guard
              disabled
              aria-disabled="true"
            >
              <LockKeyhole size={16} />
              Submit pending backend contract
            </button>
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
                <dt className="text-surface-200/60">Current response</dt>
                <dd className="mt-1 font-mono text-xs text-surface-100">
                  deflection_submit_backend_pending
                </dd>
              </div>
            </dl>
            <p className="mt-5 text-sm leading-6 text-surface-200/70">
              The live handoff remains closed until the ATLAS multipart submit
              contract is merged on main.
            </p>
          </aside>
        </div>
      </section>
    </>
  );
}
