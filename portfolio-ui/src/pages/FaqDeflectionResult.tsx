import { useEffect, useMemo, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import {
  AlertTriangle,
  ArrowLeft,
  CheckCircle2,
  CreditCard,
  FileLock2,
  Loader2,
} from "lucide-react";
import { SeoHead } from "@/components/seo/SeoHead";

const SNAPSHOT_STORAGE_PREFIX = "atlas:deflection:snapshot:";
const CHECKOUT_ENDPOINT = "/api/content-ops/deflection/checkout";
const CHECKOUT_SOURCE = "content_ops_deflection_report";
const FORBIDDEN_SNAPSHOT_KEYS = new Set([
  "answer",
  "answers",
  "evidence",
  "evidence_quotes",
  "faq_result",
  "full_report",
  "markdown",
  "source_id",
  "source_ids",
  "steps",
  "term_mappings",
]);

type DeflectionSnapshot = {
  summary: {
    generated: number;
    drafted_answer_count: number;
    no_proven_answer_count: number;
  };
  top_questions: Array<{
    rank: number;
    question: string;
    weighted_frequency: number;
    customer_wording: string;
  }>;
};

type SnapshotState =
  | { status: "available"; snapshot: DeflectionSnapshot }
  | { status: "missing" }
  | { status: "invalid"; reason: string };

type CheckoutState =
  | { status: "idle" }
  | { status: "submitting" }
  | { status: "error"; message: string };

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function collectForbiddenKeys(value: unknown, leaked = new Set<string>()) {
  if (Array.isArray(value)) {
    for (const item of value) collectForbiddenKeys(item, leaked);
    return leaked;
  }
  if (!isRecord(value)) return leaked;
  for (const [key, child] of Object.entries(value)) {
    if (FORBIDDEN_SNAPSHOT_KEYS.has(key)) leaked.add(key);
    collectForbiddenKeys(child, leaked);
  }
  return leaked;
}

function finiteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function parseSnapshot(value: unknown): SnapshotState {
  const leaked = [...collectForbiddenKeys(value)].sort();
  if (leaked.length > 0) {
    return {
      status: "invalid",
      reason: `Snapshot included paid-report fields: ${leaked.join(", ")}`,
    };
  }
  if (!isRecord(value) || !isRecord(value.summary) || !Array.isArray(value.top_questions)) {
    return { status: "invalid", reason: "Snapshot shape did not match the deflection contract." };
  }

  const generated = finiteNumber(value.summary.generated);
  const draftedAnswerCount = finiteNumber(value.summary.drafted_answer_count);
  const noProvenAnswerCount = finiteNumber(value.summary.no_proven_answer_count);
  if (
    generated === null ||
    draftedAnswerCount === null ||
    noProvenAnswerCount === null
  ) {
    return { status: "invalid", reason: "Snapshot summary metrics were incomplete." };
  }

  const topQuestions = value.top_questions.map((item) => {
    if (!isRecord(item)) return null;
    const rank = finiteNumber(item.rank);
    const weightedFrequency = finiteNumber(item.weighted_frequency);
    const question = typeof item.question === "string" ? item.question.trim() : "";
    const customerWording =
      typeof item.customer_wording === "string" ? item.customer_wording.trim() : "";
    if (rank === null || weightedFrequency === null || !question || !customerWording) {
      return null;
    }
    return {
      rank,
      question,
      weighted_frequency: weightedFrequency,
      customer_wording: customerWording,
    };
  });

  if (topQuestions.some((item) => item === null)) {
    return { status: "invalid", reason: "Snapshot questions were incomplete." };
  }

  return {
    status: "available",
    snapshot: {
      summary: {
        generated,
        drafted_answer_count: draftedAnswerCount,
        no_proven_answer_count: noProvenAnswerCount,
      },
      top_questions: topQuestions as DeflectionSnapshot["top_questions"],
    },
  };
}

function readStoredSnapshot(requestId: string | undefined): SnapshotState {
  if (!requestId || typeof window === "undefined") return { status: "missing" };
  const raw = window.sessionStorage.getItem(`${SNAPSHOT_STORAGE_PREFIX}${requestId}`);
  if (!raw) return { status: "missing" };
  try {
    return parseSnapshot(JSON.parse(raw));
  } catch {
    return { status: "invalid", reason: "Snapshot JSON could not be parsed." };
  }
}

function formatCount(value: number) {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(value);
}

export default function FaqDeflectionResult() {
  const { requestId } = useParams();
  const [searchParams] = useSearchParams();
  const accountId = searchParams.get("account_id")?.trim() ?? "";
  const checkoutStatus = searchParams.get("checkout")?.trim() ?? "";
  const [snapshotState, setSnapshotState] = useState<SnapshotState>({ status: "missing" });
  const [checkout, setCheckout] = useState<CheckoutState>({ status: "idle" });
  const canCheckout = Boolean(requestId && accountId && checkout.status !== "submitting");

  useEffect(() => {
    setSnapshotState(readStoredSnapshot(requestId));
  }, [requestId]);

  const metricCards = useMemo(() => {
    if (snapshotState.status !== "available") return [];
    const { summary } = snapshotState.snapshot;
    return [
      { label: "Questions found", value: summary.generated },
      { label: "Evidence-backed answers", value: summary.drafted_answer_count },
      { label: "Needs support proof", value: summary.no_proven_answer_count },
    ];
  }, [snapshotState]);

  const startCheckout = async () => {
    if (!requestId || !accountId) return;
    setCheckout({ status: "submitting" });
    try {
      const response = await fetch(CHECKOUT_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ request_id: requestId, account_id: accountId }),
      });
      const payload = (await response.json().catch(() => null)) as { url?: unknown; error?: unknown } | null;
      if (!response.ok || !payload || typeof payload.url !== "string") {
        const message =
          payload && typeof payload.error === "string"
            ? payload.error
            : "Checkout could not be started.";
        setCheckout({ status: "error", message });
        return;
      }
      window.location.assign(payload.url);
    } catch {
      setCheckout({ status: "error", message: "Checkout could not be reached." });
    }
  };

  return (
    <>
      <SeoHead
        meta={{
          title: "FAQ Deflection Report",
          description:
            "Review the locked FAQ deflection report snapshot and unlock the full customer-support opportunity report.",
          canonicalPath: requestId
            ? `/services/faq-deflection/results/${encodeURIComponent(requestId)}`
            : "/services/faq-deflection/results",
          noindex: true,
        }}
      />

      <section
        className="mx-auto max-w-6xl px-6 py-10 md:py-14"
        data-atlas-deflection-result
        data-atlas-deflection-request-id={requestId ?? ""}
        data-atlas-deflection-account-id={accountId}
        data-atlas-deflection-report-source={CHECKOUT_SOURCE}
      >
        <Link
          to="/services"
          className="mb-8 inline-flex items-center gap-2 text-sm font-medium text-surface-200 transition-colors hover:text-white"
        >
          <ArrowLeft size={16} />
          Services
        </Link>

        <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_360px]">
          <div>
            <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-primary-500/30 bg-primary-500/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-primary-300">
              <FileLock2 size={14} />
              Locked report
            </div>
            <h1 className="max-w-3xl text-4xl font-bold tracking-normal text-white md:text-5xl">
              FAQ deflection report is ready for review
            </h1>
            <p className="mt-5 max-w-3xl text-lg leading-8 text-surface-200/85">
              The free snapshot shows only repeated customer questions and
              summary counts. The full report unlocks after Stripe confirms the
              one-time payment and ATLAS releases the artifact from the signed
              webhook path.
            </p>

            {checkoutStatus === "success" && (
              <div className="mt-6 flex items-start gap-3 rounded-lg border border-primary-500/30 bg-primary-500/10 p-4 text-sm text-primary-100">
                <CheckCircle2 className="mt-0.5 h-5 w-5 flex-none text-primary-300" />
                <p>
                  Checkout returned successfully. ATLAS unlocks the paid report
                  only after Stripe sends the verified webhook.
                </p>
              </div>
            )}

            <div className="mt-8 grid gap-4 sm:grid-cols-3">
              {metricCards.length > 0 ? (
                metricCards.map((metric) => (
                  <div
                    key={metric.label}
                    className="rounded-lg border border-surface-700/60 bg-surface-800/40 p-5"
                  >
                    <dt className="text-sm text-surface-200/70">{metric.label}</dt>
                    <dd className="mt-3 text-3xl font-semibold text-white">
                      {formatCount(metric.value)}
                    </dd>
                  </div>
                ))
              ) : (
                <div className="sm:col-span-3 rounded-lg border border-surface-700/60 bg-surface-800/40 p-5">
                  <p className="text-sm font-semibold text-white">Snapshot not loaded</p>
                  <p className="mt-2 text-sm leading-6 text-surface-200/75">
                    This page has not received the free snapshot from the submit
                    flow yet. No estimates are displayed until real snapshot
                    values are present.
                  </p>
                </div>
              )}
            </div>

            {snapshotState.status === "invalid" && (
              <div className="mt-6 flex items-start gap-3 rounded-lg border border-amber-400/30 bg-amber-400/10 p-4 text-sm text-amber-100">
                <AlertTriangle className="mt-0.5 h-5 w-5 flex-none text-amber-300" />
                <p>{snapshotState.reason}</p>
              </div>
            )}

            {snapshotState.status === "available" && (
              <div className="mt-10">
                <h2 className="text-xl font-semibold text-white">Top repeated questions</h2>
                <div className="mt-4 divide-y divide-surface-700/60 rounded-lg border border-surface-700/60 bg-surface-800/30">
                  {snapshotState.snapshot.top_questions.map((item) => (
                    <article key={`${item.rank}-${item.question}`} className="p-5">
                      <div className="flex items-center gap-3 text-xs font-semibold uppercase tracking-[0.16em] text-primary-300">
                        <span>Rank {item.rank}</span>
                        <span>{formatCount(item.weighted_frequency)} weighted mentions</span>
                      </div>
                      <h3 className="mt-3 text-lg font-semibold text-white">{item.question}</h3>
                      <p className="mt-2 text-sm leading-6 text-surface-200/75">
                        {item.customer_wording}
                      </p>
                    </article>
                  ))}
                </div>
              </div>
            )}
          </div>

          <aside className="h-fit rounded-lg border border-surface-700/60 bg-surface-800/45 p-6">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary-500/15 text-primary-300">
                <CreditCard size={20} />
              </div>
              <div>
                <p className="text-sm font-semibold text-white">Unlock full report</p>
                <p className="text-sm text-surface-200/70">$1,500 one-time payment</p>
              </div>
            </div>

            <dl className="mt-6 space-y-4 text-sm">
              <div>
                <dt className="text-surface-200/60">Checkout metadata source</dt>
                <dd className="mt-1 break-all font-mono text-xs text-surface-100">
                  {CHECKOUT_SOURCE}
                </dd>
              </div>
              <div>
                <dt className="text-surface-200/60">request_id</dt>
                <dd className="mt-1 break-all font-mono text-xs text-surface-100">
                  {requestId || "Missing request id"}
                </dd>
              </div>
              <div>
                <dt className="text-surface-200/60">account_id</dt>
                <dd className="mt-1 break-all font-mono text-xs text-surface-100">
                  {accountId || "Missing account id"}
                </dd>
              </div>
            </dl>

            <button
              type="button"
              className="mt-6 inline-flex w-full items-center justify-center gap-2 rounded-lg bg-primary-500 px-4 py-3 text-sm font-semibold text-surface-900 transition hover:brightness-110 disabled:cursor-not-allowed disabled:bg-surface-700 disabled:text-surface-200/60"
              data-atlas-deflection-unlock
              data-checkout-source={CHECKOUT_SOURCE}
              data-checkout-request_id={requestId ?? ""}
              data-checkout-account_id={accountId}
              disabled={!canCheckout}
              onClick={startCheckout}
            >
              {checkout.status === "submitting" ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Opening Checkout
                </>
              ) : (
                <>
                  <CreditCard size={16} />
                  Continue to Checkout
                </>
              )}
            </button>

            {!accountId && (
              <p className="mt-3 text-xs leading-5 text-amber-200">
                Missing account_id. Return to the submit flow to create a
                Checkout session for the right ATLAS tenant.
              </p>
            )}
            {checkout.status === "error" && (
              <p className="mt-3 text-xs leading-5 text-amber-200">{checkout.message}</p>
            )}
            <p className="mt-4 text-xs leading-5 text-surface-200/60">
              The portfolio creates Checkout. ATLAS unlocks the artifact only
              after the verified Stripe webhook.
            </p>
          </aside>
        </div>
      </section>
    </>
  );
}
