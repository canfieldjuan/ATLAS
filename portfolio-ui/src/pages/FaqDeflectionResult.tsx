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
import type { DeflectionResultPageSnapshot } from "@/types/deflectionSnapshot";

const SNAPSHOT_STORAGE_PREFIX = "atlas:deflection:snapshot:";
const CHECKOUT_ENDPOINT = "/api/content-ops/deflection/checkout";
const CHECKOUT_SOURCE = "content_ops_deflection_report";
const LIGHT_REPEAT_TICKET_THRESHOLD = 10;
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

type SnapshotState =
  | { status: "available"; snapshot: DeflectionResultPageSnapshot }
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

function nullableString(value: unknown): string | null | undefined {
  if (value === null) return null;
  if (typeof value !== "string") return undefined;
  return value.trim() || null;
}

function nullableNumber(value: unknown): number | null | undefined {
  if (value === null) return null;
  return finiteNumber(value) ?? undefined;
}

function optionalNullableString(
  record: Record<string, unknown>,
  key: string,
): string | null | undefined {
  return key in record ? nullableString(record[key]) : null;
}

function optionalNullableNumber(
  record: Record<string, unknown>,
  key: string,
): number | null | undefined {
  return key in record ? nullableNumber(record[key]) : null;
}

function parseSnapshot(value: unknown): SnapshotState {
  const leaked = [...collectForbiddenKeys(value)].sort();
  if (leaked.length > 0) {
    return {
      status: "invalid",
      reason: `Snapshot included paid-report fields: ${leaked.join(", ")}`,
    };
  }
  if (
    !isRecord(value) ||
    !isRecord(value.summary) ||
    !Array.isArray(value.top_questions) ||
    !Array.isArray(value.top_blind_spots)
  ) {
    return { status: "invalid", reason: "Snapshot shape did not match the deflection contract." };
  }

  const generated = finiteNumber(value.summary.generated);
  const repeatTicketCount = finiteNumber(value.summary.repeat_ticket_count);
  const draftedAnswerCount = finiteNumber(value.summary.drafted_answer_count);
  const noProvenAnswerCount = finiteNumber(value.summary.no_proven_answer_count);
  const resolutionEvidencePresent = value.summary.support_ticket_resolution_evidence_present;
  const resolutionEvidenceCount = finiteNumber(
    value.summary.support_ticket_resolution_evidence_count,
  );
  const nonRepeatTicketCount = finiteNumber(value.summary.non_repeat_ticket_count);
  const sourceDateStart = optionalNullableString(value.summary, "source_date_start");
  const sourceDateEnd = optionalNullableString(value.summary, "source_date_end");
  const sourceWindowDays = optionalNullableNumber(value.summary, "source_window_days");
  if (
    generated === null ||
    repeatTicketCount === null ||
    draftedAnswerCount === null ||
    noProvenAnswerCount === null ||
    typeof resolutionEvidencePresent !== "boolean" ||
    resolutionEvidenceCount === null ||
    nonRepeatTicketCount === null ||
    sourceDateStart === undefined ||
    sourceDateEnd === undefined ||
    sourceWindowDays === undefined
  ) {
    return {
      status: "invalid",
      reason: "Snapshot summary metrics or resolution evidence were incomplete.",
    };
  }

  const topQuestions = value.top_questions.map((item) => {
    if (!isRecord(item)) return null;
    const rank = finiteNumber(item.rank);
    const ticketCount = finiteNumber(item.ticket_count);
    const weightedFrequency = finiteNumber(item.weighted_frequency);
    const question = typeof item.question === "string" ? item.question.trim() : "";
    const customerWording =
      typeof item.customer_wording === "string" ? item.customer_wording.trim() : "";
    if (
      rank === null ||
      ticketCount === null ||
      weightedFrequency === null ||
      !question ||
      !customerWording
    ) {
      return null;
    }
    return {
      rank,
      question,
      ticket_count: ticketCount,
      weighted_frequency: weightedFrequency,
      customer_wording: customerWording,
    };
  });

  if (topQuestions.some((item) => item === null)) {
    return { status: "invalid", reason: "Snapshot questions were incomplete." };
  }
  const topBlindSpots = value.top_blind_spots.map((item) => {
    if (!isRecord(item)) return null;
    const rank = finiteNumber(item.rank);
    const ticketCount = finiteNumber(item.ticket_count);
    const question = typeof item.question === "string" ? item.question.trim() : "";
    if (rank === null || ticketCount === null || !question) {
      return null;
    }
    return { rank, question, ticket_count: ticketCount };
  });

  if (topBlindSpots.some((item) => item === null)) {
    return { status: "invalid", reason: "Snapshot blind spots were incomplete." };
  }

  return {
    status: "available",
    snapshot: {
      summary: {
        generated,
        repeat_ticket_count: repeatTicketCount,
        drafted_answer_count: draftedAnswerCount,
        no_proven_answer_count: noProvenAnswerCount,
        support_ticket_resolution_evidence_present: resolutionEvidencePresent,
        support_ticket_resolution_evidence_count: resolutionEvidenceCount,
        non_repeat_ticket_count: nonRepeatTicketCount,
        source_date_start: sourceDateStart,
        source_date_end: sourceDateEnd,
        source_window_days: sourceWindowDays,
      },
      top_questions: topQuestions as DeflectionResultPageSnapshot["top_questions"],
      top_blind_spots: topBlindSpots as DeflectionResultPageSnapshot["top_blind_spots"],
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
  const checkoutStatus = searchParams.get("checkout")?.trim() ?? "";
  const [snapshotState, setSnapshotState] = useState<SnapshotState>({ status: "missing" });
  const [checkout, setCheckout] = useState<CheckoutState>({ status: "idle" });
  const canCheckout = Boolean(requestId && checkout.status !== "submitting");

  useEffect(() => {
    setSnapshotState(readStoredSnapshot(requestId));
  }, [requestId]);

  const metricCards = useMemo(() => {
    if (snapshotState.status !== "available") return [];
    const { summary } = snapshotState.snapshot;
    return [
      { label: "Questions found", value: summary.generated },
      { label: "Question-level repeat tickets", value: summary.repeat_ticket_count },
      { label: "Evidence-backed answers", value: summary.drafted_answer_count },
      { label: "Needs support proof", value: summary.no_proven_answer_count },
    ];
  }, [snapshotState]);

  const customerWordingExamples = useMemo(() => {
    if (snapshotState.status !== "available") return [];
    return snapshotState.snapshot.top_questions
      .map((question) => question.customer_wording.trim())
      .filter((phrase) => phrase.length > 0)
      .slice(0, 5);
  }, [snapshotState]);

  const resolutionEvidence = useMemo(() => {
    if (snapshotState.status !== "available") return null;
    const { summary } = snapshotState.snapshot;
    const present = summary.support_ticket_resolution_evidence_present;
    const count = summary.support_ticket_resolution_evidence_count;
    return {
      present,
      label: present ? "Present" : "Absent",
      copy: present
        ? `${formatCount(count)} resolved ticket rows can support publishable answer drafting.`
        : "This export supports a gap list only; publishable answers need agent replies or resolved ticket notes.",
    };
  }, [snapshotState]);

  const repeatVolume = useMemo(() => {
    if (snapshotState.status !== "available") return null;
    const count = snapshotState.snapshot.summary.repeat_ticket_count;
    const light = count < LIGHT_REPEAT_TICKET_THRESHOLD;
    return {
      light,
      label: count > 0 ? `${formatCount(count)} question-level repeat tickets` : "No repeated questions yet",
      copy: light
        ? "This export is light on question-level repeat volume. Review the free snapshot before paying for the full report."
        : "This export has enough question-level repeat volume for a substantial paid report preview.",
    };
  }, [snapshotState]);

  const startCheckout = async () => {
    if (!requestId) return;
    setCheckout({ status: "submitting" });
    try {
      const response = await fetch(CHECKOUT_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ request_id: requestId }),
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

            <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
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
                <div className="sm:col-span-2 lg:col-span-4 rounded-lg border border-surface-700/60 bg-surface-800/40 p-5">
                  <p className="text-sm font-semibold text-white">Snapshot not loaded</p>
                  <p className="mt-2 text-sm leading-6 text-surface-200/75">
                    This page has not received the free snapshot from the submit
                    flow yet. No estimates are displayed until real snapshot
                    values are present.
                  </p>
                </div>
              )}
            </div>

            {repeatVolume && (
              <div
                className="mt-4 flex flex-col gap-3 rounded-lg border border-surface-700/60 bg-surface-800/40 p-5 sm:flex-row sm:items-start sm:justify-between"
                data-atlas-deflection-repeat-volume
                data-repeat-volume-light={repeatVolume.light ? "true" : "false"}
              >
                <div>
                  <p className="text-sm text-surface-200/70">Question-level repeat volume</p>
                  <p
                    className={
                      repeatVolume.light
                        ? "mt-2 text-lg font-semibold text-amber-200"
                        : "mt-2 text-lg font-semibold text-primary-300"
                    }
                  >
                    {repeatVolume.label}
                  </p>
                </div>
                <p className="max-w-xl text-sm leading-6 text-surface-200/75">
                  {repeatVolume.copy}
                </p>
              </div>
            )}

            {resolutionEvidence && (
              <div
                className="mt-4 flex flex-col gap-3 rounded-lg border border-surface-700/60 bg-surface-800/40 p-5 sm:flex-row sm:items-start sm:justify-between"
                data-atlas-deflection-resolution-evidence
                data-resolution-evidence-present={resolutionEvidence.present ? "true" : "false"}
              >
                <div>
                  <p className="text-sm text-surface-200/70">Resolution evidence</p>
                  <p
                    className={
                      resolutionEvidence.present
                        ? "mt-2 text-lg font-semibold text-primary-300"
                        : "mt-2 text-lg font-semibold text-amber-200"
                    }
                  >
                    {resolutionEvidence.label}
                  </p>
                </div>
                <p className="max-w-xl text-sm leading-6 text-surface-200/75">
                  {resolutionEvidence.copy}
                </p>
              </div>
            )}

            {snapshotState.status === "invalid" && (
              <div className="mt-6 flex items-start gap-3 rounded-lg border border-amber-400/30 bg-amber-400/10 p-4 text-sm text-amber-100">
                <AlertTriangle className="mt-0.5 h-5 w-5 flex-none text-amber-300" />
                <p>{snapshotState.reason}</p>
              </div>
            )}

            {snapshotState.status === "available" && (
              <div className="mt-10">
                <h2 className="text-xl font-semibold text-white">
                  Help-desk SEO targeting list
                </h2>
                <p className="mt-2 max-w-3xl text-sm leading-6 text-surface-200/75">
                  Use actual customer phrases from the uploaded tickets for
                  help-center titles, internal-search synonyms, and FAQ wording.
                  No keyword volume, ranking, or traffic promise is implied.
                </p>
                <div className="mt-5 rounded-lg border border-surface-700/60 bg-surface-800/40 p-5">
                  <div className="flex flex-col gap-1 sm:flex-row sm:items-baseline sm:justify-between">
                    <p className="text-sm font-semibold text-white">Customer wording</p>
                    <p className="text-xs font-medium uppercase tracking-[0.14em] text-primary-300">
                      Actual ticket phrases only
                    </p>
                  </div>
                  {customerWordingExamples.length > 0 ? (
                    <ul
                      aria-label="Customer wording examples"
                      className="mt-4 grid gap-2 text-sm leading-6 text-surface-100 sm:grid-cols-2"
                    >
                      {customerWordingExamples.map((phrase, index) => (
                        <li
                          key={`${phrase}-${index}`}
                          className="rounded-md border border-surface-700/60 bg-surface-900/35 px-3 py-2"
                        >
                          {phrase}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="mt-4 text-sm leading-6 text-surface-200/75">
                      No customer wording examples are shown because this
                      snapshot did not include real ticket phrases. No invented
                      SEO terms are displayed.
                    </p>
                  )}
                </div>
                <h3 className="mt-8 text-lg font-semibold text-white">
                  Top repeated questions
                </h3>
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
                {snapshotState.snapshot.top_blind_spots.length > 0 && (
                  <>
                    <h3 className="mt-8 text-lg font-semibold text-white">
                      Unresolved blind spots
                    </h3>
                    <div
                      className="mt-4 divide-y divide-surface-700/60 rounded-lg border border-surface-700/60 bg-surface-800/30"
                      data-atlas-deflection-blind-spots
                    >
                      {snapshotState.snapshot.top_blind_spots.map((item) => (
                        <article key={`${item.rank}-${item.question}`} className="p-5">
                          <div className="flex items-center gap-3 text-xs font-semibold uppercase tracking-[0.16em] text-amber-200">
                            <span>Rank {item.rank}</span>
                            <span>{formatCount(item.ticket_count)} tickets unresolved</span>
                          </div>
                          <h3 className="mt-3 text-lg font-semibold text-white">
                            {item.question}
                          </h3>
                        </article>
                      ))}
                    </div>
                  </>
                )}
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
            </dl>

            <button
              type="button"
              className="mt-6 inline-flex w-full items-center justify-center gap-2 rounded-lg bg-primary-500 px-4 py-3 text-sm font-semibold text-surface-900 transition hover:brightness-110 disabled:cursor-not-allowed disabled:bg-surface-700 disabled:text-surface-200/60"
              data-atlas-deflection-unlock
              data-checkout-source={CHECKOUT_SOURCE}
              data-checkout-request_id={requestId ?? ""}
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
