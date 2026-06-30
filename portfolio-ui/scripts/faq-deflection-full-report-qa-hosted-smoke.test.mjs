import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { evidenceExportFromReport } from "../api/content-ops/deflection/evidence-export.js";
import {
  RESULT_PAGE_QA_SURFACE_CAPS,
  renderResultPage,
  resultPageQaObservation,
} from "../api/content-ops/deflection/result-page.js";

const portfolioRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repoRoot = resolve(portfolioRoot, "..");
const checkerPath = resolve(repoRoot, "scripts/check_deflection_full_report_hosted_smoke.py");
const accountId = "2b2b950d-f64b-4852-bc30-f92a34cdf169";
const requestId = "content-ops-hosted-qa";

const reportModel = {
  schema_version: "deflection.v1",
  title: "Resolution Audit",
  summary: { generated: 2 },
  sections: [
    {
      id: "support_tax",
      data: {
        repeat_ticket_count: 8,
        non_repeat_ticket_count: 0,
        generated_question_count: 2,
        assisted_contact_cost: 13.5,
        estimated_support_cost: 108,
        source_date_window: {},
        drafted_answer_count: 1,
        no_proven_answer_count: 1,
        ticket_source_count: 8,
      },
    },
    {
      id: "seo_targets",
      data: {
        phrases: ["export attribution reports", "report download", "SSO setup"],
        total_phrase_count: 3,
        displayed_phrase_count: 3,
        omitted_phrase_count: 0,
        limit: 50,
      },
    },
    {
      id: "ranked_questions",
      data: {
        rows: [{ rank: 1 }, { rank: 2 }],
      },
    },
    {
      id: "priority_fix_queue",
      data: {
        items: [],
        status_counts: {},
        result_page_limit: 3,
        pdf_limit: 10,
        backlog_limit: 25,
        support_cost_basis: "assisted_contact_cost",
      },
    },
    {
      id: "top_unresolved_repeats",
      data: {
        items: [],
        top_item_count: 0,
        result_page_limit: 3,
        pdf_limit: 10,
        support_cost_basis: "assisted_contact_cost",
      },
    },
    {
      id: "drafted_resolutions",
      data: {
        items: [],
        top_item_count: 0,
        result_page_limit: 3,
        pdf_limit: 10,
      },
    },
    {
      id: "already_covered_still_recurring",
      data: {
        items: [],
        top_item_count: 0,
        result_page_limit: 3,
        pdf_limit: 10,
      },
    },
    {
      id: "suppressed_repeat_review_queue",
      data: {
        items: [],
        total_item_count: 0,
        default_limit: 25,
        reason_counts: {},
      },
    },
    {
      id: "question_details",
      data: {
        rows: [{ rank: 1 }, { rank: 2 }],
      },
    },
    {
      id: "complete_evidence",
      data: {
        question_count: 2,
        evidence_row_count: 8,
        source_id_count: 8,
        surfaces: ["export"],
      },
    },
  ],
};

const evidenceExport = {
  schema_version: "deflection_evidence.v1",
  summary: {
    question_count: 2,
    evidence_row_count: 8,
    source_id_count: 8,
    drafted_answer_count: 1,
    no_proven_answer_count: 1,
  },
  questions: [
    { rank: 1, question: "How do I export attribution reports?" },
    { rank: 2, question: "Can I enable SSO for every user?" },
  ],
  evidence_rows: Array.from({ length: 8 }, (_, index) => ({ row: index + 1 })),
};

const paidArtifact = {
  summary: {
    generated: 2,
    repeat_ticket_count: 8,
  },
  faq_result: {
    items: [
      {
        question: "How do I export attribution reports?",
        customer_wording: "export attribution reports",
        ticket_count: 5,
        opportunity_score: 21,
        answer_evidence_status: "resolution_evidence",
        answer: "Open Analytics, choose Attribution, then export the report.",
        term_mappings: [{ customer_term: "report download" }],
      },
      {
        question: "Can I enable SSO for every user?",
        customer_wording: "SSO setup",
        ticket_count: 3,
        opportunity_score: 13,
        answer_evidence_status: "draft_needs_review",
      },
    ],
  },
  report_model: reportModel,
  evidence_export: evidenceExport,
};

async function test(name, fn) {
  try {
    await fn();
    console.log(`ok - ${name}`);
  } catch (error) {
    console.error(`not ok - ${name}`);
    throw error;
  }
}

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function writeJson(path, payload) {
  writeFileSync(path, `${JSON.stringify(payload)}\n`, "utf8");
}

function decodeHtmlAttribute(value) {
  return value
    .replace(/&quot;/g, "\"")
    .replace(/&#39;/g, "'")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&amp;/g, "&");
}

function extractQaObservation(html) {
  const match = html.match(/data-atlas-deflection-qa-observation="([^"]+)"/);
  assert.ok(match, "paid result page must expose a sanitized QA observation");
  return JSON.parse(decodeHtmlAttribute(match[1]));
}

function evidenceExportObservation(payload) {
  return {
    counts: {
      evidence_question_count: payload.summary?.question_count,
      evidence_row_count: payload.summary?.evidence_row_count,
      source_id_count: payload.summary?.source_id_count,
      drafted_answer_count: payload.summary?.drafted_answer_count,
      no_proven_answer_count: payload.summary?.no_proven_answer_count,
    },
  };
}

function runChecker({
  observations,
  model = reportModel,
  exportPayload = evidenceExport,
  expectedStatus = 0,
}) {
  const dir = mkdtempSync(join(tmpdir(), "atlas-hosted-qa-"));
  try {
    const modelPath = join(dir, "report_model.json");
    const exportPath = join(dir, "evidence_export.json");
    const observationsPath = join(dir, "observations.json");
    const capsPath = join(dir, "caps.json");
    const outputPath = join(dir, "scorecard.json");
    writeJson(modelPath, model);
    writeJson(exportPath, exportPayload);
    writeJson(observationsPath, observations);
    writeJson(capsPath, { result_page: RESULT_PAGE_QA_SURFACE_CAPS });

    const result = spawnSync(process.env.PYTHON || "python", [
      checkerPath,
      "--report-model",
      modelPath,
      "--evidence-export",
      exportPath,
      "--surface-observations",
      observationsPath,
      "--surface-caps",
      capsPath,
      "--output",
      outputPath,
    ], {
      cwd: repoRoot,
      encoding: "utf8",
    });
    assert.equal(
      result.status,
      expectedStatus,
      `expected checker exit ${expectedStatus}, got ${result.status}\nstdout:\n${result.stdout}\nstderr:\n${result.stderr}`,
    );
    return JSON.parse(readFileSync(outputPath, "utf8"));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

function failedAssertionIds(scorecard) {
  return new Set(
    scorecard.assertions
      .filter((assertion) => assertion.ok === false)
      .map((assertion) => assertion.id),
  );
}

await test("hosted result page and evidence export satisfy the QA scorecard", () => {
  const report = {
    ok: true,
    artifact_status: "unlocked",
    artifact: paidArtifact,
  };
  const html = renderResultPage({
    requestId,
    accountId,
    report,
  });
  const renderedObservation = extractQaObservation(html);
  assert.deepEqual(renderedObservation, resultPageQaObservation(report));

  const projection = evidenceExportFromReport(report);
  assert.equal(projection.ok, true);
  assert.deepEqual(projection.export, evidenceExport);

  const observations = {
    result_page: renderedObservation,
    evidence_export: evidenceExportObservation(projection.export),
  };
  const scorecard = runChecker({ observations });

  assert.equal(scorecard.ok, true);
  assert.deepEqual(scorecard.surfaces.required, ["result_page", "evidence_export"]);
  assert.match(html, /data-atlas-deflection-evidence-export-download/);
  assert.doesNotMatch(JSON.stringify(scorecard), /request_id|result_url|cs_|pi_|@/);
});

await test("hosted smoke fails when the result page omits a required metric", () => {
  const report = {
    ok: true,
    artifact_status: "unlocked",
    artifact: paidArtifact,
  };
  const observations = {
    result_page: resultPageQaObservation(report),
    evidence_export: evidenceExportObservation(evidenceExport),
  };
  delete observations.result_page.counts.source_id_count;

  const scorecard = runChecker({ observations, expectedStatus: 1 });
  assert.equal(scorecard.ok, false);
  assert.ok(
    failedAssertionIds(scorecard).has("harness.surface.result_page.count.source_id_count.present"),
  );
});

await test("hosted smoke fails when rendered counts disagree with the model", () => {
  const report = {
    ok: true,
    artifact_status: "unlocked",
    artifact: paidArtifact,
  };
  const observations = {
    result_page: clone(resultPageQaObservation(report)),
    evidence_export: evidenceExportObservation(evidenceExport),
  };
  observations.result_page.counts.repeat_ticket_count = 7;

  const scorecard = runChecker({ observations, expectedStatus: 1 });
  assert.equal(scorecard.ok, false);
  assert.ok(
    failedAssertionIds(scorecard).has("surface.result_page.count.repeat_ticket_count"),
  );
});
