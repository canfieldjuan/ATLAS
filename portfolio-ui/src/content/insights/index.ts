export { churnIntelligencePipeline } from "./churn-intelligence-pipeline";
export { llmFirstWorkflow } from "./llm-first-workflow";
export { edgeCloudVoicePipeline } from "./edge-cloud-voice-pipeline";
export { autonomousTasksFailSafely } from "./autonomous-tasks-fail-safely";
export { deterministicDeepDive } from "./seven-patterns-deterministic-llm";
export { gpuFailureSilentFallback } from "./gpu-failure-silent-fallback";
export { deterministicInfrastructure } from "./deterministic-infrastructure";
export { lessonCloudVsLocalCost } from "./lesson-cloud-vs-local-cost";
export { lessonTestingLlmSystems } from "./lesson-testing-llm-systems";
export { lessonAutonomyOverrated } from "./lesson-autonomy-overrated";
export { lessonPromptingRagGrounding } from "./lesson-prompting-rag-grounding";
export { lessonModelContextNotModelProblem } from "./lesson-model-context-not-model-problem";
export { lessonBuildVsBuyTax } from "./lesson-build-vs-buy-tax";

import { churnIntelligencePipeline } from "./churn-intelligence-pipeline";
import { llmFirstWorkflow } from "./llm-first-workflow";
import { edgeCloudVoicePipeline } from "./edge-cloud-voice-pipeline";
import { autonomousTasksFailSafely } from "./autonomous-tasks-fail-safely";
import { deterministicDeepDive } from "./seven-patterns-deterministic-llm";
import { gpuFailureSilentFallback } from "./gpu-failure-silent-fallback";
import { deterministicInfrastructure } from "./deterministic-infrastructure";
import { lessonCloudVsLocalCost } from "./lesson-cloud-vs-local-cost";
import { lessonTestingLlmSystems } from "./lesson-testing-llm-systems";
import { lessonAutonomyOverrated } from "./lesson-autonomy-overrated";
import { lessonPromptingRagGrounding } from "./lesson-prompting-rag-grounding";
import { lessonModelContextNotModelProblem } from "./lesson-model-context-not-model-problem";
import { lessonBuildVsBuyTax } from "./lesson-build-vs-buy-tax";
import type { InsightPost } from "@/types";

export const allInsights: InsightPost[] = [
  lessonBuildVsBuyTax,
  lessonModelContextNotModelProblem,
  lessonPromptingRagGrounding,
  lessonAutonomyOverrated,
  lessonCloudVsLocalCost,
  lessonTestingLlmSystems,
  deterministicDeepDive,
  gpuFailureSilentFallback,
  deterministicInfrastructure,
  churnIntelligencePipeline,
  autonomousTasksFailSafely,
  edgeCloudVoicePipeline,
  llmFirstWorkflow,
];

export function getInsight(slug: string): InsightPost | undefined {
  return allInsights.find((p) => p.slug === slug);
}

export function getInsightsByType(
  type: InsightPost["type"],
): InsightPost[] {
  return allInsights.filter((p) => p.type === type);
}
