export { churnIntelligencePipeline } from "./churn-intelligence-pipeline";
export { llmFirstWorkflow } from "./llm-first-workflow";
export { edgeCloudVoicePipeline } from "./edge-cloud-voice-pipeline";
export { autonomousTasksFailSafely } from "./autonomous-tasks-fail-safely";
export { deterministicInfrastructure } from "./deterministic-infrastructure";

import { churnIntelligencePipeline } from "./churn-intelligence-pipeline";
import { llmFirstWorkflow } from "./llm-first-workflow";
import { edgeCloudVoicePipeline } from "./edge-cloud-voice-pipeline";
import { autonomousTasksFailSafely } from "./autonomous-tasks-fail-safely";
import { deterministicInfrastructure } from "./deterministic-infrastructure";
import type { InsightPost } from "@/types";

export const allInsights: InsightPost[] = [
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
