export { atlasProject } from "./atlas";
export { finetunelabProject } from "./finetunelab";

import { atlasProject } from "./atlas";
import { finetunelabProject } from "./finetunelab";
import type { Project } from "@/types";

export const allProjects: Project[] = [atlasProject, finetunelabProject];

export function getProject(slug: string): Project | undefined {
  return allProjects.find((p) => p.slug === slug);
}
