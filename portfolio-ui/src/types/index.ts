/** A portfolio project / case study */
export interface Project {
  slug: string;
  title: string;
  tagline: string;
  description: string;
  techStack: string[];
  highlights: ProjectHighlight[];
  stats: ProjectStat[];
  media: MediaItem[];
  subsystems: ProjectSubsystem[];
  repo?: string;
  liveUrl?: string;
}

export interface ProjectSubsystem {
  name: string;
  description: string;
  icon: string;
  stats?: ProjectStat[];
  /** Slug of a related insight post */
  relatedInsight?: string;
}

export interface ProjectHighlight {
  title: string;
  description: string;
  icon: string; // lucide icon name
}

export interface ProjectStat {
  label: string;
  value: string;
}

/** Media (gifs, videos, screenshots) */
export interface MediaItem {
  type: "gif" | "video" | "screenshot";
  src: string;
  poster?: string; // thumbnail for videos
  alt: string;
  caption?: string;
  context?: string;
}

/** Insight post (blog / case study / build log) */
export interface InsightPost {
  slug: string;
  title: string;
  description: string;
  date: string;
  type: "case-study" | "build-log" | "industry-insight" | "lesson";
  tags: string[];
  /** Related project slug (atlas, finetunelab) */
  project?: string;
  /** HTML content */
  content: string;
  /** SEO fields */
  seoTitle?: string;
  seoDescription?: string;
  targetKeyword?: string;
  secondaryKeywords?: string[];
  faq?: FaqItem[];
  media?: MediaItem[];
}

export interface FaqItem {
  question: string;
  answer: string;
}

/** AI dev skill tier for the framework */
export interface SkillTier {
  level: number;
  title: string;
  subtitle: string;
  traditional: string[];
  aiAugmented: string[];
  keySkill: string;
}

/** SEO metadata for a page */
export interface PageMeta {
  title: string;
  description: string;
  ogImage?: string;
  ogType?: string;
  /** ISO date string for Open Graph article metadata */
  publishedTime?: string;
  /** ISO date string for Open Graph article metadata */
  modifiedTime?: string;
  /** Optional content section for article schema and OG */
  section?: string;
  twitterImage?: string;
  noindex?: boolean;
  canonicalPath?: string;
  keywords?: string[];
  jsonLd?: Record<string, unknown> | Array<Record<string, unknown>>;
}

/** Navigation link */
export interface NavLink {
  label: string;
  href: string;
  external?: boolean;
}
