import { useEffect } from "react";
import type { PageMeta } from "@/types";

const SITE_NAME = "Juan Canfield — AI Systems Architect";
const DEFAULT_OG_IMAGE = "/og/default.svg";
const DEFAULT_OG_TYPE = "website";
const SEO_TAG_MARKER = "data-portfolio-seo";
const MAX_TITLE_LENGTH = 60;
const MAX_DESCRIPTION_LENGTH = 155;

const escapeAttributeValue = (value: string) => value.replace(/"/g, "&quot;");

const sanitizeTitle = (value: string) => {
  const normalized = value.trim().replace(/\s+/g, " ");
  return normalized.length > MAX_TITLE_LENGTH
    ? `${normalized.slice(0, MAX_TITLE_LENGTH - 3).trim()}...`
    : normalized;
};

const sanitizeDescription = (value: string) => {
  const normalized = value.trim().replace(/\s+/g, " ");
  return normalized.length > MAX_DESCRIPTION_LENGTH
    ? `${normalized.slice(0, MAX_DESCRIPTION_LENGTH - 3).trim()}...`
    : normalized;
};

const formatDateString = (value?: string) => {
  if (!value) return;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return;
  return date.toISOString();
};

interface SeoHeadProps {
  meta: PageMeta;
}

/**
 * Manages document head tags for SEO/OG/JSON-LD.
 * Injects on mount and cleans up on unmount — safe with React Router.
 */
export function SeoHead({ meta }: SeoHeadProps) {
  useEffect(() => {
    const cleanupNodes: HTMLElement[] = [];

    const removeBySelector = (selector: string) => {
      document.head.querySelectorAll(selector).forEach((node) => {
        if (node.parentElement) node.parentElement.removeChild(node);
      });
    };

    const setMeta = (kind: "name" | "property", key: string, content: string) => {
      const safeKey = escapeAttributeValue(key);
      const selector = `meta[${kind}="${safeKey}"]`;
      removeBySelector(selector);
      const el = document.createElement("meta");
      el.setAttribute(kind, key);
      el.setAttribute("content", content);
      el.setAttribute(SEO_TAG_MARKER, "true");
      document.head.appendChild(el);
      cleanupNodes.push(el);
    };

    const setLink = (rel: string, href: string) => {
      removeBySelector("link[rel=\"canonical\"]");
      const el = document.createElement("link");
      el.setAttribute("rel", rel);
      el.setAttribute("href", href);
      el.setAttribute(SEO_TAG_MARKER, "true");
      document.head.appendChild(el);
      cleanupNodes.push(el);
    };

    const setJsonLd = (value: unknown) => {
      const selector = "script[data-portfolio-seo]";
      removeBySelector(selector);
      const el = document.createElement("script");
      el.setAttribute("type", "application/ld+json");
      el.setAttribute(SEO_TAG_MARKER, "true");
      el.textContent = JSON.stringify(value);
      document.head.appendChild(el);
      cleanupNodes.push(el);
    };

    const canonicalUrl = new URL(
      meta.canonicalPath ?? window.location.pathname,
      window.location.origin,
    ).toString();
    const ogImage = new URL(meta.ogImage || DEFAULT_OG_IMAGE, window.location.origin).toString();
    const safeTitle = sanitizeTitle(meta.title);
    const safeDescription = sanitizeDescription(meta.description);
    const publishedTime = formatDateString(meta.publishedTime);
    const modifiedTime = formatDateString(meta.modifiedTime);

    // Title
    const previousTitle = document.title;
    document.title = safeTitle ? `${safeTitle} | ${SITE_NAME}` : SITE_NAME;

    // Core meta
    setMeta("name", "description", safeDescription);
    setMeta("name", "robots", meta.noindex ? "noindex, nofollow" : "index, follow");
    setMeta("name", "twitter:card", "summary_large_image");
    setMeta("name", "twitter:title", safeTitle || SITE_NAME);
    setMeta("name", "twitter:description", safeDescription);
    setMeta(
      "name",
      "twitter:image",
      new URL(
        meta.twitterImage || meta.ogImage || DEFAULT_OG_IMAGE,
        window.location.origin,
      ).toString(),
    );
    setMeta("name", "twitter:site", "@juan-canfield");

    if (meta.keywords && meta.keywords.length > 0) {
      setMeta("name", "keywords", meta.keywords.join(", "));
    } else {
      removeBySelector("meta[name=\"keywords\"]");
    }

    // Open Graph
    setMeta("property", "og:site_name", "Juan Canfield");
    setMeta("property", "og:title", safeTitle || SITE_NAME);
    setMeta("property", "og:description", safeDescription);
    setMeta("property", "og:type", meta.ogType || DEFAULT_OG_TYPE);
    setMeta("property", "og:image", ogImage);
    setMeta("property", "og:image:alt", safeTitle || SITE_NAME);
    setMeta("property", "og:locale", "en_US");
    if (canonicalUrl) {
      setLink("canonical", canonicalUrl);
      setMeta("property", "og:url", canonicalUrl);
    }
    if (meta.section) {
      setMeta("property", "article:section", meta.section);
    } else if (canonicalUrl?.includes("/insights/")) {
      setMeta("property", "article:section", "Insights");
    }
    if (publishedTime) {
      setMeta("property", "article:published_time", publishedTime);
    }
    if (modifiedTime) {
      setMeta("property", "article:modified_time", modifiedTime);
    }
    if (meta.ogType === "article") {
      setMeta("property", "article:tag", meta.keywords?.join(", ") || "AI systems, engineering");
    }

    if (meta.jsonLd) {
      setJsonLd(meta.jsonLd);
    }

    return () => {
      document.title = previousTitle;
      cleanupNodes.forEach((node) => node.remove());
    };
  }, [meta]);

  return null;
}
