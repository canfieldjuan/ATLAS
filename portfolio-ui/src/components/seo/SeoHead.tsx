import { useEffect } from "react";
import type { PageMeta } from "@/types";

const SITE_NAME = "Juan Canfield — AI Systems Architect";
const DEFAULT_OG_IMAGE = "/og/default.png";

interface SeoHeadProps {
  meta: PageMeta;
}

/**
 * Manages document head tags for SEO/OG/JSON-LD.
 * Injects on mount, cleans up on unmount — safe with React Router.
 */
export function SeoHead({ meta }: SeoHeadProps) {
  useEffect(() => {
    const tags: HTMLElement[] = [];

    const set = (
      tag: "meta" | "link" | "script",
      attrs: Record<string, string>,
      content?: string,
    ) => {
      const el = document.createElement(tag);
      Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
      if (content) el.textContent = content;
      document.head.appendChild(el);
      tags.push(el);
    };

    // Title
    const prevTitle = document.title;
    document.title = meta.title
      ? `${meta.title} | ${SITE_NAME}`
      : SITE_NAME;

    // Standard meta
    set("meta", { name: "description", content: meta.description });

    // Open Graph
    set("meta", { property: "og:title", content: meta.title || SITE_NAME });
    set("meta", { property: "og:description", content: meta.description });
    set("meta", {
      property: "og:image",
      content: meta.ogImage || DEFAULT_OG_IMAGE,
    });
    if (meta.canonicalPath) {
      set("link", {
        rel: "canonical",
        href: `${window.location.origin}${meta.canonicalPath}`,
      });
      set("meta", {
        property: "og:url",
        content: `${window.location.origin}${meta.canonicalPath}`,
      });
    }

    // Twitter
    set("meta", { name: "twitter:title", content: meta.title || SITE_NAME });
    set("meta", {
      name: "twitter:description",
      content: meta.description,
    });

    // JSON-LD
    if (meta.jsonLd) {
      set(
        "script",
        { type: "application/ld+json" },
        JSON.stringify(meta.jsonLd),
      );
    }

    return () => {
      document.title = prevTitle;
      tags.forEach((el) => el.remove());
    };
  }, [meta]);

  return null;
}
