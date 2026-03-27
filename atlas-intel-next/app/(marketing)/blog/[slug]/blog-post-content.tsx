"use client";

import { useMemo } from "react";
import { marked } from "marked";
import type { ChartSpec } from "@/content/blog";
import BlogChart from "@/components/BlogChartRenderer";

export default function BlogPostContent({
  content,
  charts,
}: {
  content: string;
  charts?: ChartSpec[];
}) {
  const chartMap = useMemo(() => {
    const map = new Map<string, ChartSpec>();
    for (const c of charts || []) {
      map.set(c.chart_id, c);
    }
    return map;
  }, [charts]);

  const html = useMemo(() => {
    return marked.parse(content, { async: false }) as string;
  }, [content]);

  // Split HTML on chart placeholders and interleave chart components
  const segments = useMemo(() => {
    const parts = html.split(/\{\{chart:([a-z0-9_-]+)\}\}/gi);
    const result: { type: "html" | "chart"; value: string }[] = [];
    for (let i = 0; i < parts.length; i++) {
      if (i % 2 === 0) {
        if (parts[i]) result.push({ type: "html", value: parts[i] });
      } else {
        result.push({ type: "chart", value: parts[i] });
      }
    }
    return result;
  }, [html]);

  return (
    <div className="prose prose-invert prose-slate max-w-none">
      {segments.map((seg, i) =>
        seg.type === "html" ? (
          <div key={i} dangerouslySetInnerHTML={{ __html: seg.value }} />
        ) : chartMap.has(seg.value) ? (
          <BlogChart key={i} spec={chartMap.get(seg.value)!} />
        ) : null,
      )}
    </div>
  );
}
