import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { existsSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { resolve, join } from "node:path";

type UrlEntry = {
  loc: string;
  priority: string;
  changefreq: string;
};

const SITE_URL = process.env.PORTFOLIO_SITE_URL || process.env.VITE_SITE_URL;

function extractSlugsFromDirectory(directory: string): string[] {
  if (!existsSync(directory)) return [];

  const slugs = new Set<string>();
  const files = readdirSync(directory);

  for (const file of files) {
    if (!file.endsWith(".ts") || file === "index.ts") continue;
    const content = readFileSync(resolve(directory, file), "utf-8");
    const match = content.match(/slug:\s*["']([^"']+)["']/);
    if (match && match[1]) slugs.add(match[1]);
  }

  return [...slugs];
}

function normalizeUrl(path: string): string {
  if (!SITE_URL) return "";
  try {
    return new URL(path, SITE_URL).toString();
  } catch {
    return "";
  }
}

function sitemapPlugin() {
  return {
    name: "generate-sitemap",
    closeBundle() {
      if (!SITE_URL) {
        console.warn(
          "  Skipping sitemap generation: set PORTFOLIO_SITE_URL or VITE_SITE_URL before build.",
        );
        return;
      }

      const projectSlugs = extractSlugsFromDirectory(
        resolve(import.meta.dirname, "src/content/projects"),
      );
      const insightSlugs = extractSlugsFromDirectory(
        resolve(import.meta.dirname, "src/content/insights"),
      );

      const staticRoutes = [
        "/",
        "/about",
        "/services",
        "/systems",
        "/projects",
        "/insights",
        "/framework",
      ];

      const urls: UrlEntry[] = [
        ...staticRoutes.map((route, i) => {
          const loc = normalizeUrl(route);
          if (!loc) {
            return null;
          }

          return {
            loc,
            priority: i === 0 ? "1.0" : "0.8",
            changefreq: i === 0 ? "weekly" : "monthly",
          };
        }),
        ...projectSlugs.map((slug) => ({
          loc: normalizeUrl(`/projects/${slug}`),
          priority: "0.7",
          changefreq: "monthly",
        })),
        ...insightSlugs.map((slug) => ({
          loc: normalizeUrl(`/insights/${slug}`),
          priority: "0.7",
          changefreq: "monthly",
        })),
      ].filter((entry): entry is UrlEntry => Boolean(entry.loc));

      const today = new Date().toISOString().split("T")[0];
      const xml = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${urls
  .map(
    (u) => `  <url>
    <loc>${u.loc}</loc>
    <lastmod>${today}</lastmod>
    <changefreq>${u.changefreq}</changefreq>
    <priority>${u.priority}</priority>
  </url>`,
  )
  .join("\n")}
</urlset>
`;

      const distDir = resolve(import.meta.dirname, "dist");
      if (existsSync(distDir)) {
        writeFileSync(join(distDir, "sitemap.xml"), xml);
        console.log(`  Sitemap generated with ${urls.length} URLs`);
      }
    },
  };
}

function robotsPlugin() {
  return {
    name: "generate-robots",
    closeBundle() {
      if (!SITE_URL) {
        return;
      }

      const distDir = resolve(import.meta.dirname, "dist");
      const robotsPath = resolve(distDir, "robots.txt");

      if (!existsSync(distDir) || !existsSync(robotsPath)) {
        return;
      }

      const currentRobots = readFileSync(robotsPath, "utf-8");
      if (/^Sitemap:/im.test(currentRobots)) {
        return;
      }

      const sitemapLine = `Sitemap: ${normalizeUrl("/sitemap.xml")}`;
      const trimmed = currentRobots.trimEnd();
      const next = trimmed.length > 0 ? `${trimmed}\n${sitemapLine}\n` : `${sitemapLine}\n`;
      writeFileSync(robotsPath, next);
      console.log("  Added sitemap directive to dist/robots.txt");
    },
  };
}

export default defineConfig({
  plugins: [react(), sitemapPlugin(), robotsPlugin()],
  resolve: {
    alias: {
      "@": resolve(__dirname, "src"),
    },
  },
  server: {
    port: 5175,
  },
  build: {
    outDir: "dist",
    sourcemap: false,
  },
});
