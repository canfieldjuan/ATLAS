import type { MetadataRoute } from "next";
import { SITE_URL } from "@/lib/constants";

export default function robots(): MetadataRoute.Robots {
  return {
    rules: [
      {
        userAgent: "*",
        allow: ["/", "/blog/"],
        disallow: [
          "/api/",
          "/login",
          "/signup",
          "/onboarding",
          "/account",
          "/dashboard",
          "/brands",
          "/compare",
          "/flows",
          "/features",
          "/safety",
          "/reviews",
          "/b2b/",
        ],
      },
    ],
    sitemap: `${SITE_URL}/sitemap.xml`,
  };
}
