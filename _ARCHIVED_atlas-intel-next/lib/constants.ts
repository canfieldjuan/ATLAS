/**
 * Production site URL. Used for canonical URLs, sitemap, robots.txt,
 * JSON-LD structured data, and Open Graph metadata.
 *
 * Set NEXT_PUBLIC_SITE_URL in .env to override (e.g., for staging).
 */
export const SITE_URL =
  process.env.NEXT_PUBLIC_SITE_URL || "https://churnsignals.co";
