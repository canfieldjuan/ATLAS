/**
 * IndexNow: notify Bing/Yandex/other participating engines of new/updated URLs.
 *
 * Run after deploy:  npx tsx scripts/indexnow.ts
 *
 * Reads the sitemap to extract all URLs and submits them in a single batch.
 * IndexNow spec: https://www.indexnow.org/documentation
 */

const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL || "https://churnsignals.co";
const INDEXNOW_KEY = "6a1d58a9cdf74c7b9c948ac27f86ec7a";
const INDEXNOW_ENDPOINT = "https://api.indexnow.org/IndexNow";

async function main() {
  // Dynamically import the sitemap to get all URLs
  const { POSTS } = await import("../content/blog/index.js");

  const urls = [
    SITE_URL,
    `${SITE_URL}/blog`,
    ...POSTS.map((p: { slug: string }) => `${SITE_URL}/blog/${p.slug}`),
  ];

  console.log(`Submitting ${urls.length} URLs to IndexNow...`);

  const body = {
    host: new URL(SITE_URL).hostname,
    key: INDEXNOW_KEY,
    keyLocation: `${SITE_URL}/${INDEXNOW_KEY}.txt`,
    urlList: urls,
  };

  try {
    const res = await fetch(INDEXNOW_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json; charset=utf-8" },
      body: JSON.stringify(body),
    });
    console.log(`IndexNow response: ${res.status} ${res.statusText}`);
    if (!res.ok) {
      const text = await res.text();
      console.error("Body:", text);
    }
  } catch (err) {
    console.error("IndexNow submission failed:", err);
    process.exit(1);
  }
}

main();
