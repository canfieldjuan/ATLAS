# SEO / AEO / GEO Gap Remediation Plan

**Created:** 2026-03-27
**Branch:** `feature/nextjs-seo-migration`
**Status:** Gaps 1-7, 9-10 COMPLETE. Gaps 8, 11 deferred.
**Reference:** `/home/juan-canfield/Desktop/AEO-GEO-SEO.md`

---

## Gaps (ordered by severity)

### Gap 1: Homepage missing WebSite + Organization JSON-LD
**Severity:** HIGH
**AEO requirement:** Homepage must have WebSite + Organization structured data for entity clarity and knowledge panel eligibility.

**Fix:** Add JSON-LD script to homepage (`app/(marketing)/page.tsx`) with:
- `@type: WebSite` — name, url, inLanguage
- `@type: Organization` — name, url, logo, sameAs (social links)

**File:** `app/(marketing)/page.tsx`
**Insertion:** After metadata export, before JSX return — add `<script type="application/ld+json">` in the component body.

---

### Gap 2: Homepage missing canonical
**Severity:** HIGH
**AEO requirement:** "Every indexable page must output `<link rel=canonical>`"

**Fix:** Add `alternates.canonical` to homepage `generateMetadata` (or static `metadata` export).

**File:** `app/(marketing)/page.tsx`
**Change:** Add `alternates: { canonical: SITE_URL }` to metadata object.

---

### Gap 3: Blog listing missing canonical
**Severity:** MEDIUM
**AEO requirement:** Same as Gap 2.

**Fix:** Add `alternates: { canonical: SITE_URL + "/blog" }` to blog listing metadata.

**File:** `app/(marketing)/blog/page.tsx`
**Change:** Add `alternates` field to metadata export.

---

### Gap 4: No BreadcrumbList JSON-LD on articles
**Severity:** MEDIUM
**AEO requirement:** "BreadcrumbList JSON-LD: Recommended for articles" — helps Google show breadcrumb trails in search results and AI answers.

**Fix:** Add BreadcrumbList JSON-LD to each blog post page:
```json
{
  "@type": "BreadcrumbList",
  "itemListElement": [
    { "@type": "ListItem", "position": 1, "name": "Home", "item": "https://churnsignals.co" },
    { "@type": "ListItem", "position": 2, "name": "Blog", "item": "https://churnsignals.co/blog" },
    { "@type": "ListItem", "position": 3, "name": "Post Title" }
  ]
}
```

**File:** `app/(marketing)/blog/[slug]/page.tsx`
**Insertion:** Add third `<script type="application/ld+json">` block alongside BlogPosting and FAQPage.

---

### Gap 5: No dateModified on BlogPosting
**Severity:** LOW
**AEO requirement:** Article structured data template includes `dateModified`.

**Fix:** Add `dateModified` to BlogPosting JSON-LD. Use `post.date` as default (same as `datePublished`) since blog content files don't track modification dates separately.

**File:** `app/(marketing)/blog/[slug]/page.tsx`
**Change:** Add `dateModified: post.date` to jsonLd object.

---

### Gap 6: No og:image per post
**Severity:** MEDIUM
**AEO requirement:** "og:image: Required for articles" — social sharing and AI answer engine thumbnails.

**Fix — Two options:**
- **Option A (quick):** Set a default OG image (`/og-default.png`) in root metadata. All posts inherit it.
- **Option B (better):** Generate per-post OG images using Next.js `opengraph-image.tsx` route convention or a shared template with post title + category overlay.

**Recommendation:** Start with Option A, upgrade to B later.

**File:** `app/layout.tsx` (root metadata) + `public/og-default.png`
**Change:** Add `openGraph.images` to root metadata. Ensure `og-default.png` exists in `public/`.

---

### Gap 7: FAQPage JSON-LD not rendering
**Severity:** MEDIUM
**AEO requirement:** "FAQPage structured data" — enables FAQ rich snippets in Google.

**Bug:** The code checks `post.faq && post.faq.length > 0` but the built HTML shows no FAQPage block. Likely the consumer blog posts don't have FAQ data (`faq: undefined`).

**Investigation:** Check if any of the 14 bundled posts actually have `faq` populated. If none do, this is a data gap not a code gap.

**File:** `app/(marketing)/blog/[slug]/page.tsx` (code is correct, verify data)
**Action:** Verify, then either populate FAQ on posts or confirm code works when FAQ exists.

---

### Gap 8: No IndexNow ping on deploy
**Severity:** LOW
**AEO requirement:** "IndexNow: for participating engines, push URL updates via IndexNow"

**Fix:** Add a post-build or post-deploy script that pings IndexNow with new/changed URLs. Can be a Vercel deploy hook or a `postbuild` npm script.

**File:** New `scripts/indexnow.ts` or addition to `package.json` scripts.
**Deferred:** Low priority — Google discovers via sitemap anyway. Bing/Yandex benefit from IndexNow.

---

### Gap 9: No sameAs links on Organization
**Severity:** LOW
**AEO requirement:** "sameAs links (where real)" for authoritativeness signals.

**Fix:** Add `sameAs` array to Organization JSON-LD on homepage with real social/company URLs.

**File:** `app/(marketing)/page.tsx` (homepage JSON-LD from Gap 1)
**Change:** Include in Organization block: company LinkedIn, Twitter/X, etc.

---

### Gap 10: Chart placeholder instead of real charts
**Severity:** MEDIUM
**AEO requirement:** Content quality — "clear data presentation" for AI extractability. Charts are referenced in blog content but show a placeholder box.

**Fix:** Replace the placeholder `ChartEmbed` in `blog-post-content.tsx` with the actual `BlogChartRenderer` component (already exists in `components/BlogChartRenderer.tsx`, copied from the old project).

**File:** `app/(marketing)/blog/[slug]/blog-post-content.tsx`
**Change:** Import and use `BlogChartRenderer` instead of placeholder.

---

### Gap 11: B2B blog posts not in this frontend
**Severity:** N/A (architectural decision)
**Context:** The 14 bundled posts are consumer-side (Amazon reviews). B2B blog posts generated by the Atlas pipeline (stored in `blog_posts` table) are not in this frontend. The old SPA fetched them from the same bundled TS files.

**Decision needed:** Will B2B blog posts be:
- **(A)** Fetched from the Atlas API at build time (ISR/SSG with revalidation)?
- **(B)** Kept as bundled TS files (current approach, requires rebuild to publish)?
- **(C)** Served from a separate domain/deployment?

**Recommendation:** Option A — fetch from Atlas API at build time using `generateStaticParams` + ISR revalidation. New posts auto-appear without rebuild. This is the production architecture.

**Deferred:** Separate task — requires API endpoint for blog content delivery.

---

## Execution Order

```
Gap 1 + 2 + 9  (homepage JSON-LD + canonical + sameAs)  — single file change
Gap 3           (blog listing canonical)                  — single line
Gap 4 + 5       (BreadcrumbList + dateModified)           — blog post page
Gap 6           (og:image default)                        — root layout + asset
Gap 7           (FAQPage debug)                           — verify data
Gap 10          (chart renderer)                          — swap component
Gap 8           (IndexNow)                                — deferred
Gap 11          (B2B blog API)                            — deferred/separate task
```

Gaps 1-7 + 10 can be done in one session. Gaps 8 and 11 are follow-ups.

---

## Files Modified (planned)

| File | Gaps Addressed |
|------|----------------|
| `app/(marketing)/page.tsx` | 1, 2, 9 |
| `app/(marketing)/blog/page.tsx` | 3 |
| `app/(marketing)/blog/[slug]/page.tsx` | 4, 5 |
| `app/layout.tsx` | 6 |
| `app/(marketing)/blog/[slug]/blog-post-content.tsx` | 10 |
| `public/og-default.png` | 6 |
