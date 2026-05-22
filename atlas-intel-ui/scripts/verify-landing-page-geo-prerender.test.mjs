import assert from 'node:assert/strict'
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'

import { verifyLandingPagePrerender } from './verify-landing-page-geo-prerender.mjs'

const SITE_URL = 'https://atlas-intel-ui-two.vercel.app'

function withTempDist(fn) {
  const distDir = mkdtempSync(join(tmpdir(), 'atlas-lp-prerender-'))
  try {
    return fn(distDir)
  } finally {
    rmSync(distDir, { recursive: true, force: true })
  }
}

function writeSitemap(distDir, urls) {
  writeFileSync(
    join(distDir, 'sitemap.xml'),
    `<?xml version="1.0" encoding="UTF-8"?>
<urlset>
${urls.map(url => `  <url><loc>${url}</loc></url>`).join('\n')}
</urlset>
`,
  )
}

function writeLandingPageHtml(distDir, path, html) {
  const outDir = join(distDir, ...path.split('/').filter(Boolean))
  mkdirSync(outDir, { recursive: true })
  writeFileSync(join(outDir, 'index.html'), html)
}

function landingPageHtml({ loc = `${SITE_URL}/lp/111/acme-page` } = {}) {
  return `<!doctype html>
<html>
  <head>
    <title>Acme Support FAQ Report</title>
    <meta name="description" content="Turn support tickets into clear FAQ answers." />
    <link rel="canonical" href="${loc}" />
    <meta name="robots" content="index,follow" />
    <meta property="og:url" content="${loc}" />
    <script type="application/ld+json">{"@context":"https://schema.org","@graph":[{"@type":"WebPage"}]}</script>
  </head>
  <body>
    <article data-prerendered-landing-page="true">
      <h1>Acme Support FAQ Report</h1>
      <p>Turn repeat questions into answers customers can find.</p>
      <a href="/upload">Upload Ticket CSV</a>
    </article>
  </body>
</html>`
}

function landingPageHtmlWithTitle(title, loc = `${SITE_URL}/lp/111/acme-page`) {
  return landingPageHtml({ loc }).replace(
    '<title>Acme Support FAQ Report</title>',
    `<title>${title}</title>`,
  )
}

test('verifyLandingPagePrerender passes when lp sitemap entry has static HTML contract', () => {
  withTempDist((distDir) => {
    const loc = `${SITE_URL}/lp/111/acme-page`
    writeSitemap(distDir, [loc])
    writeLandingPageHtml(distDir, '/lp/111/acme-page', landingPageHtml({ loc }))

    const result = verifyLandingPagePrerender({
      distDir,
      logger: { log() {} },
    })

    assert.equal(result.checked, 1)
    assert.deepEqual(result.failures, [])
  })
})

test('verifyLandingPagePrerender fails when lp sitemap entry lacks static HTML', () => {
  withTempDist((distDir) => {
    writeSitemap(distDir, [`${SITE_URL}/lp/111/acme-page`])

    const result = verifyLandingPagePrerender({
      distDir,
      logger: { log() {} },
    })

    assert.equal(result.checked, 1)
    assert.match(result.failures[0], /missing .*index\.html/)
  })
})

test('verifyLandingPagePrerender fails when static HTML keeps the fallback title', () => {
  withTempDist((distDir) => {
    const loc = `${SITE_URL}/lp/111/acme-page`
    writeSitemap(distDir, [loc])
    writeLandingPageHtml(
      distDir,
      '/lp/111/acme-page',
      landingPageHtmlWithTitle(
        'Atlas Intelligence \u2014 Amazon Review Monitoring & Competitor Signals',
        loc,
      ),
    )

    const result = verifyLandingPagePrerender({
      distDir,
      logger: { log() {} },
    })

    assert.equal(result.checked, 1)
    assert.ok(
      result.failures.includes('/lp/111/acme-page has missing or fallback <title>'),
    )
  })
})

test('verifyLandingPagePrerender skips builds with no generated landing pages', () => {
  withTempDist((distDir) => {
    writeSitemap(distDir, [`${SITE_URL}/blog/example`])

    const result = verifyLandingPagePrerender({
      distDir,
      logger: { log() {} },
    })

    assert.equal(result.checked, 0)
    assert.deepEqual(result.failures, [])
  })
})
