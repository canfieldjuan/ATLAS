import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { marked } from 'marked'
import { ArrowRight, Loader2 } from 'lucide-react'
import {
  fetchPublicLandingPageDraft,
  type GeneratedAssetDraft,
} from '../api/contentOps'
import PublicLayout from '../components/PublicLayout'
import SeoHead from '../components/SeoHead'

type LandingPageSectionView = {
  id: string
  title: string
  body: string
}

type PageState =
  | { kind: 'idle'; id: string }
  | { kind: 'loaded'; id: string; page: GeneratedAssetDraft }
  | { kind: 'not_found'; id: string }
  | { kind: 'error'; id: string; message: string }

export default function PublicLandingPage() {
  const { id = '', slug = '' } = useParams<{ id: string; slug: string }>()
  const navigate = useNavigate()
  const [pageState, setPageState] = useState<PageState>({ kind: 'idle', id: '' })

  useEffect(() => {
    let cancelled = false
    fetchPublicLandingPageDraft(id)
      .then((draft) => {
        if (cancelled) return
        setPageState({ kind: 'loaded', id, page: draft })
        const canonicalSlug = textValue(draft.slug)
        if (canonicalSlug && canonicalSlug !== slug) {
          navigate(`/lp/${encodeURIComponent(id)}/${canonicalSlug}`, { replace: true })
        }
      })
      .catch((err) => {
        if (cancelled) return
        const message = err instanceof Error ? err.message : String(err)
        if (message.includes('API 404')) {
          setPageState({ kind: 'not_found', id })
        } else {
          setPageState({ kind: 'error', id, message })
        }
      })
    return () => {
      cancelled = true
    }
  }, [id, slug, navigate])

  const routeState: PageState = pageState.id === id ? pageState : { kind: 'idle', id }
  const page = routeState.kind === 'loaded' ? routeState.page : null

  const canonical = useMemo(() => {
    if (!page) return ''
    const canonicalSlug = textValue(page.slug) || slug
    return `${window.location.origin}/lp/${encodeURIComponent(id)}/${canonicalSlug}`
  }, [id, page, slug])

  if (routeState.kind === 'idle') {
    return (
      <PublicLayout>
        <section className="mx-auto flex min-h-[50vh] max-w-3xl items-center justify-center px-6 py-24">
          <Loader2 className="h-6 w-6 animate-spin text-cyan-300" />
        </section>
      </PublicLayout>
    )
  }

  if (routeState.kind === 'error') {
    return (
      <PublicLayout>
        <section className="mx-auto max-w-3xl px-6 py-24 text-center">
          <h1 className="text-3xl font-bold">Unable to load page</h1>
          <p className="mt-4 text-slate-400">{routeState.message}</p>
        </section>
      </PublicLayout>
    )
  }

  if (routeState.kind === 'not_found' || !page) {
    return (
      <PublicLayout>
        <section className="mx-auto max-w-3xl px-6 py-24 text-center">
          <h1 className="text-3xl font-bold">Landing page not found</h1>
          <p className="mt-4 text-slate-400">
            This page is not public, has moved, or is no longer available.
          </p>
          <Link
            to="/landing"
            className="mt-8 inline-flex items-center gap-2 rounded-lg bg-cyan-600 px-5 py-3 text-sm font-semibold text-white hover:bg-cyan-500"
          >
            Back to Atlas
            <ArrowRight className="h-4 w-4" />
          </Link>
        </section>
      </PublicLayout>
    )
  }

  const hero = recordValue(page.hero)
  const cta = recordValue(page.cta)
  const meta = recordValue(page.meta)
  const sections = sectionList(page.sections)
  const title = textValue(meta?.title_tag) || textValue(page.title) || 'Landing Page'
  const description =
    textValue(meta?.description) ||
    textValue(hero?.subheadline) ||
    textValue(page.value_prop) ||
    title
  const jsonLd = structuredDataWithCanonical(page.structured_data, canonical)
  const ctaLabel = textValue(cta?.label) || textValue(hero?.cta_label)
  const ctaUrl = textValue(cta?.url) || textValue(hero?.cta_url)
  const robots = textValue(page.robots) || 'noindex,follow'

  return (
    <>
      <SeoHead
        title={title}
        description={description}
        canonical={canonical}
        ogType="website"
        jsonLd={jsonLd}
        robots={robots}
      />
      <PublicLayout>
        <article>
          <section className="mx-auto max-w-5xl px-6 pb-20 pt-14">
            <div className="max-w-3xl">
              {textValue(page.persona) && (
                <p className="text-sm font-semibold uppercase tracking-wide text-cyan-300">
                  {textValue(page.persona)}
                </p>
              )}
              <h1 className="mt-4 text-4xl font-bold leading-tight sm:text-5xl">
                {textValue(hero?.headline) || textValue(page.title)}
              </h1>
              {(textValue(hero?.subheadline) || textValue(page.value_prop)) && (
                <p className="mt-6 text-lg leading-8 text-slate-300">
                  {textValue(hero?.subheadline) || textValue(page.value_prop)}
                </p>
              )}
              {ctaLabel && ctaUrl && (
                <a
                  href={ctaUrl}
                  className="mt-8 inline-flex items-center gap-2 rounded-lg bg-cyan-600 px-5 py-3 text-sm font-semibold text-white hover:bg-cyan-500"
                >
                  {ctaLabel}
                  <ArrowRight className="h-4 w-4" />
                </a>
              )}
            </div>
          </section>

          {sections.length > 0 && (
            <section className="border-t border-slate-800">
              <div className="mx-auto max-w-5xl space-y-10 px-6 py-16">
                {sections.map((section, index) => (
                  <section
                    key={section.id || `${section.title}-${index}`}
                    className="max-w-3xl"
                  >
                    <h2 className="text-2xl font-semibold text-white">
                      {section.title || `Section ${index + 1}`}
                    </h2>
                    {section.body && (
                      <div
                        className="blog-prose mt-4 text-slate-300"
                        dangerouslySetInnerHTML={{
                          __html: renderSafeMarkdown(section.body),
                        }}
                      />
                    )}
                  </section>
                ))}
              </div>
            </section>
          )}

          {ctaLabel && ctaUrl && (
            <section className="border-t border-slate-800">
              <div className="mx-auto max-w-5xl px-6 py-16">
                <div className="max-w-3xl rounded-xl border border-slate-700 bg-slate-800/60 p-8">
                  <h2 className="text-2xl font-semibold text-white">
                    {textValue(page.title)}
                  </h2>
                  <p className="mt-3 text-slate-300">{description}</p>
                  <a
                    href={ctaUrl}
                    className="mt-6 inline-flex items-center gap-2 rounded-lg bg-cyan-600 px-5 py-3 text-sm font-semibold text-white hover:bg-cyan-500"
                  >
                    {ctaLabel}
                    <ArrowRight className="h-4 w-4" />
                  </a>
                </div>
              </div>
            </section>
          )}
        </article>
      </PublicLayout>
    </>
  )
}

function sectionList(value: unknown): LandingPageSectionView[] {
  return recordList(value)
    .map((section) => ({
      id: textValue(section.id),
      title: textValue(section.title) || textValue(section.heading),
      body: textValue(section.body_markdown) || textValue(section.body),
    }))
    .filter((section) => section.title || section.body)
}

function recordList(value: unknown): Record<string, unknown>[] {
  let rows = Array.isArray(value) ? value : []
  if (typeof value === 'string' && value.trim()) {
    try {
      const parsed = JSON.parse(value)
      rows = Array.isArray(parsed) ? parsed : []
    } catch {
      rows = []
    }
  }
  return rows.filter((item): item is Record<string, unknown> =>
    Boolean(item && typeof item === 'object' && !Array.isArray(item)),
  )
}

function recordValue(value: unknown): Record<string, unknown> | null {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    return value as Record<string, unknown>
  }
  if (typeof value === 'string' && value.trim()) {
    try {
      const parsed = JSON.parse(value)
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>
      }
    } catch {
      return null
    }
  }
  return null
}

function textValue(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function renderSafeMarkdown(markdown: string): string {
  const escapedMarkdown = markdown.replace(/[<>]/g, (char) =>
    char === '<' ? '&lt;' : '&gt;',
  )
  const html = marked.parse(escapedMarkdown, { async: false }) as string
  return sanitizeRenderedHtml(html)
}

function sanitizeRenderedHtml(html: string): string {
  const template = document.createElement('template')
  template.innerHTML = html
  for (const element of Array.from(template.content.querySelectorAll('*'))) {
    for (const attr of Array.from(element.attributes)) {
      const name = attr.name.toLowerCase()
      if (name.startsWith('on') || name === 'style') {
        element.removeAttribute(attr.name)
        continue
      }
      if ((name === 'href' || name === 'src') && !safeUrl(attr.value)) {
        element.removeAttribute(attr.name)
      }
    }
  }
  return template.innerHTML
}

function safeUrl(value: string): boolean {
  const normalized = value.trim().toLowerCase()
  return (
    normalized.startsWith('https://') ||
    normalized.startsWith('http://') ||
    normalized.startsWith('mailto:') ||
    normalized.startsWith('tel:') ||
    normalized.startsWith('/') ||
    normalized.startsWith('#')
  )
}

function structuredDataWithCanonical(value: unknown, canonical: string): object | undefined {
  const raw = recordValue(value)
  if (!raw) return undefined
  const graph = recordList(raw['@graph'])
  if (graph.length === 0) return raw
  return {
    ...raw,
    '@graph': graph.map((node) => {
      const type = node['@type']
      if (type === 'WebPage') {
        return { ...node, '@id': `${canonical}#webpage`, url: canonical }
      }
      if (type === 'FAQPage') {
        return {
          ...node,
          '@id': `${canonical}#faq`,
          mainEntityOfPage: { '@id': `${canonical}#webpage` },
        }
      }
      return node
    }),
  }
}
