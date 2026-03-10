import { useEffect } from 'react'
import type { FaqItem } from '../content/blog'

interface SeoHeadProps {
  title: string
  description: string
  canonical: string
  ogType?: string
  keywords?: string[]
  faq?: FaqItem[]
  jsonLd?: object
}

/** Track elements we create so cleanup removes exactly what we added. */
const managed = new Set<Element>()

function setMeta(property: string, content: string, isName = false) {
  const attr = isName ? 'name' : 'property'
  let el = document.head.querySelector(`meta[${attr}="${property}"]`)
  if (!el) {
    el = document.createElement('meta')
    el.setAttribute(attr, property)
    document.head.appendChild(el)
    managed.add(el)
  }
  el.setAttribute('content', content)
}

function setLink(rel: string, href: string) {
  let el = document.head.querySelector(`link[rel="${rel}"]`) as HTMLLinkElement | null
  if (!el) {
    el = document.createElement('link')
    el.setAttribute('rel', rel)
    document.head.appendChild(el)
    managed.add(el)
  }
  el.setAttribute('href', href)
}

function setJsonLd(id: string, data: object) {
  let el = document.getElementById(id) as HTMLScriptElement | null
  if (!el) {
    el = document.createElement('script')
    el.id = id
    el.type = 'application/ld+json'
    document.head.appendChild(el)
    managed.add(el)
  }
  el.textContent = JSON.stringify(data)
}

function cleanup() {
  for (const el of managed) {
    el.remove()
  }
  managed.clear()
}

export default function SeoHead({ title, description, canonical, ogType = 'article', keywords, faq, jsonLd }: SeoHeadProps) {
  useEffect(() => {
    document.title = title

    setMeta('description', description, true)
    setMeta('og:title', title)
    setMeta('og:description', description)
    setMeta('og:url', canonical)
    setMeta('og:type', ogType)
    setMeta('og:site_name', 'Churn Signals')
    setMeta('twitter:card', 'summary_large_image', true)
    setMeta('twitter:title', title, true)
    setMeta('twitter:description', description, true)
    setLink('canonical', canonical)

    if (keywords && keywords.length > 0) {
      setMeta('keywords', keywords.join(', '), true)
    }

    if (jsonLd) {
      setJsonLd('seo-jsonld', jsonLd)
    }

    if (faq && faq.length > 0) {
      setJsonLd('seo-faq-jsonld', {
        '@context': 'https://schema.org',
        '@type': 'FAQPage',
        mainEntity: faq.map(item => ({
          '@type': 'Question',
          name: item.question,
          acceptedAnswer: {
            '@type': 'Answer',
            text: item.answer,
          },
        })),
      })
    }

    return cleanup
  }, [title, description, canonical, ogType, keywords, faq, jsonLd])

  return null
}
