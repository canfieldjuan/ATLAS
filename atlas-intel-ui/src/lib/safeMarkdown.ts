import { marked } from 'marked'

export function renderSafeMarkdown(markdown: string): string {
  const escapedMarkdown = markdown.replace(/[<>]/g, (char) =>
    char === '<' ? '&lt;' : '&gt;',
  )
  const html = marked.parse(escapedMarkdown, { async: false }) as string
  return sanitizeRenderedHtml(html)
}

export function sanitizeRenderedHtml(html: string): string {
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

export function safeUrl(value: string): boolean {
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
