import { existsSync, readFileSync, readdirSync } from 'node:fs'
import { join } from 'node:path'

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function parseStringField(content, field) {
  const pattern = new RegExp(`^\\s*${escapeRegExp(field)}:\\s*'((?:\\\\'|[^'])*)'`, 'm')
  const match = content.match(pattern)
  return match ? match[1].replace(/\\'/g, "'") : ''
}

function parseTemplateField(content, field) {
  const pattern = new RegExp(`^\\s*${escapeRegExp(field)}:\\s*\`([\\s\\S]*?)\`\\s*,`, 'm')
  const match = content.match(pattern)
  return match ? match[1] : ''
}

function parseArrayField(content, field) {
  const fieldMatch = new RegExp(`^\\s*${escapeRegExp(field)}:\\s*\\[`, 'm').exec(content)
  if (!fieldMatch) return ''

  const start = fieldMatch.index + fieldMatch[0].lastIndexOf('[')
  let depth = 0
  let quote = ''
  let escaped = false

  for (let index = start; index < content.length; index += 1) {
    const char = content[index]

    if (quote) {
      if (escaped) {
        escaped = false
      } else if (char === '\\') {
        escaped = true
      } else if (char === quote) {
        quote = ''
      }
      continue
    }

    if (char === '"' || char === "'" || char === '`') {
      quote = char
      continue
    }

    if (char === '[') depth += 1
    if (char === ']') depth -= 1

    if (depth === 0) {
      return content.slice(start, index + 1)
    }
  }

  throw new Error(`Unclosed array field in blog source: ${field}`)
}

function parseChartsField(content, file) {
  const chartsLiteral = parseArrayField(content, 'charts')
  if (!chartsLiteral) return []

  try {
    return JSON.parse(chartsLiteral)
  } catch (error) {
    throw new Error(`Invalid charts JSON in blog source ${file}: ${error.message}`)
  }
}

export function chartPlaceholderIds(content) {
  return [...content.matchAll(/<p>\s*\{\{chart:([^}]+)\}\}\s*<\/p>|\{\{chart:([^}]+)\}\}/g)]
    .map(match => (match[1] || match[2] || '').trim())
    .filter(Boolean)
}

function assertKnownChartPlaceholders(content, charts, slug) {
  const chartIds = new Set(charts.map(chart => chart.chart_id))
  const unknownChartIds = [...new Set(chartPlaceholderIds(content).filter(chartId => !chartIds.has(chartId)))]
  if (unknownChartIds.length) {
    throw new Error(`/blog/${slug} source references missing chart fallback data: ${unknownChartIds.join(', ')}`)
  }
}

export function collectBlogSourceMetadata(rootDir) {
  const blogDir = join(rootDir, 'src/content/blog')
  if (!existsSync(blogDir)) return []

  const posts = []
  for (const file of readdirSync(blogDir).sort()) {
    if (!file.endsWith('.ts') || file === 'index.ts') continue
    const source = readFileSync(join(blogDir, file), 'utf-8')
    const slug = parseStringField(source, 'slug')
    const title = parseStringField(source, 'title')
    const description = parseStringField(source, 'description')
    const date = parseStringField(source, 'date')
    const author = parseStringField(source, 'author')
    const seoTitle = parseStringField(source, 'seo_title')
    const seoDescription = parseStringField(source, 'seo_description')
    const content = parseTemplateField(source, 'content')
    const charts = parseChartsField(source, file)

    if (!slug) throw new Error(`Missing slug in blog source: ${file}`)
    if (!title) throw new Error(`Missing title in blog source: ${file}`)
    if (!description) throw new Error(`Missing description in blog source: ${file}`)
    if (!date) throw new Error(`Missing date in blog source: ${file}`)
    if (!author) throw new Error(`Missing author in blog source: ${file}`)
    if (!content.trim()) throw new Error(`Missing content in blog source: ${file}`)
    assertKnownChartPlaceholders(content, charts, slug)

    posts.push({
      file,
      slug,
      title,
      description,
      date,
      author,
      seoTitle,
      seoDescription,
      content,
      charts,
    })
  }

  if (!posts.length) throw new Error('No blog posts found')
  return posts.sort((a, b) => a.slug.localeCompare(b.slug))
}
