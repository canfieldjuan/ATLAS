import type { BlogPost as BlogPostType } from '../content/blog'

export interface ResolvedBlogArticleCta {
  mode: 'affiliate' | 'generic'
  url: string
  headline: string
  body: string
  buttonText: string
  name: string
}

export function resolveBlogArticleCta(post: BlogPostType): ResolvedBlogArticleCta {
  const url = String(post.data_context?.affiliate_url || '').trim()
  const partner = post.data_context?.affiliate_partner
  const partnerName = String(partner?.name || partner?.product_name || '').trim()
  const storedCta = post.cta || null

  if (url && partnerName) {
    return {
      mode: 'affiliate',
      url,
      name: partnerName,
      headline: storedCta?.headline?.trim() || `Want to try ${partnerName}?`,
      body:
        storedCta?.body?.trim() ||
        `Based on the data, ${partnerName} may be worth evaluating for your team. Start free, no credit card required.`,
      buttonText: storedCta?.button_text?.trim() || `Try ${partnerName} Free`,
    }
  }
  return {
    mode: 'generic',
    url: String(post.data_context?.booking_url || 'https://cal.com/atlas-intel/15min'),
    name: '',
    headline:
      storedCta?.headline?.trim() || 'Want churn intelligence on vendors in this space?',
    body:
      storedCta?.body?.trim() ||
      'We track real-time switching signals across 200+ B2B software vendors. See which accounts are actively evaluating alternatives.',
    buttonText:
      storedCta?.button_text?.trim() || 'Book a 15-Min Intel Briefing',
  }
}
