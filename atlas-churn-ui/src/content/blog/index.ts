import azureVsSalesforce202603 from './azure-vs-salesforce-2026-03'
import b2bSoftwareLandscape202603 from './b2b-software-landscape-2026-03'
import bestB2bSoftwareFor1000202603 from './best-b2b-software-for-1000-2026-03'
import hubspotDeepDive202603 from './hubspot-deep-dive-2026-03'
import jiraVsTrello202603 from './jira-vs-trello-2026-03'
import notionVsSalesforce202603 from './notion-vs-salesforce-2026-03'
import switchToClickup202603 from './switch-to-clickup-2026-03'
import switchToShopify202603 from './switch-to-shopify-2026-03'
import topComplaintEveryB2bSoftware202603 from './top-complaint-every-b2b-software-2026-03'
import whyTeamsLeaveAzure202603 from './why-teams-leave-azure-2026-03'
import b2bSoftwareLandscape202604 from './b2b-software-landscape-2026-04'
import copperDeepDive202604 from './copper-deep-dive-2026-04'
import zoomDeepDive202604 from './zoom-deep-dive-2026-04'

type ChartValue = string | number | null | undefined
type ChartDatum = Record<string, ChartValue>

interface ChartSeries {
  dataKey: string
  color?: string
}

interface ChartConfig {
  bars?: ChartSeries[]
  x_key?: string
  [key: string]: unknown
}

interface AffiliatePartner {
  name?: string
  product_name?: string
  slug?: string
}

interface BlogDataContext {
  affiliate_url?: string
  affiliate_partner?: AffiliatePartner
  [key: string]: unknown
}

export interface ChartSpec {
  chart_id: string
  chart_type: 'bar' | 'horizontal_bar' | 'radar' | 'line'
  title: string
  data: ChartDatum[]
  config: ChartConfig
}

export interface FaqItem {
  question: string
  answer: string
}

export interface BlogPost {
  slug: string
  title: string
  description: string
  date: string
  author: string
  tags: string[]
  content: string
  charts?: ChartSpec[]
  topic_type?: string
  data_context?: BlogDataContext
  seo_title?: string
  seo_description?: string
  target_keyword?: string
  secondary_keywords?: string[]
  faq?: FaqItem[]
  related_slugs?: string[]
  cta?: {
    headline: string
    body: string
    button_text: string
    report_type: string
    vendor_filter?: string | null
    category_filter?: string | null
  } | null
}

export const POSTS: BlogPost[] = [
  azureVsSalesforce202603,
  b2bSoftwareLandscape202603,
  bestB2bSoftwareFor1000202603,
  hubspotDeepDive202603,
  jiraVsTrello202603,
  notionVsSalesforce202603,
  switchToClickup202603,
  switchToShopify202603,
  topComplaintEveryB2bSoftware202603,
  whyTeamsLeaveAzure202603,
  b2bSoftwareLandscape202604,
  copperDeepDive202604,
  zoomDeepDive202604,
]