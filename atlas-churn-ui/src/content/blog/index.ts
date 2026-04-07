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
import closeVsZohoCrm202604 from './close-vs-zoho-crm-2026-04'
import switchToWoocommerce202604 from './switch-to-woocommerce-2026-04'
import tableauDeepDive202604 from './tableau-deep-dive-2026-04'
import magentoDeepDive202604 from './magento-deep-dive-2026-04'
import intercomDeepDive202604 from './intercom-deep-dive-2026-04'
import zohoCrmDeepDive202604 from './zoho-crm-deep-dive-2026-04'
import workdayDeepDive202604 from './workday-deep-dive-2026-04'
import realCostOfShopify202604 from './real-cost-of-shopify-2026-04'
import salesforceDeepDive202604 from './salesforce-deep-dive-2026-04'
import switchToSalesforce202604 from './switch-to-salesforce-2026-04'
import hubspotDeepDive202604 from './hubspot-deep-dive-2026-04'
import basecampDeepDive202604 from './basecamp-deep-dive-2026-04'
import amazonWebServicesDeepDive202604 from './amazon-web-services-deep-dive-2026-04'
import fortinetDeepDive202604 from './fortinet-deep-dive-2026-04'
import switchToAsana202604 from './switch-to-asana-2026-04'

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
  closeVsZohoCrm202604,
  switchToWoocommerce202604,
  tableauDeepDive202604,
  magentoDeepDive202604,
  intercomDeepDive202604,
  zohoCrmDeepDive202604,
  workdayDeepDive202604,
  realCostOfShopify202604,
  salesforceDeepDive202604,
  switchToSalesforce202604,
  hubspotDeepDive202604,
  basecampDeepDive202604,
  amazonWebServicesDeepDive202604,
  fortinetDeepDive202604,
  switchToAsana202604,
]