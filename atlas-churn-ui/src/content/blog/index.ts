import freshsalesDeepDive202603 from './freshsales-deep-dive-2026-03'
import happyfoxDeepDive202603 from './happyfox-deep-dive-2026-03'
import migrationFromSentinelone202603 from './migration-from-sentinelone-2026-03'
import realCostOfAmazonWebServices202603 from './real-cost-of-amazon-web-services-2026-03'
import amazonWebServicesVsGoogleCloudPlatform202603 from './amazon-web-services-vs-google-cloud-platform-2026-03'
import amazonWebServicesDeepDive202603 from './amazon-web-services-deep-dive-2026-03'
import googleCloudPlatformDeepDive202603 from './google-cloud-platform-deep-dive-2026-03'
import bamboohrVsRippling202603 from './bamboohr-vs-rippling-2026-03'
import realCostOfSalesforce202603 from './real-cost-of-salesforce-2026-03'
import jiraVsTeamwork202603 from './jira-vs-teamwork-2026-03'
import freshsalesVsSalesforce202603 from './freshsales-vs-salesforce-2026-03'
import insightlyVsSalesforce202603 from './insightly-vs-salesforce-2026-03'
import mailchimpAlternatives202603 from './mailchimp-alternatives-2026-03'
import mailchimpChurnReport202603 from './mailchimp-churn-report-2026-03'
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
  freshsalesDeepDive202603,
  happyfoxDeepDive202603,
  migrationFromSentinelone202603,
  realCostOfAmazonWebServices202603,
  amazonWebServicesVsGoogleCloudPlatform202603,
  amazonWebServicesDeepDive202603,
  googleCloudPlatformDeepDive202603,
  bamboohrVsRippling202603,
  realCostOfSalesforce202603,
  jiraVsTeamwork202603,
  freshsalesVsSalesforce202603,
  insightlyVsSalesforce202603,
  mailchimpAlternatives202603,
  mailchimpChurnReport202603,
]