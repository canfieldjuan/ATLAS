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
import b2bSoftwareLandscape202603 from './b2b-software-landscape-2026-03'
import topComplaintEveryB2bSoftware202603 from './top-complaint-every-b2b-software-2026-03'
import bestB2bSoftwareFor51200202603 from './best-b2b-software-for-51-200-2026-03'
import azureVsNotion202603 from './azure-vs-notion-2026-03'
import azureVsShopify202603 from './azure-vs-shopify-2026-03'
import azureVsCrowdstrike202603 from './azure-vs-crowdstrike-2026-03'
import notionVsShopify202603 from './notion-vs-shopify-2026-03'
import crowdstrikeVsNotion202603 from './crowdstrike-vs-notion-2026-03'
import crowdstrikeVsShopify202603 from './crowdstrike-vs-shopify-2026-03'
import realCostOfHubspot202603 from './real-cost-of-hubspot-2026-03'
import whyTeamsLeaveFortinet202603 from './why-teams-leave-fortinet-2026-03'
import hubspotDeepDive202603 from './hubspot-deep-dive-2026-03'
import migrationFromMagento202603 from './migration-from-magento-2026-03'
import intercomAlternatives202603 from './intercom-alternatives-2026-03'
import migrationFromFortinet202603 from './migration-from-fortinet-2026-03'
import migrationFromMondaycom202603 from './migration-from-mondaycom-2026-03'
import migrationFromRingcentral202603 from './migration-from-ringcentral-2026-03'
import migrationFromPipedrive202603 from './migration-from-pipedrive-2026-03'
import helpScoutChurnReport202603 from './help-scout-churn-report-2026-03'
import insightlyChurnReport202603 from './insightly-churn-report-2026-03'
import bestCrmFor51200202603 from './best-crm-for-51-200-2026-03'
import realCostOfMailchimp202603 from './real-cost-of-mailchimp-2026-03'
import mailchimpVsIntercom202603 from './mailchimp-vs-intercom-2026-03'
import salesforceVsHubspot202603 from './salesforce-vs-hubspot-2026-03'
import asanaVsMondaycom202603 from './asana-vs-mondaycom-2026-03'
import crowdstrikeVsSentinelone202603 from './crowdstrike-vs-sentinelone-2026-03'
import paloAltoNetworksVsMicrosoftDefenderForEndpoint202603 from './palo-alto-networks-vs-microsoft-defender-for-endpoint-2026-03'
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
  b2bSoftwareLandscape202603,
  topComplaintEveryB2bSoftware202603,
  bestB2bSoftwareFor51200202603,
  azureVsNotion202603,
  azureVsShopify202603,
  azureVsCrowdstrike202603,
  notionVsShopify202603,
  crowdstrikeVsNotion202603,
  crowdstrikeVsShopify202603,
  realCostOfHubspot202603,
  whyTeamsLeaveFortinet202603,
  hubspotDeepDive202603,
  migrationFromMagento202603,
  intercomAlternatives202603,
  migrationFromFortinet202603,
  migrationFromMondaycom202603,
  migrationFromRingcentral202603,
  migrationFromPipedrive202603,
  helpScoutChurnReport202603,
  insightlyChurnReport202603,
  bestCrmFor51200202603,
  realCostOfMailchimp202603,
  mailchimpVsIntercom202603,
  salesforceVsHubspot202603,
  asanaVsMondaycom202603,
  crowdstrikeVsSentinelone202603,
  paloAltoNetworksVsMicrosoftDefenderForEndpoint202603,
]