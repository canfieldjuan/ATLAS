import migrationFromFreshdesk202603 from './migration-from-freshdesk-2026-03'
import migrationFromSalesforce202603 from './migration-from-salesforce-2026-03'
import copperVsZohoCrm202603 from './copper-vs-zoho-crm-2026-03'
import basecampVsWrike202603 from './basecamp-vs-wrike-2026-03'
import migrationFromAzure202603 from './migration-from-azure-2026-03'
import migrationFromShopify202603 from './migration-from-shopify-2026-03'
import bestProjectManagementFor51200202603 from './best-project-management-for-51-200-2026-03'
import bestCloudInfrastructureFor150202603 from './best-cloud-infrastructure-for-1-50-2026-03'
import migrationFromWoocommerce202603 from './migration-from-woocommerce-2026-03'
import topComplaintEveryProjectManagement202603 from './top-complaint-every-project-management-2026-03'
import projectManagementLandscape202603 from './project-management-landscape-2026-03'
import bestECommerceFor150202603 from './best-e-commerce-for-1-50-2026-03'
import migrationFromTableau202603 from './migration-from-tableau-2026-03'
import closeDeepDive202603 from './close-deep-dive-2026-03'
import whyTeamsLeaveNotion202603 from './why-teams-leave-notion-2026-03'
import cloudInfrastructureLandscape202603 from './cloud-infrastructure-landscape-2026-03'
import realCostOfNotion202603 from './real-cost-of-notion-2026-03'
import realCostOfAws202603 from './real-cost-of-aws-2026-03'
import notionVsTeamwork202603 from './notion-vs-teamwork-2026-03'
import bestHelpdeskForUnknown202603 from './best-helpdesk-for-unknown-2026-03'
import realCostOfAsana202603 from './real-cost-of-asana-2026-03'
import topComplaintEveryCloudInfrastructure202603 from './top-complaint-every-cloud-infrastructure-2026-03'
import mailchimpDeepDive202603 from './mailchimp-deep-dive-2026-03'
import crmLandscape202603 from './crm-landscape-2026-03'
import eCommerceLandscape202603 from './e-commerce-landscape-2026-03'
import marketingAutomationLandscape202603 from './marketing-automation-landscape-2026-03'
export interface ChartSpec {
  chart_id: string
  chart_type: 'bar' | 'horizontal_bar' | 'radar' | 'line'
  title: string
  data: Record<string, any>[]
  config: Record<string, any>
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
  data_context?: Record<string, any>
}

export const POSTS: BlogPost[] = [
  migrationFromFreshdesk202603,
  migrationFromSalesforce202603,
  copperVsZohoCrm202603,
  basecampVsWrike202603,
  migrationFromAzure202603,
  migrationFromShopify202603,
  bestProjectManagementFor51200202603,
  bestCloudInfrastructureFor150202603,
  migrationFromWoocommerce202603,
  topComplaintEveryProjectManagement202603,
  projectManagementLandscape202603,
  bestECommerceFor150202603,
  migrationFromTableau202603,
  closeDeepDive202603,
  whyTeamsLeaveNotion202603,
  cloudInfrastructureLandscape202603,
  realCostOfNotion202603,
  realCostOfAws202603,
  notionVsTeamwork202603,
  bestHelpdeskForUnknown202603,
  realCostOfAsana202603,
  topComplaintEveryCloudInfrastructure202603,
  mailchimpDeepDive202603,
  crmLandscape202603,
  eCommerceLandscape202603,
  marketingAutomationLandscape202603,
].sort((a, b) => b.date.localeCompare(a.date))
