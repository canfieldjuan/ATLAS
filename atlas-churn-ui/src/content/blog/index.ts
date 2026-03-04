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
import bestMarketingAutomationForUnknown202603 from './best-marketing-automation-for-unknown-2026-03'
import communicationLandscape202603 from './communication-landscape-2026-03'
import topComplaintEveryECommerce202603 from './top-complaint-every-e-commerce-2026-03'
import whyTeamsLeaveAws202603 from './why-teams-leave-aws-2026-03'
import realCostOfKlaviyo202603 from './real-cost-of-klaviyo-2026-03'
import topComplaintEveryMarketingAutomation202603 from './top-complaint-every-marketing-automation-2026-03'
import realCostOfZoom202603 from './real-cost-of-zoom-2026-03'
import helpdeskLandscape202603 from './helpdesk-landscape-2026-03'
import bestCommunicationFor150202603 from './best-communication-for-1-50-2026-03'
import whyTeamsLeaveAsana202603 from './why-teams-leave-asana-2026-03'
import realCostOfSlack202603 from './real-cost-of-slack-2026-03'
import bestDataAnalyticsFor150202603 from './best-data-analytics-for-1-50-2026-03'
import googleCloudDeepDive202603 from './google-cloud-deep-dive-2026-03'
import whyTeamsLeaveSlack202603 from './why-teams-leave-slack-2026-03'
import bestCrmFor150202603 from './best-crm-for-1-50-2026-03'
import topComplaintEveryCrm202603 from './top-complaint-every-crm-2026-03'
import intercomDeepDive202603 from './intercom-deep-dive-2026-03'
import bestHrHcmFor51200202603 from './best-hr-hcm-for-51-200-2026-03'
import bestProjectManagementFor150202603 from './best-project-management-for-1-50-2026-03'
import migrationFromZoom202603 from './migration-from-zoom-2026-03'
import migrationFromClickup202603 from './migration-from-clickup-2026-03'
import migrationFromAsana202603 from './migration-from-asana-2026-03'
import migrationFromSlack202603 from './migration-from-slack-2026-03'
import migrationFromAws202603 from './migration-from-aws-2026-03'
import bestMarketingAutomationFor150202603 from './best-marketing-automation-for-1-50-2026-03'
import bestCrmForUnknown202603 from './best-crm-for-unknown-2026-03'
import magentoDeepDive202603 from './magento-deep-dive-2026-03'
import topComplaintEveryCommunication202603 from './top-complaint-every-communication-2026-03'
import activecampaignDeepDive202603 from './activecampaign-deep-dive-2026-03'
import topComplaintEveryHelpdesk202603 from './top-complaint-every-helpdesk-2026-03'
import dataAnalyticsLandscape202603 from './data-analytics-landscape-2026-03'
import hrHcmLandscape202603 from './hr-hcm-landscape-2026-03'
import bestCybersecurityFor51200202603 from './best-cybersecurity-for-51-200-2026-03'
import topComplaintEveryDataAnalytics202603 from './top-complaint-every-data-analytics-2026-03'
import topComplaintEveryHrHcm202603 from './top-complaint-every-hr-hcm-2026-03'
import cybersecurityLandscape202603 from './cybersecurity-landscape-2026-03'
import topComplaintEveryCybersecurity202603 from './top-complaint-every-cybersecurity-2026-03'
import jiraDeepDive202603 from './jira-deep-dive-2026-03'
import trelloDeepDive202603 from './trello-deep-dive-2026-03'
import smartsheetDeepDive202603 from './smartsheet-deep-dive-2026-03'
import pipedriveDeepDive202603 from './pipedrive-deep-dive-2026-03'
import klaviyoDeepDive202603 from './klaviyo-deep-dive-2026-03'
import digitaloceanDeepDive202603 from './digitalocean-deep-dive-2026-03'
import gustoDeepDive202603 from './gusto-deep-dive-2026-03'
import brevoDeepDive202603 from './brevo-deep-dive-2026-03'
import getresponseDeepDive202603 from './getresponse-deep-dive-2026-03'
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
  bestMarketingAutomationForUnknown202603,
  communicationLandscape202603,
  topComplaintEveryECommerce202603,
  whyTeamsLeaveAws202603,
  realCostOfKlaviyo202603,
  topComplaintEveryMarketingAutomation202603,
  realCostOfZoom202603,
  helpdeskLandscape202603,
  bestCommunicationFor150202603,
  whyTeamsLeaveAsana202603,
  realCostOfSlack202603,
  bestDataAnalyticsFor150202603,
  googleCloudDeepDive202603,
  whyTeamsLeaveSlack202603,
  bestCrmFor150202603,
  topComplaintEveryCrm202603,
  intercomDeepDive202603,
  bestHrHcmFor51200202603,
  bestProjectManagementFor150202603,
  migrationFromZoom202603,
  migrationFromClickup202603,
  migrationFromAsana202603,
  migrationFromSlack202603,
  migrationFromAws202603,
  bestMarketingAutomationFor150202603,
  bestCrmForUnknown202603,
  magentoDeepDive202603,
  topComplaintEveryCommunication202603,
  activecampaignDeepDive202603,
  topComplaintEveryHelpdesk202603,
  dataAnalyticsLandscape202603,
  hrHcmLandscape202603,
  bestCybersecurityFor51200202603,
  topComplaintEveryDataAnalytics202603,
  topComplaintEveryHrHcm202603,
  cybersecurityLandscape202603,
  topComplaintEveryCybersecurity202603,
  jiraDeepDive202603,
  trelloDeepDive202603,
  smartsheetDeepDive202603,
  pipedriveDeepDive202603,
  klaviyoDeepDive202603,
  digitaloceanDeepDive202603,
  gustoDeepDive202603,
  brevoDeepDive202603,
  getresponseDeepDive202603,
].sort((a, b) => b.date.localeCompare(a.date))
