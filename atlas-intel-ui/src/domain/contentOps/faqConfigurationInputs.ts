export const FAQ_MARKDOWN_OUTPUT = 'faq_markdown'
export const FAQ_DEFLECTION_REPORT_CONFIGURATION_OUTPUT = 'faq_deflection_report'

export const FAQ_CONFIGURATION_OUTPUTS = [
  FAQ_MARKDOWN_OUTPUT,
  FAQ_DEFLECTION_REPORT_CONFIGURATION_OUTPUT,
] as const

export function faqConfigurationInputsSelected(
  outputs: readonly string[],
): boolean {
  return outputs.some((output) =>
    FAQ_CONFIGURATION_OUTPUTS.includes(
      output as (typeof FAQ_CONFIGURATION_OUTPUTS)[number],
    ),
  )
}
