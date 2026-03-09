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

interface BlogDataContext {
  [key: string]: unknown
}

export interface ChartSpec {
  chart_id: string
  chart_type: 'bar' | 'horizontal_bar' | 'radar' | 'line'
  title: string
  data: ChartDatum[]
  config: ChartConfig
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
}

import amazonReviewMonitoringTools from './amazon-review-monitoring-tools-2026-03'
import migrationComputerAccessoriesPeripherals202603 from './migration-computer-accessories-peripherals-2026-03'
import migrationComputerComponents202603 from './migration-computer-components-2026-03'
import migrationMaintenanceUpkeepRepairs202603 from './migration-maintenance-upkeep-repairs-2026-03'
import migrationNetworkingProducts202603 from './migration-networking-products-2026-03'
import migrationBasicCases202603 from './migration-basic-cases-2026-03'
import migrationDataStorage202603 from './migration-data-storage-2026-03'
import migrationComputersTablets202603 from './migration-computers-tablets-2026-03'
import migrationAccessories202603 from './migration-accessories-2026-03'
import safetyComputerAccessoriesPeripherals202603 from './safety-computer-accessories-peripherals-2026-03'
import safetyStrengthTrainingEquipment202603 from './safety-strength-training-equipment-2026-03'
import safetyElectronics202603 from './safety-electronics-2026-03'
import safetyCycling202603 from './safety-cycling-2026-03'
import safetySkatesSkateboardsScooters202603 from './safety-skates-skateboards-scooters-2026-03'

export const POSTS: BlogPost[] = [
  amazonReviewMonitoringTools,
  migrationComputerAccessoriesPeripherals202603,
  migrationComputerComponents202603,
  migrationMaintenanceUpkeepRepairs202603,
  migrationNetworkingProducts202603,
  migrationBasicCases202603,
  migrationDataStorage202603,
  migrationComputersTablets202603,
  migrationAccessories202603,
  safetyComputerAccessoriesPeripherals202603,
  safetyStrengthTrainingEquipment202603,
  safetyElectronics202603,
  safetyCycling202603,
  safetySkatesSkateboardsScooters202603,
].sort((a, b) => b.date.localeCompare(a.date))
