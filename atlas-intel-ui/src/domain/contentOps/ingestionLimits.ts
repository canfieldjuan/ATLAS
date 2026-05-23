import type { ContentOpsIngestionLimits } from './types'

export interface ContentOpsIngestionFileSelection {
  name: string
  size: number
}

export function contentOpsIngestionFilePreflightError(
  file: ContentOpsIngestionFileSelection,
  limits: ContentOpsIngestionLimits,
): string | null {
  const maxFileBytes = limits.fileUpload.maxFileBytes
  if (
    Number.isFinite(file.size) &&
    Number.isFinite(maxFileBytes) &&
    file.size > maxFileBytes
  ) {
    return `${file.name} is ${formatContentOpsBytes(file.size)}. Upload files must be ${formatContentOpsBytes(maxFileBytes)} or smaller.`
  }
  return null
}

export function formatContentOpsBytes(value: number): string {
  if (!Number.isFinite(value) || value < 0) return 'unknown size'
  if (value < 1024) return `${value} B`
  const kib = value / 1024
  if (kib < 1024) return `${kib.toFixed(1)} KB`
  return `${(kib / 1024).toFixed(1)} MB`
}
