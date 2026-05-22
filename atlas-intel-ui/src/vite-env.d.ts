/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE?: string
  readonly VITE_PUBLIC_LANDING_PAGE_SITEMAP_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
