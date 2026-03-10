import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { existsSync, readFileSync, readdirSync, writeFileSync } from 'node:fs'
import { resolve, join } from 'node:path'

function sitemapPlugin() {
  return {
    name: 'generate-sitemap',
    closeBundle() {
      // Read blog index to extract slugs
      const indexPath = resolve(import.meta.dirname, 'src/content/blog/index.ts')
      if (!existsSync(indexPath)) return

      const indexContent = readFileSync(indexPath, 'utf-8')
      const slugRegex = /slug:\s*'([^']+)'/g
      const slugs: string[] = []
      let match
      while ((match = slugRegex.exec(indexContent)) !== null) {
        slugs.push(match[1])
      }

      // Also extract from individual .ts files in the blog directory
      const blogDir = resolve(import.meta.dirname, 'src/content/blog')
      for (const file of readdirSync(blogDir)) {
        if (file.endsWith('.ts') && file !== 'index.ts') {
          const content = readFileSync(join(blogDir, file), 'utf-8')
          const m = content.match(/slug:\s*'([^']+)'/)
          if (m && !slugs.includes(m[1])) slugs.push(m[1])
        }
      }

      const today = new Date().toISOString().split('T')[0]
      const urls = [
        { loc: 'https://atlas-intel-ui-two.vercel.app/', priority: '1.0', changefreq: 'weekly' },
        { loc: 'https://atlas-intel-ui-two.vercel.app/blog', priority: '0.9', changefreq: 'daily' },
        ...slugs.map(slug => ({
          loc: `https://atlas-intel-ui-two.vercel.app/blog/${slug}`,
          priority: '0.7',
          changefreq: 'monthly',
        })),
      ]

      const xml = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${urls.map(u => `  <url>
    <loc>${u.loc}</loc>
    <lastmod>${today}</lastmod>
    <changefreq>${u.changefreq}</changefreq>
    <priority>${u.priority}</priority>
  </url>`).join('\n')}
</urlset>
`
      const distDir = resolve(import.meta.dirname, 'dist')
      if (existsSync(distDir)) {
        writeFileSync(join(distDir, 'sitemap.xml'), xml)
        console.log(`  Sitemap generated with ${urls.length} URLs`)
      }
    },
  }
}

export default defineConfig({
  plugins: [react(), sitemapPlugin()],
  server: {
    host: true,
    port: 5175,
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
