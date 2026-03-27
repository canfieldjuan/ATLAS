"use client";

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import AtlasRobotLogo from '@/components/AtlasRobotLogo'

const NAV_LINKS = [
  { label: 'Blog', href: '/blog' },
]

export default function PublicLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname()

  return (
    <div className="min-h-screen bg-slate-900 text-white flex flex-col">
      {/* Nav */}
      <nav className="flex items-center justify-between max-w-6xl mx-auto w-full px-6 py-5">
        <div className="flex items-center gap-6">
          <Link href="/landing" className="flex items-center gap-2">
            <AtlasRobotLogo className="h-7 w-7" />
            <span className="text-xl font-bold">Atlas Intelligence</span>
          </Link>
          {NAV_LINKS.map(link => (
            <Link
              key={link.href}
              href={link.href}
              className={`text-sm font-medium transition-colors ${
                pathname.startsWith(link.href)
                  ? 'text-white'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              {link.label}
            </Link>
          ))}
        </div>
        <div className="flex items-center gap-4">
          <Link href="/login" className="text-sm text-slate-400 hover:text-white transition-colors">
            Sign in
          </Link>
          <Link
            href="/signup?product=consumer"
            className="text-sm px-4 py-2 bg-cyan-600 hover:bg-cyan-500 rounded-lg font-medium transition-colors"
          >
            Start Free Trial
          </Link>
        </div>
      </nav>

      {/* Page content */}
      <main className="flex-1">{children}</main>

      {/* Footer */}
      <footer className="border-t border-slate-800 py-8">
        <div className="max-w-6xl mx-auto px-6 flex items-center justify-between text-sm text-slate-500">
          <div className="flex items-center gap-2">
            <AtlasRobotLogo className="h-4 w-4" />
            <span>Atlas Intelligence</span>
          </div>
          <div className="flex items-center gap-4">
            <Link href="/blog" className="hover:text-slate-300 transition-colors">Blog</Link>
            <Link href="/login" className="hover:text-slate-300 transition-colors">Sign in</Link>
            <Link href="/signup?product=consumer" className="hover:text-slate-300 transition-colors">Sign up</Link>
          </div>
        </div>
      </footer>
    </div>
  )
}
