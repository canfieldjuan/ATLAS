import { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { Menu, X } from "lucide-react";
import type { NavLink } from "@/types";

const NAV_LINKS: NavLink[] = [
  { label: "What We Build", href: "/services" },
  { label: "Projects", href: "/projects" },
  { label: "Systems", href: "/systems" },
  { label: "Insights", href: "/insights" },
  { label: "Framework", href: "/framework" },
  { label: "About", href: "/about" },
];

export function Header() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const location = useLocation();

  return (
    <header className="fixed top-0 left-0 right-0 z-50 border-b border-surface-700/50 bg-surface-900/80 backdrop-blur-xl">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
        <Link to="/" className="flex items-center gap-3 group">
          <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-primary-500 to-accent-cyan flex items-center justify-center text-sm font-bold text-surface-900">
            JC
          </div>
          <span className="font-semibold text-white group-hover:text-primary-400 transition-colors hidden sm:inline">
            Juan Canfield
          </span>
        </Link>

        {/* Desktop nav */}
        <nav className="hidden md:flex items-center gap-8">
          {NAV_LINKS.map((link) => (
            <Link
              key={link.href}
              to={link.href}
              className={`text-sm font-medium transition-colors ${
                location.pathname === link.href
                  ? "text-primary-400"
                  : "text-surface-200 hover:text-white"
              }`}
            >
              {link.label}
            </Link>
          ))}
        </nav>

        {/* Mobile toggle */}
        <button
          className="md:hidden text-surface-200 hover:text-white"
          onClick={() => setMobileOpen(!mobileOpen)}
          aria-label="Toggle menu"
        >
          {mobileOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
      </div>

      {/* Mobile nav */}
      {mobileOpen && (
        <nav className="md:hidden border-t border-surface-700/50 bg-surface-900/95 backdrop-blur-xl">
          {NAV_LINKS.map((link) => (
            <Link
              key={link.href}
              to={link.href}
              onClick={() => setMobileOpen(false)}
              className={`block px-6 py-3 text-sm font-medium border-b border-surface-700/30 transition-colors ${
                location.pathname === link.href
                  ? "text-primary-400 bg-primary-500/5"
                  : "text-surface-200 hover:text-white hover:bg-surface-800/50"
              }`}
            >
              {link.label}
            </Link>
          ))}
        </nav>
      )}
    </header>
  );
}
