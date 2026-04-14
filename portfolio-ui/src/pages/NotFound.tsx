import { Link } from "react-router-dom";
import { SeoHead } from "@/components/seo/SeoHead";

export default function NotFound() {
  return (
    <>
      <SeoHead
        meta={{
          title: "Page Not Found",
          description:
            "The requested page could not be found. Return to the homepage or browse available AI systems architecture portfolio pages.",
          canonicalPath: "/404",
          noindex: true,
        }}
      />

      <section className="mx-auto flex min-h-[70vh] max-w-3xl items-center px-6 py-16">
        <div className="rounded-xl border border-surface-700/50 bg-surface-800/30 p-8 text-center">
          <p className="mb-2 text-sm font-semibold text-primary-400">404</p>
          <h1 className="mb-4 text-3xl font-bold text-white">
            This page doesn&apos;t exist
          </h1>
          <p className="mb-8 text-surface-200/80">
            If you followed an old link, the route may have changed. Start back
            at the homepage and browse from there.
          </p>
          <Link
            to="/"
            className="inline-flex rounded-full bg-primary-500 px-5 py-2 text-sm font-semibold text-surface-900 transition-colors hover:brightness-110"
          >
            Go Home
          </Link>
        </div>
      </section>
    </>
  );
}
