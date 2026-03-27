import type { Metadata } from "next";
import { SITE_URL } from "@/lib/constants";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: {
    default: "Churn Signals | B2B Vendor Churn Intelligence",
    template: "%s | Churn Signals",
  },
  description:
    "Data-driven churn intelligence for B2B software. Review analysis, displacement tracking, and competitive insights across 16 platforms.",
  openGraph: {
    type: "website",
    siteName: "Churn Signals",
    locale: "en_US",
    images: [
      {
        url: `${SITE_URL}/og-default.png`,
        width: 1200,
        height: 630,
        alt: "Churn Signals - B2B Vendor Churn Intelligence",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
  },
  robots: {
    index: true,
    follow: true,
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
