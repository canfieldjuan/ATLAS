import type { Metadata } from "next";
import Providers from "@/components/Providers";
import "./globals.css";

export const metadata: Metadata = {
  title: "Behavioral Pressure Pipeline — Atlas",
  description:
    "Linguistic shift monitoring and sector disruption prediction system",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased bg-obsidian text-foreground font-sans">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
