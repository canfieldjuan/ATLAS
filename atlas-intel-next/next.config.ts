import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Proxy /api to the Atlas backend in development
  async rewrites() {
    const backendUrl =
      process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
    return [
      {
        source: "/api/:path*",
        destination: `${backendUrl}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
