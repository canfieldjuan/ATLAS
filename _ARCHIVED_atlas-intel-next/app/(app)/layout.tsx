"use client";

import { Suspense } from "react";
import AuthProvider from "@/lib/auth/AuthContext";
import Layout from "@/components/Layout";

export default function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <AuthProvider>
      <Layout>
        <Suspense>{children}</Suspense>
      </Layout>
    </AuthProvider>
  );
}
