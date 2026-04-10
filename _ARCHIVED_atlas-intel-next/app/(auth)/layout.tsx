"use client";

import { Suspense } from "react";
import AuthProvider from "@/lib/auth/AuthContext";

export default function AuthLayout({ children }: { children: React.ReactNode }) {
  return (
    <AuthProvider>
      <Suspense>
        <div className="min-h-screen bg-slate-900 text-white flex items-center justify-center">
          {children}
        </div>
      </Suspense>
    </AuthProvider>
  );
}
