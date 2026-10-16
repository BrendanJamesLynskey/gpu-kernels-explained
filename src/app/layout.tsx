/**
 * Root layout for the App Router.
 *
 * Server Component, copied from transformer-explainer's layout (via the companion sites): HTML
 * scaffold, the global stylesheet and the site header. Dark mode follows
 * the system setting (`darkMode: "media"` in tailwind.config.ts).
 */
import type { Metadata } from "next";
import type { ReactNode } from "react";

import { SiteHeader } from "@/components/ui/SiteHeader";
import { SITE_URL } from "@/lib/site";
import { withShare } from "@/lib/share";

import "./globals.css";

export const metadata: Metadata = withShare("/", {
  metadataBase: new URL(SITE_URL),
  title: {
    default: "GPU Kernels Explained",
    template: "%s · GPU Kernels Explained",
  },
  description:
    "How a GPU actually executes the maths: the memory hierarchy, the roofline, warps and divergence, coalescing, bank conflicts and occupancy, then GEMM, reductions, softmax and FlashAttention, split-K and overlap, quantised kernels and the Hopper and Blackwell tensor-core pipeline (TMA, tcgen05, Tensor Memory, FP4), each animated by a tested execution model.",
});

export default function RootLayout({
  children,
}: {
  children: ReactNode;
}): JSX.Element {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen font-sans antialiased">
        <SiteHeader />
        {children}
      </body>
    </html>
  );
}
