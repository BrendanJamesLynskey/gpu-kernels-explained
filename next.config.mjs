/**
 * Next.js config.
 *
 * MDX content is loaded at runtime via `next-mdx-remote` (as in
 * transformer-explainer), so we deliberately do NOT use `@next/mdx` for
 * page-level MDX compilation. Every page is statically rendered: there is
 * no database, no auth and no API route.
 *
 * @type {import('next').NextConfig}
 */
const nextConfig = {
  // Static export (brief vercel-isr-01): every page is built to out/ and served as plain files,
  // so Vercel serves them from its CDN instead of the ISR cache (which meters ISR Reads).
  output: "export",
  images: { unoptimized: true },
  reactStrictMode: true,
  experimental: {
    typedRoutes: true,
  },
};

export default nextConfig;
