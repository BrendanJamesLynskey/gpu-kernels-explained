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
  reactStrictMode: true,
  experimental: {
    typedRoutes: true,
  },
};

export default nextConfig;
