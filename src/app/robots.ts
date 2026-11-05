/**
 * /robots.txt: normal search engines and the AI assistants' crawlers may read everything; the SEO
 * scrapers that were driving this site's traffic are disallowed (brief vercel-isr-01). Points at the sitemap.
 */
import type { MetadataRoute } from "next";

import { SITE_URL } from "@/lib/site";

export const dynamic = "force-static";

/** Crawlers seen hammering the site in the firewall logs; none of them send readers. */
const BLOCKED = [
  "SemrushBot",
  "AhrefsBot",
  "MJ12bot",
  "Bytespider",
  "PetalBot",
  "ExaBot",
];

export default function robots(): MetadataRoute.Robots {
  return {
    rules: [
      { userAgent: "*", allow: "/" },
      { userAgent: BLOCKED, disallow: "/" },
    ],
    sitemap: `${SITE_URL}/sitemap.xml`,
  };
}
