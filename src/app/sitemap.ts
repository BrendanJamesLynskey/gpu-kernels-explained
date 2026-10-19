/**
 * /sitemap.xml: every page and every chapter, at this site's production address.
 */
import type { MetadataRoute } from "next";

import { SECTIONS } from "@/lib/mdx/sections";
import { SITE_URL } from "@/lib/site";

/** The site's other pages (the chapters are added from the catalogue). */
const PAGES = ["/", "/learn", "/hardware", "/gpus", "/about"];

export default function sitemap(): MetadataRoute.Sitemap {
  return [...PAGES, ...SECTIONS.map((s) => `/learn/${s.slug}`)].map((p) => ({
    url: p === "/" ? SITE_URL : `${SITE_URL}${p}`,
  }));
}
