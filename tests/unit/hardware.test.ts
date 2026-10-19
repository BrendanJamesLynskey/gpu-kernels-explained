/**
 * The Hardware track's start page (brief 44): four sites in order, and every
 * glossary link on this site names a chapter and an anchor that exist (the
 * other sites' anchors are in their own chapters, checked by their tests).
 */
import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

import {
  HARDWARE_GLOSSARY,
  HARDWARE_SITES,
  KERNELS,
} from "@/lib/learn/hardware";
import { SECTIONS } from "@/lib/mdx/sections";

const DIR = path.resolve(__dirname, "../../content/chapters");

describe("the Hardware start page", () => {
  it("lists the four sites in reading order", () => {
    expect(HARDWARE_SITES.map((s) => s.name)).toEqual([
      "Kernels",
      "Systolic",
      "Accelerators",
      "Racks",
    ]);
  });

  it("links every term to a chapter section, and this site's exist", () => {
    expect(HARDWARE_GLOSSARY.length).toBeGreaterThanOrEqual(30);
    const slugs = new Set<string>(SECTIONS.map((s) => s.slug));
    for (const g of HARDWARE_GLOSSARY) {
      const m = /^(https:\/\/[^/]+)\/learn\/([\w-]+)#([\w-]+)$/.exec(g.href);
      expect(m, g.href).not.toBeNull();
      if (m![1] !== KERNELS) continue;
      expect(slugs.has(m![2]!), g.href).toBe(true);
      const src = readFileSync(path.join(DIR, `${m![2]}.mdx`), "utf8");
      expect(src, g.href).toContain(`id="${m![3]}"`);
    }
  });
});
