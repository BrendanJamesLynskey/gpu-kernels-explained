/**
 * The chapters' MDX: every ```ts block is cut from the file the chapter
 * names (whitespace-collapsed, because Prettier reformats MDX code blocks);
 * every CUDA block is labelled as not compiled here; every equation
 * compiles in KaTeX; internal links point at real pages; deck links point
 * at the owner's NVIDIA GPU and CUDA decks with a slide anchor.
 */
import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";

import katex from "katex";
import { describe, expect, it } from "vitest";

import { SECTIONS } from "@/lib/mdx/sections";

const ROOT = path.join(__dirname, "../..");
const DIR = path.join(ROOT, "content/chapters");
const FILES = readdirSync(DIR).filter((f) => f.endsWith(".mdx"));
const squash = (s: string) => s.replace(/\s+/g, " ").trim();

describe("chapter files", () => {
  it("there is one per section, in order", () => {
    expect(FILES.sort()).toEqual(SECTIONS.map((s) => `${s.slug}.mdx`));
  });
});

for (const f of FILES) {
  const src = readFileSync(path.join(DIR, f), "utf8");
  describe(f, () => {
    it("opens with its animation (the hero comes before any prose)", () => {
      expect(src.trimStart()).toMatch(/^<[A-Z][a-zA-Z]+Widget>/);
    });

    it("cuts every TypeScript block from the model source", () => {
      const model = squash(
        readFileSync(path.join(ROOT, "src/lib/gpu/model.ts"), "utf8"),
      );
      const blocks = [...src.matchAll(/```ts\n([\s\S]*?)```/g)].map(
        (m) => m[1]!,
      );
      expect(blocks.length).toBeGreaterThan(0);
      for (const b of blocks) expect(model, b).toContain(squash(b));
    });

    it("labels every CUDA block as not compiled here", () => {
      for (const m of src.matchAll(/```cuda\n/g)) {
        const before = src.slice(Math.max(0, m.index! - 300), m.index);
        expect(before).toMatch(/not compiled by this site's CI/);
      }
    });

    it("compiles every animation equation", () => {
      for (const m of src.matchAll(/tex="([^"]+)"/g)) {
        expect(() =>
          katex.renderToString(m[1]!, {
            displayMode: true,
            throwOnError: true,
            strict: "ignore",
            trust: (ctx) => ctx.command === "\\htmlClass",
          }),
        ).not.toThrow();
      }
    });

    it("links only to real pages and to deck slides", () => {
      const slugs = new Set<string>(SECTIONS.map((s) => s.slug));
      for (const m of src.matchAll(/\]\((\/[^)]+)\)/g)) {
        const href = m[1]!;
        if (href.startsWith("/learn/"))
          expect(slugs.has(href.slice(7)), href).toBe(true);
        else expect(["/gpus", "/about", "/learn"]).toContain(href);
      }
      for (const m of src.matchAll(
        /https:\/\/brendanjameslynskey\.github\.io\/([A-Za-z0-9_]+)\/(#slide-\d\d)?/g,
      )) {
        expect(m[1], m[0]).toMatch(/^(NVIDIA_GPU|CUDA)_\d\d_/);
        expect(m[2], m[0]).toBeDefined();
      }
    });
  });
}
