/**
 * Sharing and first impression (brief 39): the landing page's start button, its "How this is
 * checked" strip (whose link lands on a real About section) and its hero figure above the fold
 * on a phone; link-preview tags on the home page and a chapter, with cards that are 1200 × 630
 * PNGs; the favicon and Apple icon; and a sitemap listing every chapter, named by robots.txt.
 */
import {
  expect,
  test,
  type APIRequestContext,
  type Page,
} from "@playwright/test";

import { SECTIONS } from "@/lib/mdx/sections";

const FIRST = `/learn/${SECTIONS[0]!.slug}`;
const CHAPTERS: readonly string[] = SECTIONS.map((s) => `/learn/${s.slug}`);
const HOME = "/";
/** An ordinary page, which carries the site's card. */
const OTHER = "/about";

async function meta(page: Page, key: string): Promise<string | null> {
  return page
    .locator(`meta[property="${key}"], meta[name="${key}"]`)
    .first()
    .getAttribute("content");
}

/** The path (and query) of an absolute production URL, to fetch from the server under test. */
const local = (u: string) => {
  const x = new URL(u);
  return x.pathname + x.search;
};

async function pngSize(request: APIRequestContext, u: string) {
  const res = await request.get(local(u));
  expect(res.status()).toBe(200);
  expect(res.headers()["content-type"]).toBe("image/png");
  const b = await res.body();
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

test("the landing page: start button, checked strip, figure above the fold on a phone", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(HOME);
  const start = page.locator("[data-start-button]");
  await expect(start).toHaveText("Start with chapter 1 →");
  await expect(start).toHaveAttribute("href", FIRST);
  const fig = page.locator("[data-hero-figure]").first();
  await expect(fig).toBeVisible();
  const box = (await fig.boundingBox())!;
  expect(box.y, "hero figure starts on the first screen").toBeLessThan(
    844 - 160,
  );
  const strip = page.locator("[data-checked-strip]");
  await expect(strip).toContainText("How this is checked");
  const href = (await strip
    .locator("[data-checked-link]")
    .getAttribute("href"))!;
  const [path, id] = href.split("#");
  expect(id, "the strip links to a section").toBeTruthy();
  await page.goto(path!);
  await expect(page.locator(`[id="${id}"]`)).toHaveCount(1);
});

for (const path of [HOME, FIRST, OTHER]) {
  test(`${path} carries Open Graph and Twitter cards`, async ({
    page,
    request,
  }) => {
    await page.goto(path);
    expect(await meta(page, "og:title")).toBeTruthy();
    expect(await meta(page, "og:description")).toBeTruthy();
    expect(await meta(page, "og:site_name")).toBeTruthy();
    expect(new URL((await meta(page, "og:url"))!).pathname).toBe(path);
    expect(await meta(page, "twitter:card")).toBe("summary_large_image");
    for (const key of ["og:image", "twitter:image"]) {
      const u = (await meta(page, key))!;
      expect(u, key).toMatch(/^https:\/\//);
      expect(await pngSize(request, u)).toEqual({ w: 1200, h: 630 });
      // a chapter has its own card
      if (path === FIRST && /^\/(learn|training)\//.test(FIRST))
        expect(new URL(u).pathname).toContain(FIRST);
    }
  });
}

test("the favicon and the Apple icon", async ({ page, request }) => {
  await page.goto(HOME);
  const icon = (await page
    .locator('link[rel="icon"]')
    .first()
    .getAttribute("href"))!;
  expect(icon).toMatch(/^\/icon\.svg/);
  const svg = await request.get(icon);
  expect(svg.headers()["content-type"]).toContain("image/svg+xml");
  const apple = (await page
    .locator('link[rel="apple-touch-icon"]')
    .getAttribute("href"))!;
  expect(await pngSize(request, `https://x${apple}`)).toEqual({
    w: 180,
    h: 180,
  });
});

test("the sitemap lists every chapter, and robots.txt names it", async ({
  request,
}) => {
  const xml = await (await request.get("/sitemap.xml")).text();
  for (const c of CHAPTERS) expect(xml).toContain(`${c}</loc>`);
  const robots = await (await request.get("/robots.txt")).text();
  expect(robots).toContain("Allow: /");
  expect(robots).toMatch(/Sitemap: https:\/\/\S+\/sitemap\.xml/);
});
