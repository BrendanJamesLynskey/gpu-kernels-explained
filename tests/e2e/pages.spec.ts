/**
 * Every page renders at desktop and phone widths, in light and dark mode,
 * with no page errors, no console errors and no horizontal overflow.
 */
import { expect, test, type Page } from "@playwright/test";

const PAGES = [
  "/",
  "/learn",
  "/gpus",
  "/about",
  "/learn/01-memory-hierarchy",
  "/learn/02-roofline",
  "/learn/03-warps-and-divergence",
  "/learn/04-coalescing",
  "/learn/05-bank-conflicts",
  "/learn/06-occupancy",
  "/learn/07-gemm",
  "/learn/08-reductions",
  "/learn/09-softmax-and-flashattention",
  "/learn/10-split-k-and-overlap",
  "/learn/11-quantised-kernels",
];

async function collectErrors(page: Page): Promise<string[]> {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(`console: ${m.text()}`);
  });
  return errors;
}

for (const scheme of ["light", "dark"] as const) {
  for (const width of [1280, 390]) {
    test.describe(`${scheme} @ ${width}px`, () => {
      test.use({ colorScheme: scheme, viewport: { width, height: 900 } });
      for (const path of PAGES) {
        test(`${path} renders cleanly`, async ({ page }) => {
          const errors = await collectErrors(page);
          const res = await page.goto(path);
          expect(res?.status()).toBe(200);
          await expect(page.locator("h1").first()).toBeVisible();
          await page.waitForLoadState("networkidle");
          // chapter interactives load after the page: wait for every one
          await expect(page.locator("[data-pending-widget]")).toHaveCount(0);
          const overflow = await page.evaluate(() => {
            const el = document.scrollingElement!;
            return el.scrollWidth - el.clientWidth;
          });
          expect(overflow, "horizontal overflow (px)").toBeLessThanOrEqual(0);
          const bg = await page.evaluate(
            () => getComputedStyle(document.body).backgroundColor,
          );
          expect(bg).toBe(
            scheme === "dark" ? "rgb(10, 10, 10)" : "rgb(255, 255, 255)",
          );
          expect(errors).toEqual([]);
        });
      }
    });
  }
}

test("the site switch: four tracks, and the brand names the site and the button says where it sits", async ({
  page,
}) => {
  const menu = page.locator("[data-site-switch='menu']");
  const panel = menu.locator("summary + div");
  const site = (key: string) => panel.locator(`[data-site-key='${key}']`);
  const brand = page.locator("header [data-site-brand]");
  for (const width of [1280, 1024, 390]) {
    await page.setViewportSize({ width, height: 800 });
    await page.goto("/");
    // closed: the button reads "<Track> › <Site>", and the brand names the site
    await expect(menu).toBeVisible();
    await expect(menu.locator("summary")).toHaveText(/^Hardware\W+Kernels\W*$/);
    // the brand names the site; the place ("Track › Site") is said once, on the button
    await expect(brand).toHaveText("GPU Kernels Explained");
    await expect(brand.locator("[data-site-place]")).toHaveCount(0);
    await expect(page.locator("header [data-site-place]")).toHaveCount(1);
    if (width === 1280)
      expect((await brand.boundingBox())!.height).toBeLessThan(30);
    await expect(panel).toBeHidden();
    // one click opens every track, in the order Foundations, Hardware, Inference, Agents
    await menu.locator("summary").click();
    await expect(panel).toBeVisible();
    await expect(menu.locator("nav > p:first-child")).toHaveText([
      "Foundations",
      "Hardware",
      "Inference",
      "Agents",
    ]);
    await expect(menu.locator("nav > p:nth-child(2)")).toHaveText([
      "One small model, end to end",
      "The machines underneath, chip to rack",
      "Serving models at scale",
      "Building on models",
    ]);
    // every site, in order, with its name, its one-line description and its address
    const SITES = [
      [
        "decoder",
        "Decoder",
        "How it runs, one forward pass",
        "https://transformer-decoder-explained.vercel.app",
      ],
      [
        "training",
        "Training",
        "How it learns, from one step to many GPUs",
        "https://transformer-training-explained.vercel.app",
      ],
      [
        "kernels",
        "Kernels",
        "How a GPU executes the maths",
        "https://gpu-kernels-explained.vercel.app",
      ],
      [
        "silicon",
        "Systolic",
        "Inside a matrix engine",
        "https://systolic-arrays-explained.vercel.app",
      ],
      [
        "accelerators",
        "Accelerators",
        "The families of AI chips",
        "https://ai-accelerators-explained.vercel.app",
      ],
      [
        "racks",
        "Racks",
        "GPUs, networks, power, cooling at scale",
        "https://ai-racks-explained.vercel.app",
      ],
      [
        "inference",
        "Serving",
        "How real systems generate text",
        "https://llm-inference-explained.vercel.app",
      ],
      [
        "architectures",
        "Architectures",
        "How model designs differ",
        "https://llm-architectures-explained.vercel.app",
      ],
      [
        "numerics",
        "Numerics",
        "Number formats and quantisation",
        "https://numerics-explained.vercel.app",
      ],
      [
        "tradeoffs",
        "Trade-offs",
        "Which lever helps which metric",
        "https://inference-tradeoffs-explained.vercel.app",
      ],
      [
        "harnesses",
        "Harnesses",
        "The loop between a model and the world",
        "https://agent-harnesses-explained.vercel.app",
      ],
      [
        "protocols",
        "Protocols",
        "MCP, transports, auth and A2A",
        "https://agent-protocols-explained.vercel.app",
      ],
      [
        "context",
        "Context",
        "Retrieval, memory and context engineering",
        "https://agent-context-explained.vercel.app",
      ],
      [
        "orchestration",
        "Orchestration",
        "Graphs, multi-agent patterns, durability",
        "https://agent-orchestration-explained.vercel.app",
      ],
      [
        "evals",
        "Evals",
        "Measuring agents",
        "https://agent-evals-explained.vercel.app",
      ],
      [
        "security",
        "Security",
        "Threats and defences",
        "https://agent-security-explained.vercel.app",
      ],
    ] as const;
    expect(
      await panel
        .locator("[data-site-key]")
        .evaluateAll((els) => els.map((e) => e.getAttribute("data-site-key"))),
    ).toEqual(SITES.map((s) => s[0]));
    for (const [key, name, blurb, href] of SITES) {
      await expect(site(key)).toContainText(name);
      await expect(site(key)).toContainText(blurb);
      if (href) await expect(site(key)).toHaveAttribute("href", href);
      // not live yet: shown, marked "(soon)", not a link
      else {
        await expect(site(key)).toContainText("(soon)");
        await expect(site(key)).not.toHaveAttribute("href", /./);
      }
    }
    // each track holds its own sites
    for (const [group, n] of [
      ["foundations", 2],
      ["hardware", 4],
      ["inference", 4],
      ["agents", 6],
    ] as const)
      await expect(
        menu.locator(`nav[data-site-group='${group}'] [data-site-key]`),
      ).toHaveCount(n);
    // no site is named after a track
    await expect(
      panel.getByRole("link", {
        name: /^(Foundations|Hardware|Inference|Agents)\b/,
      }),
    ).toHaveCount(0);
    // this site is highlighted
    await expect(site("kernels")).toHaveAttribute("aria-current", "true");
    await expect(panel.locator("[aria-current]")).toHaveCount(1);
    await expect(panel.getByText("Compute in transit")).toHaveCount(0);
    // every entry is a 44 px touch target
    expect(
      (await site("decoder").boundingBox())!.height,
    ).toBeGreaterThanOrEqual(44);
    // the tracks sit side by side on desktop and stack on a phone
    const f = (await menu
      .locator("nav[data-site-group='foundations']")
      .boundingBox())!;
    const a = (await menu
      .locator("nav[data-site-group='agents']")
      .boundingBox())!;
    if (width >= 1024) expect(Math.round(a.y)).toBe(Math.round(f.y));
    else expect(a.y).toBeGreaterThan(f.y);
    // the open panel stays on screen and the page never scrolls sideways
    const p = (await panel.boundingBox())!;
    expect(p.x).toBeGreaterThanOrEqual(0);
    expect(p.x + p.width).toBeLessThanOrEqual(width);
    expect(p.y + p.height).toBeLessThanOrEqual(800);
    const overflow = await page.evaluate(
      () =>
        document.scrollingElement!.scrollWidth -
        document.scrollingElement!.clientWidth,
    );
    expect(overflow).toBeLessThanOrEqual(0);
  }
  // keyboard: Enter on the button opens the menu (a native <details>, no JavaScript),
  // and Tab goes on to the first site
  await page.goto("/");
  await menu.locator("summary").focus();
  await page.keyboard.press("Enter");
  await expect(panel).toBeVisible();
  await page.keyboard.press("Tab");
  await expect(site("decoder")).toBeFocused();
});

test("the about page ends with the author credit", async ({ page }) => {
  await page.goto("/about");
  const credit = page.locator("main > :last-child");
  await expect(credit).toHaveAttribute("data-author-credit", "true");
  await expect(credit).toHaveText("Built by Brendan Lynskey");
  const link = credit.getByRole("link", { name: "Brendan Lynskey" });
  await expect(link).toHaveAttribute(
    "href",
    "https://github.com/BrendanJamesLynskey",
  );
});
