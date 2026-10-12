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

test("the site switch: one Site menu listing every group, at every width", async ({
  page,
}) => {
  const menu = page.locator("[data-site-switch='menu']");
  const panel = menu.locator("summary + div");
  const training = menu.locator("nav[data-site-group='training']");
  const inference = menu.locator("nav[data-site-group='inference']");
  const agents = menu.locator("nav[data-site-group='agents']");
  for (const width of [1280, 390]) {
    await page.setViewportSize({ width, height: 800 });
    await page.goto("/");
    // closed: one button naming this site
    await expect(menu).toBeVisible();
    await expect(menu.locator("summary")).toContainText("Kernels");
    await expect(panel).toBeHidden();
    // one click opens every group, in the order Training, Inference, Agents
    await menu.locator("summary").click();
    await expect(panel).toBeVisible();
    await expect(menu.locator("nav > p")).toHaveText([
      "Training",
      "Inference",
      "Agents",
    ]);
    // this site is highlighted
    await expect(
      menu.getByRole("link", { name: "Kernels", exact: true }),
    ).toHaveAttribute("aria-current", "true");
    await expect(
      training.getByRole("link", { name: "Training", exact: true }),
    ).toHaveAttribute(
      "href",
      "https://transformer-training-explained.vercel.app",
    );
    await expect(panel.getByText("Compute in transit")).toHaveCount(0);
    for (const [name, href] of [
      ["Decoder", "https://transformer-decoder-explained.vercel.app"],
      ["Inference", "https://llm-inference-explained.vercel.app"],
      ["Architectures", "https://llm-architectures-explained.vercel.app"],
      ["Kernels", "https://gpu-kernels-explained.vercel.app"],
      ["Numerics", "https://numerics-explained.vercel.app"],
      ["Silicon", "https://systolic-arrays-explained.vercel.app"],
      ["Trade-offs", "https://inference-tradeoffs-explained.vercel.app"],
    ] as const)
      await expect(
        inference.getByRole("link", { name, exact: true }),
      ).toHaveAttribute("href", href);
    for (const [name, href] of [
      ["Harnesses", "https://agent-harnesses-explained.vercel.app"],
      ["Protocols", "https://agent-protocols-explained.vercel.app"],
      ["Context", "https://agent-context-explained.vercel.app"],
      ["Orchestration", "https://agent-orchestration-explained.vercel.app"],
    ] as const)
      await expect(
        agents.getByRole("link", { name, exact: true }),
      ).toHaveAttribute("href", href);
    for (const soon of ["Evals", "Security"]) {
      await expect(agents.getByText(soon)).toBeVisible();
      await expect(agents.getByRole("link", { name: soon })).toHaveCount(0);
    }
    // every entry is a 44 px touch target
    const decoder = await inference
      .getByRole("link", { name: "Decoder", exact: true })
      .boundingBox();
    expect(decoder!.height).toBeGreaterThanOrEqual(44);
    // the groups sit side by side on desktop and stack on a phone
    const t = (await training.boundingBox())!;
    const a = (await agents.boundingBox())!;
    if (width >= 1024) expect(Math.round(a.y)).toBe(Math.round(t.y));
    else expect(a.y).toBeGreaterThan(t.y);
    // the open panel stays on screen and the page never scrolls sideways
    const p = (await panel.boundingBox())!;
    expect(p.x).toBeGreaterThanOrEqual(0);
    expect(p.x + p.width).toBeLessThanOrEqual(width);
    const overflow = await page.evaluate(
      () =>
        document.scrollingElement!.scrollWidth -
        document.scrollingElement!.clientWidth,
    );
    expect(overflow).toBeLessThanOrEqual(0);
  }
  // keyboard: Enter on the button opens the menu (a native <details>, no JavaScript)
  await page.goto("/");
  await menu.locator("summary").focus();
  await page.keyboard.press("Enter");
  await expect(panel).toBeVisible();
});
