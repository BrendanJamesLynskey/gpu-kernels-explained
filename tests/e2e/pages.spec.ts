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

test("the three-group site switch: a toggle and a row on desktop, a dropdown below lg", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto("/");
  const full = page.locator("[data-site-switch='full']");
  await expect(full).toBeVisible();
  const inference = full.locator("nav[data-site-group='inference']");
  const training = full.locator("nav[data-site-group='training']");
  const agents = full.locator("nav[data-site-group='agents']");
  const TRANSIT = "https://compute-in-transit-results.vercel.app";
  // each group's row is one line that ends inside the viewport (no wrap at 1280 px)
  const oneLine = async (row: typeof inference) => {
    const tops = await row
      .locator("a, span[aria-disabled]")
      .evaluateAll((els) =>
        els.map((e) => Math.round(e.getBoundingClientRect().top)),
      );
    expect(new Set(tops).size).toBe(1);
    const b = await row.boundingBox();
    expect(b!.x + b!.width).toBeLessThanOrEqual(1280);
  };
  // starts on this site's group
  await expect(inference).toBeVisible();
  await expect(training).toBeHidden();
  await expect(agents).toBeHidden();
  await expect(
    inference.getByRole("link", { name: "Kernels" }),
  ).toHaveAttribute("aria-current", "true");
  // each toggle label shows its own group's row (CSS only)
  await full.locator("label[for='site-switch-inference']").click();
  await expect(inference).toBeVisible();
  await expect(training).toBeHidden();
  await expect(agents).toBeHidden();
  await oneLine(inference);
  for (const [name, href] of [
    ["Decoder", "https://transformer-decoder-explained.vercel.app"],
    ["Inference", "https://llm-inference-explained.vercel.app"],
    ["Architectures", "https://llm-architectures-explained.vercel.app"],
    ["Kernels", "https://gpu-kernels-explained.vercel.app"],
    ["Numerics", "https://numerics-explained.vercel.app"],
    ["Silicon", "https://systolic-arrays-explained.vercel.app"],
    ["Trade-offs", "https://inference-tradeoffs-explained.vercel.app"],
  ] as const)
    await expect(inference.getByRole("link", { name })).toHaveAttribute(
      "href",
      href,
    );
  await expect(
    inference.getByRole("link", { name: "Compute in transit" }),
  ).toHaveCount(0);
  await full.locator("label[for='site-switch-training']").click();
  await expect(training).toBeVisible();
  await expect(inference).toBeHidden();
  await expect(agents).toBeHidden();
  await oneLine(training);
  // Training is part of the decoder site and not live yet
  await expect(
    training.locator("span[aria-disabled]", { hasText: "Training" }),
  ).toBeVisible();
  await expect(training.getByRole("link", { name: "Training" })).toHaveCount(0);
  await expect(
    training.getByRole("link", { name: "Compute in transit" }),
  ).toHaveAttribute("href", TRANSIT);
  await full.locator("label[for='site-switch-agents']").click();
  await expect(agents).toBeVisible();
  await expect(inference).toBeHidden();
  await expect(training).toBeHidden();
  await oneLine(agents);
  for (const [name, href] of [
    ["Harnesses", "https://agent-harnesses-explained.vercel.app"],
    ["Protocols", "https://agent-protocols-explained.vercel.app"],
    ["Context", "https://agent-context-explained.vercel.app"],
    ["Orchestration", "https://agent-orchestration-explained.vercel.app"],
  ] as const)
    await expect(agents.getByRole("link", { name })).toHaveAttribute(
      "href",
      href,
    );
  for (const soon of ["Evals", "Security"]) {
    await expect(agents.getByText(soon)).toBeVisible();
    await expect(agents.getByRole("link", { name: soon })).toHaveCount(0);
  }
  // keyboard: the toggle is a set of radio buttons
  await page.getByRole("radio", { name: "Show the inference sites" }).focus();
  await page.keyboard.press("Space");
  await expect(inference).toBeVisible();
  await expect(agents).toBeHidden();

  // below lg (not sm): the dropdown, with all three groups under headings
  await page.setViewportSize({ width: 768, height: 800 });
  await expect(full).toBeHidden();
  const compact = page.locator("[data-site-switch='compact']");
  await expect(compact).toBeVisible();
  await page.setViewportSize({ width: 390, height: 800 });
  await expect(compact).toBeVisible();
  await compact.locator("summary").click();
  await expect(compact.locator("nav > p")).toHaveText([
    "Inference",
    "Training",
    "Agents",
  ]);
  await expect(compact.getByRole("link", { name: "Kernels" })).toHaveAttribute(
    "aria-current",
    "true",
  );
  await expect(
    compact.getByRole("link", { name: "Trade-offs" }),
  ).toHaveAttribute("href", "https://inference-tradeoffs-explained.vercel.app");
  await expect(compact.getByRole("link", { name: "Context" })).toHaveAttribute(
    "href",
    "https://agent-context-explained.vercel.app",
  );
  await expect(
    compact.getByRole("link", { name: "Orchestration" }),
  ).toHaveAttribute("href", "https://agent-orchestration-explained.vercel.app");
  await expect(
    compact.locator("span[aria-disabled]", { hasText: "Training" }),
  ).toBeVisible();
  await expect(compact.getByText("Security")).toBeVisible();
  await expect(compact.getByRole("link", { name: "Security" })).toHaveCount(0);
  await expect(
    compact.getByRole("link", { name: "Compute in transit" }),
  ).toHaveAttribute("href", TRANSIT);
  const box = await compact
    .getByRole("link", { name: "Decoder" })
    .boundingBox();
  expect(box!.height).toBeGreaterThanOrEqual(44);
  const overflow = await page.evaluate(
    () =>
      document.scrollingElement!.scrollWidth -
      document.scrollingElement!.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(0);
});
