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

test("the two-group site switch: a toggle and a row on desktop, a dropdown below lg", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto("/");
  const full = page.locator("[data-site-switch='full']");
  await expect(full).toBeVisible();
  const llm = full.locator("nav[data-site-group='llm']");
  const agents = full.locator("nav[data-site-group='agents']");
  // starts on this site's group
  await expect(llm).toBeVisible();
  await expect(agents).toBeHidden();
  await expect(llm.getByRole("link", { name: "Kernels" })).toHaveAttribute(
    "aria-current",
    "true",
  );
  for (const [name, href] of [
    ["Decoder", "https://transformer-decoder-explained.vercel.app"],
    ["Inference", "https://llm-inference-explained.vercel.app"],
    ["Architectures", "https://llm-architectures-explained.vercel.app"],
    ["Kernels", "https://gpu-kernels-explained.vercel.app"],
    ["Numerics", "https://numerics-explained.vercel.app"],
    ["Silicon", "https://systolic-arrays-explained.vercel.app"],
    ["Trade-offs", "https://inference-tradeoffs-explained.vercel.app"],
  ] as const)
    await expect(llm.getByRole("link", { name })).toHaveAttribute("href", href);
  // the toggle shows the agent sites (CSS only)
  await full.getByText("Agents", { exact: true }).click();
  await expect(agents).toBeVisible();
  await expect(llm).toBeHidden();
  await expect(agents.getByRole("link", { name: "Harnesses" })).toHaveAttribute(
    "href",
    "https://agent-harnesses-explained.vercel.app",
  );
  for (const soon of [
    "Protocols",
    "Context",
    "Orchestration",
    "Evals",
    "Security",
  ]) {
    await expect(agents.getByText(soon)).toBeVisible();
    await expect(agents.getByRole("link", { name: soon })).toHaveCount(0);
  }
  // keyboard: the toggle is a pair of radio buttons
  await page.getByRole("radio", { name: "Show the LLM systems sites" }).focus();
  await page.keyboard.press("Space");
  await expect(llm).toBeVisible();

  // below lg (not sm): the dropdown
  await page.setViewportSize({ width: 768, height: 800 });
  await expect(full).toBeHidden();
  const compact = page.locator("[data-site-switch='compact']");
  await expect(compact).toBeVisible();
  await page.setViewportSize({ width: 390, height: 800 });
  await expect(compact).toBeVisible();
  await compact.locator("summary").click();
  await expect(compact.getByText("LLM systems")).toBeVisible();
  await expect(compact.getByText("Agents", { exact: true })).toBeVisible();
  await expect(compact.getByRole("link", { name: "Kernels" })).toHaveAttribute(
    "aria-current",
    "true",
  );
  await expect(
    compact.getByRole("link", { name: "Harnesses" }),
  ).toHaveAttribute("href", "https://agent-harnesses-explained.vercel.app");
  await expect(compact.getByText("Security")).toBeVisible();
  await expect(compact.getByRole("link", { name: "Security" })).toHaveCount(0);
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
