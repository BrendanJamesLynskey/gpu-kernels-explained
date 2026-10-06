/**
 * axe (serious and critical violations) on the main pages, in light and
 * dark mode.
 */
import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

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

for (const [scheme, width] of [
  ["light", 1280],
  ["dark", 1280],
  ["dark", 390],
] as const) {
  test.describe(`axe, ${scheme} @ ${width}px`, () => {
    test.use({ colorScheme: scheme, viewport: { width, height: 900 } });
    // scan the chapters with every layer (Concept, Maths, Code) shown
    test.beforeEach(async ({ page }) => {
      await page.addInitScript(() =>
        window.localStorage.setItem(
          "te:layers",
          JSON.stringify({ concept: true, maths: true, code: true }),
        ),
      );
    });
    for (const path of PAGES) {
      test(`${path} has no serious or critical violations`, async ({
        page,
      }) => {
        await page.goto(path);
        await page.waitForLoadState("networkidle");
        await expect(page.locator("[data-pending-widget]")).toHaveCount(0);
        // pause the animations so the scan sees a stable page
        for (const b of await page.getByTestId("play").all())
          if ((await b.getAttribute("aria-label")) === "Pause") await b.click();
        const r = await new AxeBuilder({ page }).analyze();
        const bad = r.violations.filter(
          (v) => v.impact === "serious" || v.impact === "critical",
        );
        expect(
          bad.map(
            (v) =>
              `${v.id}: ${v.nodes
                .map((n) => n.target.join(" "))
                .slice(0, 3)
                .join(", ")}`,
          ),
        ).toEqual([]);
      });
    }
  });
}
