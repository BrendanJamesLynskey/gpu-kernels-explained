/**
 * Every animation plays, pauses, steps, scrubs, resets and answers the
 * keyboard, with no console errors, in light and dark mode at 1280 and
 * 390 px (visual standard §4).
 */
import { expect, test, type Locator, type Page } from "@playwright/test";

const ANIMATIONS = [
  ["/learn/01-memory-hierarchy", "hierarchy-widget"],
  ["/learn/02-roofline", "roofline-widget"],
  ["/learn/03-warps-and-divergence", "simt-widget"],
  ["/learn/04-coalescing", "coalescing-widget"],
  ["/learn/05-bank-conflicts", "bank-widget"],
  ["/learn/06-occupancy", "occupancy-widget"],
  ["/learn/07-gemm", "gemm-widget"],
  ["/learn/08-reductions", "reduction-widget"],
  ["/learn/09-softmax-and-flashattention", "flash-widget"],
  ["/learn/09-softmax-and-flashattention", "softmax-widget"],
  ["/learn/10-split-k-and-overlap", "timeline-widget"],
  ["/learn/10-split-k-and-overlap", "splitk-widget"],
  ["/learn/11-quantised-kernels", "dequant-widget"],
  ["/learn/11-quantised-kernels", "quant-widget"],
] as const;

async function step(fig: Locator): Promise<number> {
  return Number(await fig.getAttribute("data-step"));
}

async function pause(fig: Locator): Promise<void> {
  if ((await fig.getAttribute("data-playing")) === "true")
    await fig.getByTestId("play").click();
  await expect(fig).toHaveAttribute("data-playing", "false");
}

function errors(page: Page): string[] {
  const out: string[] = [];
  page.on("pageerror", (e) => out.push(`pageerror: ${e.message}`));
  page.on("console", (m) => {
    if (m.type() === "error") out.push(`console: ${m.text()}`);
  });
  return out;
}

for (const scheme of ["light", "dark"] as const) {
  for (const width of [1280, 390]) {
    test.describe(`${scheme} @ ${width}px`, () => {
      test.use({ colorScheme: scheme, viewport: { width, height: 900 } });
      for (const [path, id] of ANIMATIONS) {
        test(`${id} plays, steps, scrubs and resets`, async ({ page }) => {
          const errs = errors(page);
          await page.goto(path);
          const fig = page.getByTestId(id);
          await expect(fig).toBeVisible();
          await fig.scrollIntoViewIfNeeded();
          // it plays (it may already be playing: it starts when visible)
          await pause(fig);
          await fig
            .getByRole("button", { name: "Reset to the first step" })
            .click();
          expect(await step(fig)).toBe(0);
          await fig.getByTestId("play").click();
          await expect(fig).toHaveAttribute("data-playing", "true");
          await expect
            .poll(() => step(fig), { timeout: 8000 })
            .toBeGreaterThan(0);
          await pause(fig);
          // steps forward and back
          const s0 = await step(fig);
          const scrub = fig.getByTestId("scrub");
          const max = Number(await scrub.getAttribute("max"));
          if (s0 >= max)
            await fig.getByRole("button", { name: "Step back" }).click();
          const s1 = await step(fig);
          await fig.getByRole("button", { name: "Step forward" }).click();
          expect(await step(fig)).toBe(s1 + 1);
          await fig.getByRole("button", { name: "Step back" }).click();
          expect(await step(fig)).toBe(s1);
          // scrubs, and the caption follows
          const before = await fig.getByTestId("caption").textContent();
          await scrub.fill(String(max));
          expect(await step(fig)).toBe(max);
          await expect(fig.getByTestId("caption")).not.toHaveText(before ?? "");
          // keyboard: arrows step, Home resets, Space plays
          await fig.focus();
          await page.keyboard.press("ArrowLeft");
          expect(await step(fig)).toBe(max - 1);
          await page.keyboard.press("Home");
          expect(await step(fig)).toBe(0);
          await page.keyboard.press("ArrowRight");
          expect(await step(fig)).toBe(1);
          await page.keyboard.press(" ");
          await expect(fig).toHaveAttribute("data-playing", "true");
          await page.keyboard.press(" ");
          await expect(fig).toHaveAttribute("data-playing", "false");
          // speed and reset
          await fig.getByRole("combobox").selectOption("4");
          await fig
            .getByRole("button", { name: "Reset to the first step" })
            .click();
          expect(await step(fig)).toBe(0);
          // controls are at least 44 px tall (touch targets)
          const box = await fig.getByTestId("play").boundingBox();
          expect(box!.height).toBeGreaterThanOrEqual(44);
          const overflow = await page.evaluate(
            () =>
              document.scrollingElement!.scrollWidth -
              document.scrollingElement!.clientWidth,
          );
          expect(overflow).toBeLessThanOrEqual(0);
          expect(errs).toEqual([]);
        });
      }
    });
  }
}

test.describe("reduced motion", () => {
  test.use({ contextOptions: { reducedMotion: "reduce" } });
  for (const [path, id] of ANIMATIONS) {
    test(`${id} does not play by itself`, async ({ page }) => {
      await page.goto(path);
      const fig = page.getByTestId(id);
      await expect(fig).toBeVisible();
      await fig.scrollIntoViewIfNeeded();
      await page.waitForTimeout(2500);
      await expect(fig).toHaveAttribute("data-playing", "false");
      expect(await step(fig)).toBe(0);
      // stepping still works
      await fig.getByRole("button", { name: "Step forward" }).click();
      expect(await step(fig)).toBe(1);
    });
  }
});

test("animations play when scrolled into view and pause when scrolled away", async ({
  page,
}) => {
  await page.goto("/learn/03-warps-and-divergence");
  const fig = page.getByTestId("simt-widget");
  await fig.scrollIntoViewIfNeeded();
  await expect(fig).toHaveAttribute("data-playing", "true");
  await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
  await expect(fig).toHaveAttribute("data-playing", "false");
});

test("touching the scrub bar pauses, even on the step already shown", async ({
  page,
}) => {
  await page.goto("/learn/03-warps-and-divergence");
  const fig = page.getByTestId("simt-widget");
  await fig.scrollIntoViewIfNeeded();
  await expect(fig).toHaveAttribute("data-playing", "true");
  const before = await step(fig);
  await fig.getByTestId("scrub").dispatchEvent("pointerdown");
  await expect(fig).toHaveAttribute("data-playing", "false");
  // pausing did not move the animation
  expect(await step(fig)).toBeGreaterThanOrEqual(before);
});

test.describe("phone labels", () => {
  test.use({ viewport: { width: 390, height: 900 } });
  for (const [path, id] of ANIMATIONS) {
    test(`${id}: every SVG label is at least 11 px on a 390 px screen`, async ({
      page,
    }) => {
      await page.goto(path);
      const fig = page.getByTestId(id);
      await fig.scrollIntoViewIfNeeded();
      // let the size observer settle
      await page.waitForTimeout(300);
      const sizes = await fig.evaluate((el) =>
        [...el.querySelectorAll("svg text")]
          .filter((t) => (t.textContent ?? "").trim().length > 0)
          .map((t) => {
            const css = parseFloat(getComputedStyle(t).fontSize);
            const ctm = (t as SVGGraphicsElement).getScreenCTM();
            // the scale of the transform (rotated labels included)
            const k = ctm ? Math.hypot(ctm.a, ctm.b) : 1;
            return { text: t.textContent, px: css * k };
          }),
      );
      const small = sizes.filter((s) => s.px < 10.95);
      expect(small).toEqual([]);
    });
  }
});
