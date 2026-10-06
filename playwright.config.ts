import { defineConfig, devices } from "@playwright/test";

/**
 * e2e runs against a production build (`next build` + `next start`), as
 * transformer-explainer's CI does. `--no-experimental-require-module`
 * makes Node refuse to require() an ES module, as Vercel's function runtime
 * does, so a server dependency that would only fail on Vercel fails here
 * (transformer-explainer RUNBOOK §7, 2026-10-04). Set PLAYWRIGHT_BASE_URL
 * to test a deployment instead (no local server is started then).
 */
const external = process.env.PLAYWRIGHT_BASE_URL;

export default defineConfig({
  testDir: "./tests/e2e",
  timeout: 45_000,
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : 2,
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL: external ?? "http://localhost:3000",
    trace: "on-first-retry",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: external
    ? undefined
    : {
        command: process.env.E2E_SKIP_BUILD
          ? "node --no-experimental-require-module node_modules/next/dist/bin/next start -p 3000"
          : "pnpm build && node --no-experimental-require-module node_modules/next/dist/bin/next start -p 3000",
        url: "http://localhost:3000",
        reuseExistingServer: !process.env.CI,
        timeout: 600_000,
      },
});
