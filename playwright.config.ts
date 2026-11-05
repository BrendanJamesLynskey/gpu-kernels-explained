import { defineConfig, devices } from "@playwright/test";

/**
 * e2e runs against a production build (`next build` + `next start`), as
 * transformer-explainer's CI does, but the site is a static export (`output: "export"`),
 * so the built `out/` folder is served by scripts/serve-out.mjs instead of `next start`. Set PLAYWRIGHT_BASE_URL
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
          ? "node scripts/serve-out.mjs 3000"
          : "pnpm build && node scripts/serve-out.mjs 3000",
        url: "http://localhost:3000",
        reuseExistingServer: !process.env.CI,
        timeout: 600_000,
      },
});
