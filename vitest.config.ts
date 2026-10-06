import { defineConfig } from "vitest/config";
import path from "node:path";

export default defineConfig({
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
      "@content": path.resolve(__dirname, "./content"),
    },
  },
  test: {
    globals: true,
    environment: "node",
    include: ["tests/unit/**/*.test.ts"],
    coverage: {
      provider: "v8",
      reporter: ["text", "html", "lcov"],
      include: ["src/lib/**/*.ts"],
      thresholds: {
        // The execution model, its captions and the values the prose quotes.
        "src/lib/gpu/**": {
          lines: 100,
          functions: 100,
          statements: 100,
          branches: 90,
        },
        lines: 90,
        functions: 90,
        branches: 85,
        statements: 90,
      },
    },
  },
});
