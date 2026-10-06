import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx,mdx}", "./content/**/*.{md,mdx}"],
  // Follow the OS setting (CLAUDE.md §5: "default is system preference").
  // This was "class", but nothing ever set the class, so every `dark:`
  // variant was dead and the site was light-only.
  darkMode: "media",
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: [
          "JetBrains Mono",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "monospace",
        ],
      },
      colors: {
        // Accent — see SPEC §14 Q1. Default indigo-600.
        accent: {
          DEFAULT: "#4f46e5",
          fg: "#ffffff",
        },
      },
    },
  },
  plugins: [],
};

export default config;
