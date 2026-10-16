import { describe, expect, it } from "vitest";

import { SHARE, withShare } from "@/lib/share";
import { SITE_URL } from "@/lib/site";

describe("share: link previews", () => {
  it("names the site, its colour and its address", () => {
    expect(SHARE.name).toBe("GPU Kernels Explained");
    expect(SHARE.url).toBe(SITE_URL);
    expect(SHARE.color).toMatch(/^#[0-9a-f]{6}$/);
    expect(SHARE.claim.length).toBeLessThanOrEqual(80);
    expect(SHARE.checkedShort.length).toBeLessThanOrEqual(110);
  });

  it("gives the home page the site's own card", () => {
    const m = withShare("/", { description: "d" });
    expect(m.openGraph).toMatchObject({
      title: SHARE.name,
      url: "/",
      siteName: SHARE.name,
      type: "website",
      description: "d",
    });
    expect(m.twitter).toMatchObject({ card: "summary_large_image" });
  });

  it("names a page by its own title and path", () => {
    const m = withShare("/about", { title: "About" });
    expect(m.title).toBe("About");
    expect(m.openGraph).toMatchObject({
      title: `About · ${SHARE.name}`,
      url: "/about",
      type: "article",
      description: SHARE.description,
    });
    const a = withShare("/x", { title: { absolute: "X" } });
    expect(a.openGraph).toMatchObject({ title: `X · ${SHARE.name}` });
    const n = withShare("/y", { title: { default: "Y", template: "%s" } });
    expect(n.openGraph).toMatchObject({ title: SHARE.name });
    expect(n.openGraph).toHaveProperty("images");
    const c = withShare("/learn/x", { title: "X" }, SHARE, true);
    expect(c.openGraph).not.toHaveProperty("images");
    expect(c.twitter).not.toHaveProperty("images");
  });
});
