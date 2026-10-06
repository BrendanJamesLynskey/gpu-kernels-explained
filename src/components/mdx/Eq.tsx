/**
 * A display equation rendered by KaTeX on the server, for an animation's
 * panel. Terms wrapped in \htmlClass{hl-<key>}{…} are coloured like the
 * thing they stand for and highlighted when the animation is on them (the
 * widget sets `data-hl="<key>"` on the panel; see globals.css), and hovering
 * a term highlights the matching part of the picture.
 *
 * Server Component: no KaTeX JavaScript reaches the browser.
 */
import katex from "katex";

export function Eq({
  tex,
  label,
}: {
  tex: string;
  label: string;
}): JSX.Element {
  const html = katex
    .renderToString(tex, {
      displayMode: true,
      throwOnError: true,
      output: "html",
      strict: "ignore",
      trust: (ctx) => ctx.command === "\\htmlClass",
    })
    .replace(
      // a wide equation scrolls on a phone: make it keyboard-reachable
      '<span class="katex-display">',
      '<span class="katex-display" tabindex="0">',
    );
  return (
    <div
      role="math"
      aria-label={label}
      className="eq"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
