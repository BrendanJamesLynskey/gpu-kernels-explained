/**
 * Markdown tables in the chapters: wrapped so a wide table scrolls inside
 * its own box on a phone, and that box is keyboard-focusable (axe's
 * scrollable-region-focusable rule). Server Component.
 */
import type { ComponentProps } from "react";

export function MdxTable(props: ComponentProps<"table">): JSX.Element {
  return (
    <div
      role="region"
      aria-label="Table"
      tabIndex={0}
      className="focus-ring my-4 max-w-full overflow-x-auto rounded"
    >
      <table {...props} />
    </div>
  );
}

/**
 * Code blocks scroll sideways on a phone too, so they take keyboard focus
 * (the same axe rule). This site's addition.
 */
export function MdxPre(props: ComponentProps<"pre">): JSX.Element {
  return <pre tabIndex={0} {...props} />;
}
