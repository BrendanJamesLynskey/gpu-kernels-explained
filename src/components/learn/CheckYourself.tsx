/**
 * `CheckYourself`: the end-of-chapter block that holds a chapter's three
 * `SelfCheck` questions (brief 44, the learning layer). Its id,
 * `check-yourself`, is the target of the chapter's "Check yourself" link.
 * Server Component.
 *
 * This file is byte-identical on every *-explained site
 * (src/components/learn/CheckYourself.tsx); change it on all of them at once.
 */
import type { ReactNode } from "react";

export const CHECK_YOURSELF_ID = "check-yourself";

export function CheckYourself({
  children,
}: {
  children: ReactNode;
}): JSX.Element {
  return (
    <section
      id={CHECK_YOURSELF_ID}
      aria-labelledby={`${CHECK_YOURSELF_ID}-heading`}
      data-check-yourself
      className="mt-12 scroll-mt-20 border-t border-neutral-200 pt-8 dark:border-neutral-800"
    >
      <h2
        id={`${CHECK_YOURSELF_ID}-heading`}
        className="text-xl font-semibold tracking-tight"
      >
        Check yourself
      </h2>
      <p className="mt-2 text-sm text-neutral-600 dark:text-neutral-400">
        Commit to an answer before you reveal it: a wrong guess you can explain
        teaches more than a right one you skimmed. Every number in the answers
        comes from the same tested model as the chapter&rsquo;s.
      </p>
      {children}
    </section>
  );
}
