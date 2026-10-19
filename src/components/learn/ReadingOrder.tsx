/**
 * `ReadingOrder`: a track's sites in the order they build on each other, with
 * what each adds and what it assumes (brief 44; generalised from the Agents
 * sites' brief-43 component of the same name). The track's data is passed in,
 * so one file serves every track; `current` marks the site the reader is on.
 * Server Component.
 *
 * This file is byte-identical on every *-explained site
 * (src/components/learn/ReadingOrder.tsx); change it on all of them at once.
 */
const LINK =
  "focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent";

export type TrackSite = {
  key: string;
  name: string;
  url: string;
  /** What the site adds, and what it assumes: one line, no full stop. */
  why: string;
};

export function ReadingOrder({
  track,
  intro,
  sites,
  current,
}: {
  /** The track's name, e.g. "Foundations". */
  track: string;
  intro?: string;
  sites: readonly TrackSite[];
  current?: string;
}): JSX.Element {
  return (
    <section
      aria-labelledby="reading-order-heading"
      data-testid="reading-order"
      className="mt-12 rounded-lg border border-neutral-200 p-5 dark:border-neutral-800"
    >
      <h2
        id="reading-order-heading"
        className="text-xl font-semibold tracking-tight"
      >
        Reading order for the {track} track
      </h2>
      {intro && (
        <p className="mt-2 max-w-2xl text-sm text-neutral-600 dark:text-neutral-300">
          {intro}
        </p>
      )}
      <ol className="mt-4 grid gap-2 text-sm sm:grid-cols-2">
        {sites.map((s, i) => (
          <li key={s.key} className="flex min-w-0 gap-2">
            <span className="font-mono text-xs leading-5 text-neutral-500 dark:text-neutral-400">
              {i + 1}
            </span>
            <span className="min-w-0">
              {s.key === current ? (
                <strong aria-current="page">{s.name} (this site)</strong>
              ) : (
                <a href={s.url} className={LINK}>
                  {s.name}
                </a>
              )}
              <span className="text-neutral-600 dark:text-neutral-400">
                : {s.why}.
              </span>
            </span>
          </li>
        ))}
      </ol>
    </section>
  );
}
