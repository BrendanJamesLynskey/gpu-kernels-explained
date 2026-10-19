"use client";

/**
 * `SelfCheck`: one predict-then-reveal question (brief 44, the learning layer).
 *
 * The reader commits to an answer (a choice, a number, or a slider), then
 * reveals the model's answer, why it is so in a sentence or two, and a link to
 * the section that explains it. Numeric answers arrive as props from the site's
 * tested model or fixtures (each site's MDX adapter resolves them, through the
 * same path its prose numbers use); nothing here computes or types one in.
 *
 * Progressive enhancement: the markup is a native `<details>`, so with no
 * JavaScript the question, the choices and the answer are all in the page
 * (for search engines and no-JS readers) and the answer opens with a click.
 * With JavaScript the reveal waits until the reader has committed, and then
 * says how close they were.
 *
 * This file is byte-identical on every *-explained site
 * (src/components/learn/SelfCheck.tsx); change it on all of them at once.
 */
import { useId, useState, type ReactNode } from "react";

export type SelfCheckProps = {
  /** Stable id, unique on the page, e.g. "ch05-q1". */
  id: string;
  /** The question's number in the chapter (1, 2, 3). */
  n: number;
  question: ReactNode;
  /** How the reader answers. */
  input: "choice" | "number" | "slider";
  /** choice: the options, and the index of the right one. */
  choices?: ReactNode[];
  answer?: number;
  /** number, slider: the model's value in the units the reader types. */
  value?: number;
  /** number, slider: the value as the site prints it (e.g. "37.5%"). */
  display?: string;
  /** number, slider: a unit shown after the input, e.g. "%" or "GB". */
  unit?: string;
  /** number, slider: relative tolerance for "right" (default 0.1 = 10%). */
  tol?: number;
  /** number, slider: absolute tolerance, for answers at or near 0. */
  abs?: number;
  /** slider: range and step. */
  min?: number;
  max?: number;
  step?: number;
  /** Why the answer is what it is: one or two sentences. */
  why: ReactNode;
  /** The section that explains it, and its name. */
  href?: string;
  hrefLabel?: string;
};

const BUTTON =
  "focus-ring inline-flex min-h-11 cursor-pointer select-none items-center rounded border px-4 py-2 text-sm font-medium";

/** Is `guess` within the tolerance of `value`? */
export function isClose(
  guess: number,
  value: number,
  tol = 0.1,
  abs = 0,
): boolean {
  return Math.abs(guess - value) <= Math.max(tol * Math.abs(value), abs);
}

export function SelfCheck(props: SelfCheckProps): JSX.Element {
  const {
    id,
    n,
    question,
    input,
    choices = [],
    answer = 0,
    value = 0,
    display,
    unit,
    tol = 0.1,
    abs = 0,
    min = 0,
    max = 100,
    step = 1,
    why,
    href,
    hrefLabel,
  } = props;
  const uid = useId();
  const [picked, setPicked] = useState<number | null>(null);
  const [typed, setTyped] = useState("");
  const [slid, setSlid] = useState<number | null>(null);
  const [open, setOpen] = useState(false);
  const [nudge, setNudge] = useState(false);

  const guess =
    input === "number"
      ? typed.trim() === "" || !Number.isFinite(Number(typed))
        ? null
        : Number(typed)
      : input === "slider"
        ? slid
        : picked;
  const committed = guess !== null;
  const right =
    guess === null
      ? null
      : input === "choice"
        ? guess === answer
        : isClose(guess, value, tol, abs);
  const shown = display ?? String(value);
  const hintId = `${uid}-hint`;

  return (
    <div
      data-self-check={id}
      className="my-6 rounded-lg border border-neutral-200 p-4 dark:border-neutral-800"
    >
      <fieldset className="min-w-0">
        <legend className="float-left w-full font-medium leading-7">
          <span className="mr-2 font-mono text-xs uppercase tracking-widest text-neutral-500 dark:text-neutral-400">
            Q{n}
          </span>
          {question}
        </legend>
        <div className="clear-both pt-3">
          {input === "choice" && (
            <ul className="grid gap-2">
              {choices.map((c, i) => (
                <li key={i}>
                  <label className="flex min-h-11 cursor-pointer items-start gap-3 rounded border border-neutral-200 px-3 py-2 text-sm hover:border-accent dark:border-neutral-700 dark:hover:border-indigo-400">
                    <input
                      type="radio"
                      name={`${uid}-choice`}
                      value={i}
                      checked={picked === i}
                      onChange={() => {
                        setPicked(i);
                        setNudge(false);
                      }}
                      className="mt-1 accent-accent"
                    />
                    <span className="min-w-0">{c}</span>
                  </label>
                </li>
              ))}
            </ul>
          )}
          {input === "number" && (
            <label className="flex flex-wrap items-center gap-2 text-sm">
              <span>Your answer</span>
              <input
                type="number"
                inputMode="decimal"
                step="any"
                value={typed}
                onChange={(e) => {
                  setTyped(e.target.value);
                  setNudge(false);
                }}
                className="focus-ring min-h-11 w-36 rounded border border-neutral-300 bg-white px-3 py-2 font-mono text-sm dark:border-neutral-700 dark:bg-neutral-900"
              />
              {unit && <span>{unit}</span>}
            </label>
          )}
          {input === "slider" && (
            <label className="block text-sm">
              <span className="flex flex-wrap items-baseline gap-2">
                <span>Your answer</span>
                <output className="font-mono">
                  {slid === null ? "move the slider" : `${slid}${unit ?? ""}`}
                </output>
              </span>
              <input
                type="range"
                min={min}
                max={max}
                step={step}
                value={slid ?? min}
                onChange={(e) => {
                  setSlid(Number(e.target.value));
                  setNudge(false);
                }}
                className="mt-2 block min-h-11 w-full accent-accent"
              />
            </label>
          )}
        </div>
      </fieldset>

      <details
        open={open}
        onToggle={(e) => setOpen((e.currentTarget as HTMLDetailsElement).open)}
        className="mt-3"
      >
        <summary
          aria-describedby={committed ? undefined : hintId}
          onClick={(e) => {
            if (!committed && !open) {
              e.preventDefault();
              setNudge(true);
            }
          }}
          className={`${BUTTON} list-none border-accent text-accent dark:border-indigo-300 dark:text-indigo-300 [&::-webkit-details-marker]:hidden`}
        >
          {open ? "Hide the answer" : "Reveal the answer"}
        </summary>
        <div
          className="mt-3 rounded bg-neutral-50 p-3 text-sm leading-6 dark:bg-neutral-900"
          data-self-check-answer
        >
          {right !== null && (
            <p data-verdict={right ? "right" : "wrong"} className="font-medium">
              {right
                ? "Right."
                : input === "choice"
                  ? "Not quite."
                  : `Not quite: you said ${guess}${unit ?? ""}.`}
            </p>
          )}
          <p>
            <span className="font-medium">Answer: </span>
            <span data-answer>
              {input === "choice" ? choices[answer] : `${shown}${unit ?? ""}`}
            </span>
            {input !== "choice" && (
              <span className="text-neutral-600 dark:text-neutral-400">
                {" "}
                (
                {tol === 0 && abs === 0
                  ? "exact"
                  : `counted right within ${
                      abs > 0 && tol * Math.abs(value) < abs
                        ? `± ${abs}${unit ?? ""}`
                        : `${Math.round(tol * 100)}%`
                    }`}
                )
              </span>
            )}
          </p>
          <div className="mt-2">{why}</div>
          {href && (
            <p className="mt-2">
              <a
                href={href}
                className="focus-ring rounded text-accent underline underline-offset-2 dark:text-indigo-300"
              >
                {hrefLabel ?? "Where this is explained"} →
              </a>
            </p>
          )}
        </div>
      </details>
      <p
        id={hintId}
        aria-live="polite"
        className="mt-2 text-xs text-neutral-600 dark:text-neutral-400"
      >
        {!committed
          ? nudge
            ? "Commit to an answer first, then reveal it."
            : input === "choice"
              ? "Pick one, then reveal."
              : "Commit to a number, then reveal."
          : ""}
      </p>
    </div>
  );
}
