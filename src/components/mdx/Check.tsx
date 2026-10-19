/**
 * The MDX side of the learning layer (brief 44): `<Check>` resolves a
 * self-check's numeric answer and hands it to the shared `SelfCheck`
 * (src/components/learn/, byte-identical on every *-explained site).
 *
 *   <Check id="k06-q1" n={1} input="number" v="ch06.t768" fmt="pct"
 *          q="768 threads per block on the A100: what occupancy?"
 *          href="#k-occupancy-examples" more="The guide's two examples">
 *     Why, in a sentence or two (Markdown; `<V>` works here).
 *   </Check>
 *
 * A numeric answer comes from `of` (a path into the model's value tree, as
 * `<V of>` prints it) or `v` (a value the model computes for the learning
 * layer, src/lib/learn/values.ts). It is printed by the chapters' own
 * `formatValue`, and the reader answers in the unit that prints: for
 * "2.07 ms" they type 2.07 and the unit shown is "ms". Nothing is typed in
 * by hand. Server Component.
 */
import type { ReactNode } from "react";

import { SelfCheck } from "@/components/learn/SelfCheck";
import type { Fmt } from "@/lib/gpu/values";
import { numericAnswer } from "@/lib/learn/answer";

export function Check({
  id,
  n,
  q,
  input = "choice",
  choices,
  answer,
  of,
  v,
  fmt,
  unit,
  tol,
  abs,
  min,
  max,
  step,
  href,
  more,
  children,
}: {
  id: string;
  n: number;
  /** the question */
  q: string;
  input?: "choice" | "number" | "slider";
  choices?: string[];
  /** choice: index of the right one */
  answer?: number;
  /** number or slider: a value-tree path, or a learn-value name */
  of?: string;
  v?: string;
  fmt?: Fmt;
  /** optional guard: the unit the question asks for must be the printed one */
  unit?: string;
  tol?: number;
  abs?: number;
  min?: number;
  max?: number;
  step?: number;
  /** the section that explains it, and its name */
  href?: string;
  more?: string;
  /** why the answer is what it is */
  children?: ReactNode;
}): JSX.Element {
  let value: number | undefined;
  let display: string | undefined;
  let shownUnit: string | undefined;
  if (input !== "choice") {
    const a = numericAnswer(of, v, fmt);
    if (unit !== undefined && unit.trim() !== a.unit.trim())
      throw new Error(`self-check ${id}: prints "${a.unit}", asks "${unit}"`);
    value = a.value;
    display = a.display;
    shownUnit = a.unit || undefined;
  } else if (
    choices === undefined ||
    answer === undefined ||
    answer < 0 ||
    answer >= choices.length
  ) {
    throw new Error(`self-check ${id}: a choice needs choices and an answer`);
  }
  return (
    <SelfCheck
      id={id}
      n={n}
      question={q}
      input={input}
      choices={choices}
      answer={answer}
      value={value}
      display={display}
      unit={shownUnit}
      tol={tol}
      abs={abs}
      min={min}
      max={max}
      step={step}
      why={children}
      href={href}
      hrefLabel={more}
    />
  );
}
