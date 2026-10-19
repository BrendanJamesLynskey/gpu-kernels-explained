/**
 * A self-check's numeric answer (brief 44), resolved as the chapters print
 * their numbers: `of` is a path into the model's value tree (as `<V of>`),
 * `v` a learn value (src/lib/learn/values.ts); the site's own `formatValue`
 * prints it, and the reader answers in the printed unit. Used by the MDX
 * adapter (src/components/mdx/Check.tsx) and by the tests.
 */
import { formatValue, lookup, type Fmt } from "@/lib/gpu/values";
import { learnValue } from "@/lib/learn/values";

/** The raw model value behind `of` or `v`. */
export function resolveRaw(
  of?: string,
  v?: string,
): ReturnType<typeof lookup> | number {
  if (of !== undefined) return lookup(of);
  if (v !== undefined) return learnValue(v);
  throw new Error("a numeric self-check needs `of` or `v`");
}

/**
 * The answer as the chapter prints it: the printed number (what the reader
 * types), and the unit after it ("%" glued on, a word unit after a space).
 */
export function numericAnswer(
  of: string | undefined,
  v: string | undefined,
  fmt: Fmt = "num",
): { raw: number; value: number; display: string; unit: string } {
  const raw = resolveRaw(of, v);
  if (typeof raw !== "number")
    throw new Error(`${of ?? v} is not a number (${raw})`);
  const text = formatValue(raw, fmt);
  const m = /^(-?[\d,]*\.?\d+)(%|\s.+)?$/.exec(text);
  if (!m) throw new Error(`cannot read a number in "${text}" (${of ?? v})`);
  const display = m[1]!;
  return {
    raw,
    value: Number(display.replace(/,/g, "")),
    display,
    unit: m[2] ?? "",
  };
}
