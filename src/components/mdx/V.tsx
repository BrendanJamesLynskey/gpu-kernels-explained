/**
 * <V of="a100.bw.smem" fmt="rate" />: a number from the tested model,
 * formatted, in running prose. <V v="ch06.t768" fmt="pct" />: one of the
 * learning layer's model values (src/lib/learn/values.ts), the same way.
 * Server Component.
 */
import { formatValue, lookup, type Fmt } from "@/lib/gpu/values";
import { learnValue } from "@/lib/learn/values";

export function V({
  of,
  v,
  fmt = "num",
}: {
  of?: string;
  v?: string;
  fmt?: Fmt;
}): JSX.Element {
  if (of !== undefined)
    return <span data-v={of}>{formatValue(lookup(of), fmt)}</span>;
  if (v === undefined) throw new Error("<V> needs `of` or `v`");
  return <span data-learn-value={v}>{formatValue(learnValue(v), fmt)}</span>;
}
