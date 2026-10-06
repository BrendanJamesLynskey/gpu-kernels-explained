/**
 * <V of="a100.bw.smem" fmt="rate" />: a number from the tested model,
 * formatted, in running prose. Server Component.
 */
import { formatValue, lookup, type Fmt } from "@/lib/gpu/values";

export function V({ of, fmt = "num" }: { of: string; fmt?: Fmt }): JSX.Element {
  return <span data-v={of}>{formatValue(lookup(of), fmt)}</span>;
}
