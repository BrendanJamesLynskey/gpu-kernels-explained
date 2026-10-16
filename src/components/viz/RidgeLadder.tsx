/**
 * The landing page's picture: the tensor cores' ridge point (peak dense
 * throughput over HBM bandwidth, flops per byte) for the A100, H100 and
 * B200, BF16 bars to scale with the narrower formats stacked on. Plain SVG
 * from the tested model (src/lib/gpu/blackwell.ts), rendered on the server.
 */
import { trim } from "@/lib/format";
import { TC_ORDER, ridges, tcPreset, type Format } from "@/lib/gpu/blackwell";
import { OKABE_ITO } from "@/lib/viz/palette";

const FMT: [Format, string, string][] = [
  ["bf16", "BF16", OKABE_ITO.blue],
  ["fp8", "FP8", OKABE_ITO.orange],
  ["fp4", "NVFP4", OKABE_ITO.vermillion],
];
const SHORT: Record<string, string> = {
  a100: "A100",
  h100: "H100",
  b200: "B200",
};

export function RidgeLadder(): JSX.Element {
  const W = 360;
  const LABEL = 52;
  const BAR = W - LABEL - 12;
  const ROW = 44;
  const max = Math.max(
    ...TC_ORDER.map((p) => ridges(p).fp4 ?? ridges(p).fp8 ?? ridges(p).bf16!),
  );
  const x = (v: number) => LABEL + (BAR * v) / max;
  const rows = TC_ORDER.map((pid) => ({
    pid,
    r: ridges(pid),
    name: tcPreset(pid).name,
  }));
  return (
    <svg
      viewBox={`0 0 ${W} ${rows.length * ROW + 30}`}
      className="h-auto w-full"
      role="img"
      data-ridge-ladder
      aria-label={`Tensor-core ridge points, flops per HBM byte: ${rows
        .map(
          (row) =>
            `${SHORT[row.pid]} BF16 ${trim(row.r.bf16!)}${row.r.fp8 ? `, FP8 ${trim(row.r.fp8)}` : ""}${row.r.fp4 ? `, NVFP4 ${trim(row.r.fp4)}` : ""}`,
        )
        .join("; ")}.`}
    >
      {rows.map((row, i) => {
        const y = 4 + i * ROW;
        let from = 0;
        return (
          <g key={row.pid}>
            <text
              x={0}
              y={y + 15}
              className="fill-neutral-800 text-[13.2px] font-semibold dark:fill-neutral-200"
            >
              {SHORT[row.pid]}
            </text>
            {FMT.map(([f, , colour]) => {
              const v = row.r[f];
              if (v === null) return null;
              const a = from;
              from = v;
              return (
                <rect
                  key={f}
                  x={x(a)}
                  y={y + 3}
                  width={Math.max(2, x(v) - x(a))}
                  height={16}
                  rx={2}
                  fill={colour}
                  opacity={f === "bf16" ? 1 : 0.75}
                />
              );
            })}
            <text
              x={LABEL}
              y={y + 36}
              className="fill-neutral-700 font-mono text-[12px] dark:fill-neutral-300"
            >
              {FMT.filter(([f]) => row.r[f] !== null)
                .map(([f, label]) => `${label} ${trim(row.r[f]!)}`)
                .join(" · ")}
            </text>
          </g>
        );
      })}
      {FMT.map(([, label, colour], i) => (
        <g
          key={label}
          transform={`translate(${LABEL + i * 80}, ${rows.length * ROW + 12})`}
        >
          <rect width={12} height={12} y={-1} rx={2} fill={colour} />
          <text
            x={16}
            y={10}
            className="fill-neutral-700 text-[12px] dark:fill-neutral-300"
          >
            {label}
          </text>
        </g>
      ))}
    </svg>
  );
}
