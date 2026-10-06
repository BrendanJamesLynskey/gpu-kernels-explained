/**
 * The bandwidth of each memory level as bars drawn to scale (linear), for
 * one preset: the landing page's picture. Plain SVG, rendered on the
 * server; colours are the family's level colours.
 */
import { fmtRate } from "@/lib/format";
import { LEVELS, derived, preset, type PresetId } from "@/lib/gpu/model";
import { LEVEL_COLOUR, LEVEL_NAME } from "@/lib/viz/palette";

export function BandwidthLadder({ pid }: { pid: PresetId }): JSX.Element {
  const p = preset(pid);
  const d = derived(p);
  const W = 360;
  const LABEL = 104;
  const BAR = W - LABEL - 64;
  const max = d.bw.reg;
  return (
    <svg
      viewBox={`0 0 ${W} ${LEVELS.length * 30 + 6}`}
      className="h-auto w-full"
      role="img"
      aria-label={`Bandwidth of each memory level of the ${p.name}, to scale: ${LEVELS.map((lv) => `${LEVEL_NAME[lv]} ${fmtRate(d.bw[lv])}`).join(", ")}.`}
    >
      {LEVELS.map((lv, i) => {
        const y = 4 + i * 30;
        const w = Math.max(2, (BAR * d.bw[lv]) / max);
        return (
          <g key={lv}>
            <text
              x={0}
              y={y + 15}
              className="fill-neutral-800 text-[11px] dark:fill-neutral-200"
            >
              {LEVEL_NAME[lv]}
            </text>
            <rect
              x={LABEL}
              y={y + 3}
              width={w}
              height={16}
              rx={2}
              fill={LEVEL_COLOUR[lv]}
            />
            <text
              x={LABEL + w + 6}
              y={y + 15}
              className="fill-neutral-700 font-mono text-[10px] dark:fill-neutral-300"
            >
              {fmtRate(d.bw[lv])}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
