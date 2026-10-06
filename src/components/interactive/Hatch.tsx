/**
 * The "stalled / waiting" pattern of the visual standard: diagonal hatching
 * in the warning hue, so a stall never relies on colour alone. Put it in an
 * SVG's <defs> and fill with `url(#id)`.
 */
import { STATE_COLOUR } from "@/lib/viz/palette";

export function Hatch({ id }: { id: string }): JSX.Element {
  return (
    <pattern
      id={id}
      width={6}
      height={6}
      patternUnits="userSpaceOnUse"
      patternTransform="rotate(45)"
    >
      <rect width={6} height={6} fill={STATE_COLOUR.stalled} opacity={0.15} />
      <line
        x1={0}
        y1={0}
        x2={0}
        y2={6}
        stroke={STATE_COLOUR.stalled}
        strokeWidth={2}
      />
    </pattern>
  );
}
