/**
 * This site's favicon glyph (src/app/icon.svg) and its card's graphic.
 * Glyph: the chip of the Hardware track, the letter K, #059669.
 */
import { derived, preset } from "@/lib/gpu/model";
import { LEVEL_COLOUR } from "@/lib/viz/palette";

export function Glyph({ size }: { size: number }): JSX.Element {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32">
      <rect x="3.5" y="3.5" width="25" height="25" rx="2.5" fill="#059669" />
      <path
        d="M9 0h3v4h-3zM20 0h3v4h-3zM9 28h3v4h-3zM20 28h3v4h-3zM0 9h4v3h-4zM0 20h4v3h-4zM28 9h4v3h-4zM28 20h4v3h-4z"
        fill="#059669"
      />
      <path
        d="M21.63 23H18.25L14.59 17.1L13.33 18V23H10.37V9H13.33V15.41Q13.63 15 13.92 14.59Q14.22 14.18 14.51 13.76L18.29 9H21.59L16.71 15.2Z"
        fill="#fff"
      />
    </svg>
  );
}

/**
 * The roofline of chapter 2, for the H100: attainable flops against arithmetic intensity (log-log),
 * the HBM slope meeting the FP32 roof and, far higher, the tensor-core roof, with both ridge
 * points marked. From the model at build time.
 */
export function Motif(): JSX.Element {
  const p = preset("h100");
  const d = derived(p);
  const lx0 = -1;
  const lx1 = 4;
  const ly1 = Math.log10(p.peak_tensor) + 0.15;
  const ly0 = ly1 - 4.6;
  const X = (i: number) => 20 + (400 * (Math.log10(i) - lx0)) / (lx1 - lx0);
  const Y = (f: number) => 400 - (380 * (Math.log10(f) - ly0)) / (ly1 - ly0);
  const roof = (peak: number, ridge: number) =>
    `${X(10 ** lx0)},${Y(d.bw.hbm * 10 ** lx0)} ${X(ridge)},${Y(peak)} ${X(10 ** lx1)},${Y(peak)}`;
  return (
    <svg width="440" height="420" viewBox="0 0 440 420">
      <line
        x1={20}
        y1={404}
        x2={420}
        y2={404}
        stroke="#475569"
        strokeWidth={3}
      />
      <line x1={16} y1={20} x2={16} y2={404} stroke="#475569" strokeWidth={3} />
      <polyline
        points={roof(p.peak_tensor, d.ridge_tensor)}
        fill="none"
        stroke={LEVEL_COLOUR.reg}
        strokeWidth={8}
        strokeLinejoin="round"
      />
      <polyline
        points={roof(p.peak_fp32, d.ridge_fp32)}
        fill="none"
        stroke={LEVEL_COLOUR.smem}
        strokeWidth={8}
        strokeLinejoin="round"
      />
      <polyline
        points={`${X(10 ** lx0)},${Y(d.bw.hbm * 10 ** lx0)} ${X(d.ridge_fp32)},${Y(p.peak_fp32)}`}
        fill="none"
        stroke={LEVEL_COLOUR.hbm}
        strokeWidth={8}
        strokeLinecap="round"
      />
      <circle
        cx={X(d.ridge_fp32)}
        cy={Y(p.peak_fp32)}
        r={12}
        fill="#0b1020"
        stroke={LEVEL_COLOUR.smem}
        strokeWidth={5}
      />
      <circle
        cx={X(d.ridge_tensor)}
        cy={Y(p.peak_tensor)}
        r={12}
        fill="#0b1020"
        stroke={LEVEL_COLOUR.reg}
        strokeWidth={5}
      />
    </svg>
  );
}
