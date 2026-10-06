"use client";

/**
 * Readable SVG labels on phones (visual standard §3: "degrade gracefully on
 * phones: fewer elements, the same story"). An SVG drawn in a fixed viewBox
 * shrinks with the screen, and so do its labels: a 10-unit label in a
 * 360-unit picture is 8.6 px on a 390 px phone. `fs(units)` returns a font
 * size in viewBox units that renders at no less than `min` CSS pixels at
 * the SVG's current width, and `narrow` tells a widget to drop or shorten
 * the labels that would then collide.
 *
 * The scale is measured with a ResizeObserver, so it follows rotation and
 * window resizes.
 */
import { useCallback, useEffect, useRef, useState } from "react";

export type SvgFont = {
  /** Attach to the <svg>. */
  ref: (el: SVGSVGElement | null) => void;
  /** A font size in viewBox units, at least `min` px on screen. */
  fs: (units: number) => number;
  /** Rendered px per viewBox unit. */
  scale: number;
  /** The picture is drawn smaller than its viewBox (a phone). */
  narrow: boolean;
};

export function useSvgFont(viewWidth: number, min = 11): SvgFont {
  const [scale, setScale] = useState(1);
  const observer = useRef<ResizeObserver | null>(null);

  const ref = useCallback(
    (el: SVGSVGElement | null) => {
      observer.current?.disconnect();
      if (!el) return;
      const measure = () => {
        const w = el.getBoundingClientRect().width;
        if (w > 0) setScale(w / viewWidth);
      };
      measure();
      if (typeof ResizeObserver === "undefined") return;
      observer.current = new ResizeObserver(measure);
      observer.current.observe(el);
    },
    [viewWidth],
  );

  useEffect(() => () => observer.current?.disconnect(), []);

  const fs = useCallback(
    (units: number) => Math.max(units, min / scale),
    [min, scale],
  );

  return { ref, fs, scale, narrow: scale < 0.98 };
}
