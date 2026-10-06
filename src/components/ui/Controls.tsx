"use client";

/**
 * Small form controls shared by the interactives, styled like
 * transformer-explainer's widget inputs (focus ring, neutral borders,
 * accent for the active choice).
 */
import { useId } from "react";

/** A labelled range slider with its value shown. */
export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  format = (v) => String(v),
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
}): JSX.Element {
  const id = useId();
  return (
    <div className="flex min-w-0 flex-col gap-1 text-sm">
      <label
        htmlFor={id}
        className="flex justify-between gap-2 font-medium text-neutral-700 dark:text-neutral-300"
      >
        <span>{label}</span>
        <span className="font-mono text-xs text-neutral-600 dark:text-neutral-400">
          {format(value)}
        </span>
      </label>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="focus-ring w-full accent-indigo-600"
      />
    </div>
  );
}

/** A radio group drawn as a segmented control. */
export function Segmented<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: readonly { value: T; label: string }[];
  onChange: (v: T) => void;
}): JSX.Element {
  return (
    <div className="min-w-0">
      <span className="text-xs font-medium uppercase tracking-widest text-neutral-500 dark:text-neutral-400">
        {label}
      </span>
      <div
        role="radiogroup"
        aria-label={label}
        className="mt-1 flex flex-wrap gap-1 rounded border border-neutral-300 bg-white p-1 text-xs dark:border-neutral-700 dark:bg-neutral-950"
      >
        {options.map((o) => (
          <button
            key={o.value}
            type="button"
            role="radio"
            aria-checked={value === o.value}
            onClick={() => onChange(o.value)}
            className={`focus-ring rounded px-2 py-1 font-mono ${
              value === o.value
                ? "bg-accent text-accent-fg"
                : "text-neutral-600 hover:bg-neutral-100 dark:text-neutral-300 dark:hover:bg-neutral-800"
            }`}
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

/** A labelled number readout. */
export function Stat({
  label,
  value,
  hint,
  plain = false,
}: {
  label: string;
  value: string;
  hint?: string;
  /** Keep the label's case (for labels with symbols such as α or λ). */
  plain?: boolean;
}): JSX.Element {
  return (
    <div className="min-w-0 rounded bg-white p-2 ring-1 ring-neutral-200 dark:bg-neutral-950 dark:ring-neutral-800">
      <div
        className={`text-[0.7rem] tracking-wide text-neutral-500 dark:text-neutral-400 ${plain ? "" : "uppercase"}`}
      >
        {label}
      </div>
      <div className="mt-0.5 break-words font-mono text-sm text-neutral-900 dark:text-neutral-100">
        {value}
      </div>
      {hint && (
        <div className="mt-0.5 text-[0.7rem] text-neutral-500 dark:text-neutral-400">
          {hint}
        </div>
      )}
    </div>
  );
}

/** A primary action button. */
export function ActionButton({
  children,
  onClick,
  disabled,
  variant = "primary",
}: {
  children: React.ReactNode;
  onClick: () => void;
  disabled?: boolean;
  variant?: "primary" | "secondary";
}): JSX.Element {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={
        variant === "primary"
          ? "focus-ring rounded bg-accent px-4 py-1.5 text-sm font-medium text-accent-fg hover:opacity-90 disabled:opacity-40"
          : "focus-ring rounded border border-neutral-300 px-3 py-1.5 text-xs text-neutral-700 hover:bg-neutral-100 disabled:opacity-40 dark:border-neutral-700 dark:text-neutral-300 dark:hover:bg-neutral-800"
      }
    >
      {children}
    </button>
  );
}
