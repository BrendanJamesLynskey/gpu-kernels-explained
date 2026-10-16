/**
 * The link-preview card (Open Graph and Twitter), 1200 × 630, drawn at build
 * time by `next/og` from this site's `SHARE` (src/lib/share.ts) and its motif
 * (./motif.tsx): the track and address, the site's name, its one-line claim,
 * how it is checked, and a graphic from the site's hero. A chapter's card puts
 * the chapter's number and title above the site's name.
 *
 * The same file sits in every companion site (brief 39); only `SHARE` and the
 * motif differ. The fonts are Noto Sans subsets (SIL OFL 1.1, ./OFL.txt).
 */
import { readFile } from "node:fs/promises";
import { join } from "node:path";

import { ImageResponse } from "next/og";

import { SHARE, type Share } from "@/lib/share";

import { Glyph, Motif } from "./motif";

export const OG_SIZE = { width: 1200, height: 630 } as const;
export const OG_TYPE = "image/png";

const DIR = join(process.cwd(), "src/app/_og");
const font = (file: string) => readFile(join(DIR, file));

const INK = "#f8fafc";
const MUTED = "#cbd5e1";
const FAINT = "#94a3b8";
const BG = "#0b1020";

export type CardChapter = { number: number; title: string };

/** A second address's own glyph and graphic (the default: this site's). */
export type CardArt = { glyph: JSX.Element; motif: JSX.Element };

export async function card(
  chapter?: CardChapter,
  share: Share = SHARE,
  art: CardArt = { glyph: <Glyph size={52} />, motif: <Motif /> },
): Promise<ImageResponse> {
  const [bold, regular, mono] = await Promise.all([
    font("NotoSans-Bold.ttf"),
    font("NotoSans-Regular.ttf"),
    font("NotoSansMono-Bold.ttf"),
  ]);
  const c = share.color;
  const host = share.url.replace(/^https:\/\//, "");
  const big = chapter ? chapter.title : share.name;
  const size = big.length > 26 ? 60 : big.length > 22 ? 68 : 78;
  return new ImageResponse(
    <div
      style={{
        width: "100%",
        height: "100%",
        display: "flex",
        position: "relative",
        background: BG,
        backgroundImage: `radial-gradient(circle at 82% 45%, ${c}66 0%, ${BG}00 55%), radial-gradient(circle at 0% 100%, ${c}33 0%, ${BG}00 45%)`,
        fontFamily: "Noto Sans",
        color: INK,
      }}
    >
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          width: 700,
          padding: "64px 0 56px 72px",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 18 }}>
          {art.glyph}
          <div
            style={{
              display: "flex",
              fontFamily: "Noto Sans Mono",
              fontSize: 22,
              letterSpacing: 3,
              color: INK,
              whiteSpace: "nowrap",
            }}
          >
            {share.kicker ??
              (share.track ? `${share.track.toUpperCase()} TRACK` : "")}
          </div>
        </div>
        <div style={{ display: "flex", flexDirection: "column" }}>
          {chapter && (
            <div
              style={{
                display: "flex",
                fontFamily: "Noto Sans Mono",
                fontSize: 24,
                color: MUTED,
                marginBottom: 14,
              }}
            >
              {`CHAPTER ${String(chapter.number).padStart(2, "0")} · ${share.name}`}
            </div>
          )}
          <div
            style={{
              display: "flex",
              fontSize: size,
              fontWeight: 700,
              lineHeight: 1.06,
              letterSpacing: -1.5,
            }}
          >
            {big}
          </div>
          <div
            style={{
              display: "flex",
              marginTop: 22,
              fontSize: 32,
              lineHeight: 1.3,
              color: MUTED,
            }}
          >
            {share.claim}
          </div>
        </div>
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            borderLeft: `6px solid ${c}`,
            paddingLeft: 18,
          }}
        >
          <div
            style={{
              display: "flex",
              fontFamily: "Noto Sans Mono",
              fontSize: 16,
              letterSpacing: 2,
              color: FAINT,
            }}
          >
            HOW THIS IS CHECKED
          </div>
          <div
            style={{
              display: "flex",
              marginTop: 6,
              fontSize: 22,
              lineHeight: 1.35,
              color: MUTED,
            }}
          >
            {share.checkedShort}
          </div>
        </div>
      </div>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          width: 500,
          paddingRight: 40,
        }}
      >
        {art.motif}
      </div>
      <div
        style={{
          position: "absolute",
          right: 56,
          bottom: 44,
          display: "flex",
          fontSize: 22,
          color: FAINT,
        }}
      >
        {host}
      </div>
      <div
        style={{
          position: "absolute",
          left: 0,
          bottom: 0,
          width: "100%",
          height: 10,
          display: "flex",
          backgroundImage: `linear-gradient(90deg, ${c}, ${c}00)`,
        }}
      />
    </div>,
    {
      ...OG_SIZE,
      fonts: [
        { name: "Noto Sans", data: bold, weight: 700, style: "normal" },
        { name: "Noto Sans", data: regular, weight: 400, style: "normal" },
        { name: "Noto Sans Mono", data: mono, weight: 700, style: "normal" },
      ],
    },
  );
}
