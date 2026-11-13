/**
 * Multi-coat tonal hatching for darker value/color regions (PRO-37).
 * Coat 1 mid-dark spaced · Coat 2 denser cross · Coat 3 deep accents.
 * Stroke directions follow Edge Tangent Flow.
 */

import type { Pt } from "../strokeGuides/pathMath";
import type { ColorStructure } from "./colorStructure";
import type { EtfField } from "./xdogEtf";

export type TonalCoatSet = {
  /** Mid-dark spaced coats */
  mid: Pt[][];
  /** Darker denser / cross coats */
  dark: Pt[][];
  /** Deepest ink accents */
  deep: Pt[][];
};

function inMask(mask: Uint8Array, w: number, x: number, y: number): boolean {
  const xi = Math.round(x);
  const yi = Math.round(y);
  if (xi < 0 || yi < 0 || xi >= w) return false;
  const h = (mask.length / w) | 0;
  if (yi >= h) return false;
  return mask[yi * w + xi] === 1;
}

/**
 * Trace a short hatch segment along ETF (or perpendicular for cross coats)
 * starting at (x,y), staying inside mask.
 */
function hatchRay(
  mask: Uint8Array,
  etf: EtfField,
  w: number,
  h: number,
  x0: number,
  y0: number,
  cross: boolean,
  maxLen: number,
): Pt[] | null {
  if (!inMask(mask, w, x0, y0)) return null;
  const i0 = Math.round(y0) * w + Math.round(x0);
  let tx = etf.tx[i0];
  let ty = etf.ty[i0];
  if (cross) {
    const px = -ty;
    const py = tx;
    tx = px;
    ty = py;
  }
  const len = Math.hypot(tx, ty) || 1;
  tx /= len;
  ty /= len;

  const pts: Pt[] = [{ x: x0, y: y0 }];
  let x = x0;
  let y = y0;
  let traveled = 0;
  const step = 1.6;
  while (traveled < maxLen) {
    x += tx * step;
    y += ty * step;
    traveled += step;
    if (!inMask(mask, w, x, y)) break;
    // Refresh tangent gently
    const xi = Math.min(w - 1, Math.max(0, Math.round(x)));
    const yi = Math.min(h - 1, Math.max(0, Math.round(y)));
    const j = yi * w + xi;
    let ntx = cross ? -etf.ty[j] : etf.tx[j];
    let nty = cross ? etf.tx[j] : etf.ty[j];
    const nl = Math.hypot(ntx, nty) || 1;
    ntx /= nl;
    nty /= nl;
    // Keep consistent orientation
    if (ntx * tx + nty * ty < 0) {
      ntx = -ntx;
      nty = -nty;
    }
    tx = 0.7 * tx + 0.3 * ntx;
    ty = 0.7 * ty + 0.3 * nty;
    const tl = Math.hypot(tx, ty) || 1;
    tx /= tl;
    ty /= tl;
    pts.push({ x, y });
  }
  return pts.length >= 3 ? pts : null;
}

function sampleHatches(
  mask: Uint8Array,
  etf: EtfField,
  w: number,
  h: number,
  spacing: number,
  cross: boolean,
  maxLen: number,
  budget: number,
  seedOrder?: Array<{ x: number; y: number; spacing: number }>,
): Pt[][] {
  const out: Pt[][] = [];
  const used = new Uint8Array(w * h);
  const mark = (pts: Pt[]) => {
    for (const p of pts) {
      const x = Math.round(p.x);
      const y = Math.round(p.y);
      if (x < 0 || y < 0 || x >= w || y >= h) continue;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          const yy = y + dy;
          if (xx >= 0 && yy >= 0 && xx < w && yy < h) used[yy * w + xx] = 1;
        }
      }
    }
  };

  const trySeed = (x: number, y: number, sp: number) => {
    if (out.length >= budget) return;
    if (x < 0 || y < 0 || x >= w || y >= h) return;
    if (!mask[y * w + x] || used[y * w + x]) return;
    const j = y * w + x;
    const ox = x + etf.tx[j] * (sp * 0.2);
    const oy = y + etf.ty[j] * (sp * 0.2);
    const seg = hatchRay(mask, etf, w, h, ox, oy, cross, maxLen);
    if (!seg) return;
    out.push(seg);
    mark(seg);
  };

  if (seedOrder?.length) {
    for (const s of seedOrder) {
      if (out.length >= budget) break;
      trySeed(s.x, s.y, s.spacing);
    }
    return out;
  }

  for (let y = spacing; y < h - spacing && out.length < budget; y += spacing) {
    for (let x = spacing; x < w - spacing && out.length < budget; x += spacing) {
      trySeed(x, y, spacing);
    }
  }
  return out;
}

/** Face-first seeds: eyes/muzzle dense, ear band coarse — top-down grids wasted budget on ear fuzz. */
function portraitHatchSeeds(
  mask: Uint8Array,
  w: number,
  h: number,
  baseSpacing: number,
): Array<{ x: number; y: number; spacing: number }> {
  let minY = h;
  let maxY = 0;
  let minX = w;
  let maxX = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!mask[y * w + x]) continue;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  }
  if (maxY <= minY || maxX <= minX) return [];
  const fh = maxY - minY;
  const earEnd = minY + fh * 0.22;
  const faceEnd = minY + fh * 0.78;
  const seeds: Array<{ x: number; y: number; spacing: number }> = [];
  const pushBand = (y0: number, y1: number, sp: number) => {
    for (let y = Math.max(minY, y0 | 0); y <= Math.min(maxY, y1 | 0); y += sp) {
      for (let x = minX; x <= maxX; x += sp) {
        if (mask[y * w + x]) seeds.push({ x, y, spacing: sp });
      }
    }
  };
  // Eyes + cheeks + muzzle first (dense)
  pushBand(earEnd, faceEnd, Math.max(3, Math.round(baseSpacing * 0.85)));
  // Chin / whisker roots
  pushBand(faceEnd, maxY, Math.max(4, Math.round(baseSpacing * 1.1)));
  // Ears last + coarse so they cannot monopolize the budget
  pushBand(minY, earEnd, Math.max(6, Math.round(baseSpacing * 1.8)));
  return seeds;
}

function clearBorder(mask: Uint8Array, w: number, h: number, frac: number): Uint8Array {
  const out = mask.slice();
  const bx = Math.max(2, Math.round(w * frac));
  const by = Math.max(2, Math.round(h * frac));
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x < bx || y < by || x >= w - bx || y >= h - by) out[y * w + x] = 0;
    }
  }
  return out;
}

function filterShort(segs: Pt[][], minLen: number): Pt[][] {
  return segs.filter((s) => {
    if (s.length < 3) return false;
    let len = 0;
    for (let i = 1; i < s.length; i++) {
      len += Math.hypot(s[i].x - s[i - 1].x, s[i].y - s[i - 1].y);
    }
    return len >= minLen;
  });
}

/**
 * Build three tonal coats from color-structure dark masks + ETF.
 * Coats stay interior (border cleared) and visually subordinate.
 */
export function buildTonalCoats(
  cs: ColorStructure,
  opts?: {
    budget?: number;
    clearBorderFrac?: number;
    minSegLen?: number;
  },
): TonalCoatSet {
  const { w, h, etf } = cs;
  const budget = opts?.budget ?? 60;
  const border = opts?.clearBorderFrac ?? 0.04;
  const minSeg = opts?.minSegLen ?? 16;
  const midDarkMask = clearBorder(cs.midDarkMask, w, h, border);
  const darkMask = clearBorder(cs.darkMask, w, h, border);
  const deepMask = clearBorder(cs.deepMask, w, h, border);
  const diag = Math.hypot(w, h);
  const mid = filterShort(
    sampleHatches(
      midDarkMask,
      etf,
      w,
      h,
      Math.max(8, Math.round(diag / 48)),
      false,
      Math.max(22, diag / 16),
      Math.floor(budget * 0.4),
    ),
    minSeg,
  );
  const dark = filterShort(
    sampleHatches(
      darkMask,
      etf,
      w,
      h,
      Math.max(6, Math.round(diag / 60)),
      true,
      Math.max(20, diag / 18),
      Math.floor(budget * 0.35),
    ),
    minSeg,
  );
  const deep = filterShort(
    sampleHatches(
      deepMask,
      etf,
      w,
      h,
      Math.max(5, Math.round(diag / 80)),
      false,
      Math.max(16, diag / 24),
      Math.floor(budget * 0.25),
    ),
    minSeg,
  );
  return { mid, dark, deep };
}

function andMask(a: Uint8Array, b: Uint8Array): Uint8Array {
  const out = new Uint8Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] && b[i] ? 1 : 0;
  return out;
}

/**
 * Dense graphite-style coats for animal/face photos.
 * Short ETF-aligned hatches inside the subject only — reads as fur/value
 * without tracing every hair (which scribbles). Foreground mask required.
 *
 * Uses *subject-relative* L* (not global percentiles): orange/cream faces sit
 * above global mid-dark tiers and otherwise get empty cheeks + ear scribble.
 */
export function buildPortraitGraphiteCoats(
  cs: ColorStructure,
  fg: Uint8Array,
  opts?: {
    budget?: number;
  },
): TonalCoatSet {
  const { w, h, etf, L, edgeMag } = cs;
  const budget = opts?.budget ?? 560;
  const border = 0.025;
  const fgClear = clearBorder(fg, w, h, border);

  let fgN = 0;
  let sumL = 0;
  const fgL: number[] = [];
  for (let i = 0; i < L.length; i++) {
    if (!fgClear[i]) continue;
    fgN++;
    sumL += L[i];
    if (i % 3 === 0) fgL.push(L[i]);
  }
  if (fgN < 80) {
    return { mid: [], dark: [], deep: [] };
  }
  const fgMean = sumL / fgN;
  fgL.sort((a, b) => a - b);
  const q = (t: number) => fgL[Math.min(fgL.length - 1, Math.floor(t * (fgL.length - 1)))] ?? fgMean;
  // Relative tiers inside the animal (cream muzzle stays lighter / less hatched)
  const deepThr = q(0.22);
  const darkThr = q(0.42);
  const midThr = q(0.72);

  let edgeMean = 0;
  let edgeN = 0;
  for (let i = 0; i < edgeMag.length; i++) {
    if (!fgClear[i]) continue;
    edgeMean += edgeMag[i];
    edgeN++;
  }
  edgeMean = edgeN ? edgeMean / edgeN : 0;
  const stripeThr = edgeMean * 1.35;

  const midMask = new Uint8Array(w * h);
  const darkMask = new Uint8Array(w * h);
  const deepMask = new Uint8Array(w * h);
  const stripeMask = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) {
    if (!fgClear[i]) continue;
    const v = L[i];
    if (v <= midThr) midMask[i] = 1;
    if (v <= darkThr) darkMask[i] = 1;
    if (v <= deepThr) deepMask[i] = 1;
    // Stripe/marking edges — but skip ultra-busy ear fuzz (very high edge + top of FG)
    if (edgeMag[i] >= stripeThr) stripeMask[i] = 1;
  }
  // Soften ear scribble: clear stripe seeds in top 18% of FG bbox when edge is extreme
  {
    let minY = h;
    let maxY = 0;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        if (!fgClear[y * w + x]) continue;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
    const earBand = minY + (maxY - minY) * 0.18;
    const hot = edgeMean * 2.2;
    for (let y = minY; y <= earBand; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (stripeMask[i] && edgeMag[i] >= hot) stripeMask[i] = 0;
      }
    }
  }

  const markMask = new Uint8Array(w * h);
  for (let i = 0; i < markMask.length; i++) {
    markMask[i] = deepMask[i] || (darkMask[i] && stripeMask[i]) || stripeMask[i] ? 1 : 0;
  }

  const diag = Math.hypot(w, h);
  const midSp = Math.max(4, Math.round(diag / 95));
  const darkSp = Math.max(3, Math.round(diag / 120));
  const deepSp = Math.max(3, Math.round(diag / 140));
  const mid = filterShort(
    sampleHatches(
      midMask,
      etf,
      w,
      h,
      midSp,
      false,
      Math.max(12, diag / 30),
      Math.floor(budget * 0.45),
      portraitHatchSeeds(midMask, w, h, midSp),
    ),
    Math.max(7, diag * 0.008),
  );
  const dark = filterShort(
    sampleHatches(
      darkMask,
      etf,
      w,
      h,
      darkSp,
      true,
      Math.max(11, diag / 34),
      Math.floor(budget * 0.3),
      portraitHatchSeeds(darkMask, w, h, darkSp),
    ),
    Math.max(6, diag * 0.007),
  );
  const deep = filterShort(
    sampleHatches(
      markMask,
      etf,
      w,
      h,
      deepSp,
      false,
      Math.max(10, diag / 38),
      Math.floor(budget * 0.25),
      portraitHatchSeeds(markMask, w, h, deepSp),
    ),
    Math.max(5, diag * 0.006),
  );
  return { mid, dark, deep };
}

export function flattenCoats(coats: TonalCoatSet): Pt[][] {
  return [...coats.mid, ...coats.dark, ...coats.deep];
}
