/**
 * Photo → stroke-guide pipeline (client-side, 100% offline).
 *
 * Stages (see stages.ts): XDoG+ETF → Zhang–Suen → junction merge →
 * geometry → hierarchy/kinematics → StrokeGuidePlayer.
 * iOS may inject Apple Vision FG / Loomis hints. Match gate in generate.ts.
 */

import {
  ptsToPathD,
  type Pt,
  type StepStrokeGuide,
  type GuideStroke,
} from "../strokeGuides/pathMath";
import { runCanonicalStages, type VisionHints } from "./stages";

export type RgbaImage = {
  width: number;
  height: number;
  data: Uint8ClampedArray | Uint8Array;
};

export type BoardRect = { x: number; y: number; width: number; height: number };

export type PipelineOptions = {
  /** Long-side working resolution (default 960). */
  maxWork?: number;
  maxStrokes?: number;
  /** Multiplier on RDP / fit tolerance (1 = default). */
  simplifyScale?: number;
  /** Prefer geometry primitive fitting (default true). */
  useGeometry?: boolean;
  forceMode?: "line-art" | "photo" | "auto";
  inkThrBias?: number;
  label?: string;
  /** iOS Vision hints (FG mask / face / pose) in work-pixel space after downsample. */
  vision?: VisionHints;
  /** Subject-class drawing policy (portrait / structure / …). */
  policy?: import("./subjectClass").DrawPolicy;
};

export type PhotoGuideResult =
  | {
      ok: true;
      steps: StepStrokeGuide[];
      strokeCount: number;
      workSize: { w: number; h: number };
      mode: "line-art" | "photo";
      /** Source image placement inside the 800×560 stroke board. */
      board: BoardRect;
      /** Outline polylines in work space (edge validation). */
      workStrokes: Pt[][];
      /** Multi-coat tonal hatching in work space (dark-mask validation). */
      tonalWorkStrokes: Pt[][];
      /** Downsampled RGBA used for validation. */
      workImage: RgbaImage;
      attemptLabel?: string;
    }
  | { ok: false; reason: string; attemptLabel?: string };

export const VIEW_W = 800;
export const VIEW_H = 560;
const PAD = 36;
const GRAPHITE = "#2C3340";
/** Long-side working resolution — enough for dense ink, still OK on tablets. */
const MAX_WORK = 960;
/** Contour-heavy budget — short ticks filtered separately (PRO-38). */
const MAX_STROKES = 1000;

const N8: [number, number][] = [
  [1, 0],
  [1, -1],
  [0, -1],
  [-1, -1],
  [-1, 0],
  [-1, 1],
  [0, 1],
  [1, 1],
];

export function toGray(img: RgbaImage): Float32Array {
  const n = img.width * img.height;
  const g = new Float32Array(n);
  const d = img.data;
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    g[i] = 0.299 * d[p] + 0.587 * d[p + 1] + 0.114 * d[p + 2];
  }
  return g;
}

export function boxBlur(
  src: Float32Array,
  w: number,
  h: number,
  radius = 1,
): Float32Array {
  const r = Math.max(0, Math.floor(radius));
  if (r === 0) return src.slice();
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);
  const span = 2 * r + 1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sum = 0;
      for (let k = -r; k <= r; k++) {
        const xx = Math.min(w - 1, Math.max(0, x + k));
        sum += src[y * w + xx];
      }
      tmp[y * w + x] = sum / span;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sum = 0;
      for (let k = -r; k <= r; k++) {
        const yy = Math.min(h - 1, Math.max(0, y + k));
        sum += tmp[yy * w + x];
      }
      out[y * w + x] = sum / span;
    }
  }
  return out;
}

/** Otsu threshold on 0–255 gray histogram. */
export function otsuThreshold(gray: Float32Array): number {
  const hist = new Float64Array(256);
  for (let i = 0; i < gray.length; i++) {
    hist[Math.min(255, Math.max(0, gray[i] | 0))]++;
  }
  const total = gray.length;
  let sum = 0;
  for (let i = 0; i < 256; i++) sum += i * hist[i];
  let sumB = 0;
  let wB = 0;
  let best = 0;
  let bestVar = -1;
  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (!wB) continue;
    const wF = total - wB;
    if (!wF) break;
    sumB += t * hist[t];
    const mB = sumB / wB;
    const mF = (sum - sumB) / wF;
    const v = wB * wF * (mB - mF) * (mB - mF);
    if (v > bestVar) {
      bestVar = v;
      best = t;
    }
  }
  return best;
}

/**
 * Artwork / illustration signals shared by graphite scans and color drawings.
 * Color alone must NOT reject — marker / crayon / digital art on paper used to
 * fall through to photo stylize (silhouette soup on faces + walls).
 */
export type ArtworkSignals = {
  mean: number;
  inkFrac: number;
  midInkFrac: number;
  paperFrac: number;
  nearPaperFrac: number;
  midToneFrac: number;
  thr: number;
  meanChroma: number;
  vividFrac: number;
  /** Non-paper regions look stroke/flat rather than continuous photo texture. */
  strokeLikeTex: boolean;
  /** Mean local contrast in non-paper pixels (fur/skin runs higher). */
  texMean: number;
  /** Fraction of sampled pixels with very low local contrast. */
  flatFrac: number;
  /** Posterized / flat-fill illustration (comics, marker fills). */
  posterized: boolean;
  /** Large bright paper-like substrate present. */
  paperBacked: boolean;
};

export function measureArtworkSignals(
  gray: Float32Array,
  rgba?: RgbaImage,
  width?: number,
  height?: number,
): ArtworkSignals {
  let mean = 0;
  let dark = 0;
  let midInk = 0;
  let bright = 0;
  let nearPaper = 0;
  let midTone = 0;
  for (let i = 0; i < gray.length; i++) {
    const v = gray[i];
    mean += v;
    if (v < 90) dark++;
    // Soft graphite often sits 90–175 on paper — still "ink"
    if (v < 175) midInk++;
    if (v > 200) bright++;
    // Cream / scanned paper often lands 185–200 (not pure white)
    if (v > 185) nearPaper++;
    // Photographic midtones (fur, skin, sky) — not blank paper
    if (v >= 70 && v <= 190) midTone++;
  }
  const n = gray.length || 1;
  mean /= n;
  const inkFrac = dark / n;
  const midInkFrac = midInk / n;
  const paperFrac = bright / n;
  const nearPaperFrac = nearPaper / n;
  const midToneFrac = midTone / n;
  const thr = otsuThreshold(gray);

  let meanChroma = 0;
  let vividFrac = 0;
  if (rgba && rgba.data.length >= n * 4) {
    let chroma = 0;
    let samples = 0;
    let vivid = 0;
    const step = Math.max(1, Math.floor(n / 4000));
    for (let i = 0; i < n; i += step) {
      const o = i * 4;
      const r = rgba.data[o];
      const g = rgba.data[o + 1];
      const b = rgba.data[o + 2];
      const mx = Math.max(r, g, b);
      const mn = Math.min(r, g, b);
      const c = mx - mn;
      chroma += c;
      if (c > 28) vivid++;
      samples++;
    }
    meanChroma = chroma / Math.max(1, samples);
    vividFrac = vivid / Math.max(1, samples);
  }

  // Local texture in non-paper pixels: photos stay busy; drawings are flat or stroke-thin.
  const w =
    width && height && width * height === n
      ? width
      : Math.max(1, Math.round(Math.sqrt(n * 0.6)));
  const h = width && height && width * height === n ? height : Math.max(1, (n / w) | 0);
  let texSum = 0;
  let texN = 0;
  let flat = 0;
  let flatN = 0;
  const stepY = Math.max(1, (h / 120) | 0);
  const stepX = Math.max(1, (w / 120) | 0);
  for (let y = 1; y < h - 1; y += stepY) {
    for (let x = 1; x < w - 1; x += stepX) {
      const i = y * w + x;
      if (i >= n) continue;
      const v = gray[i];
      const d =
        Math.abs(v - gray[i - 1]) +
        Math.abs(v - gray[i + 1]) +
        Math.abs(v - gray[Math.max(0, i - w)]) +
        Math.abs(v - gray[Math.min(n - 1, i + w)]);
      if (v <= 185) {
        texSum += d;
        texN++;
      }
      flatN++;
      if (d < 18) flat++;
    }
  }
  const texMean = texN ? texSum / texN : 99;
  const flatFrac = flatN ? flat / flatN : 0;
  // Drawings: non-paper either sparse dark strokes (low coverage) or flat fills.
  const strokeLikeTex =
    (nearPaperFrac > 0.28 && texMean < 42 && midInkFrac < 0.7) ||
    (flatFrac > 0.62 && texMean < 36);

  // Posterization: limited palette + flat patches (comics / marker fills).
  let posterized = false;
  if (rgba && rgba.data.length >= n * 4) {
    const counts = new Map<number, number>();
    const step = Math.max(1, Math.floor(n / 3500));
    let samples = 0;
    for (let i = 0; i < n; i += step) {
      const o = i * 4;
      const key =
        ((rgba.data[o] >> 4) << 8) |
        ((rgba.data[o + 1] >> 4) << 4) |
        (rgba.data[o + 2] >> 4);
      counts.set(key, (counts.get(key) || 0) + 1);
      samples++;
    }
    const sorted = [...counts.values()].sort((a, b) => b - a);
    const top = sorted.slice(0, 12).reduce((s, v) => s + v, 0);
    const topFrac = top / Math.max(1, samples);
    const palette = counts.size;
    posterized =
      flatFrac > 0.55 &&
      ((palette <= 48 && topFrac > 0.72) ||
        (palette <= 80 && topFrac > 0.85 && vividFrac > 0.25));
  }

  const paperBacked =
    mean > 145 &&
    (paperFrac > 0.38 || nearPaperFrac > 0.36) &&
    midInkFrac > 0.012 &&
    midInkFrac < 0.75 &&
    inkFrac < 0.5 &&
    thr < 230;

  return {
    mean,
    inkFrac,
    midInkFrac,
    paperFrac,
    nearPaperFrac,
    midToneFrac,
    thr,
    meanChroma,
    vividFrac,
    strokeLikeTex,
    texMean,
    flatFrac,
    posterized,
    paperBacked,
  };
}

/**
 * Detect already-drawn artwork (graphite, ink, marker, crayon, flat illustration).
 * Bright paper / posterized fills + stroke-like structure — chroma does NOT veto.
 */
export function isLineArt(
  gray: Float32Array,
  /** Optional RGBA — used for chroma / posterization, not as a hard photo veto. */
  rgba?: RgbaImage,
): boolean {
  const s = measureArtworkSignals(
    gray,
    rgba,
    rgba?.width,
    rgba?.height,
  );

  // Photos of soft subjects fill the frame with midtones and little paper.
  if (s.paperFrac < 0.18 && s.nearPaperFrac < 0.3 && s.midToneFrac > 0.55 && !s.posterized)
    return false;
  if (s.mean < 130 && s.paperFrac < 0.2 && s.nearPaperFrac < 0.28) return false; // night photos

  const graphiteLike = s.meanChroma <= 18 || s.vividFrac <= 0.05;

  // Studio pet / product shots on white paper-like backdrops are NOT drawings.
  // They share paperFrac with graphite scans but keep colorful photographic midtones.
  const photoOnPaper =
    s.paperBacked &&
    !s.posterized &&
    !graphiteLike &&
    s.vividFrac > 0.12 &&
    s.midToneFrac > 0.2 &&
    s.texMean >= 16;
  if (photoOnPaper) return false;

  // Busy photographic texture on non-paper (fur/skin) without posterization.
  if (
    !s.paperBacked &&
    !s.posterized &&
    s.midToneFrac > 0.5 &&
    s.nearPaperFrac < 0.35 &&
    !s.strokeLikeTex
  ) {
    return false;
  }

  // Paper-heavy drawings (hard ink OR soft pencil). Soft graphite rarely dips <90.
  // softPencil is graphite-only — white-bg color photos must not ride this gate.
  const softPencil =
    graphiteLike &&
    s.mean > 165 &&
    s.paperFrac > 0.45 &&
    s.midInkFrac > 0.02 &&
    s.midInkFrac < 0.55 &&
    s.thr < 210;
  // Cream-paper / JPEG scans of pencil art: near-white paper may sit under 200.
  const softGraphiteScan =
    graphiteLike &&
    s.mean > 150 &&
    s.nearPaperFrac > 0.35 &&
    s.midInkFrac > 0.02 &&
    s.midInkFrac < 0.72 &&
    s.thr < 220 &&
    s.inkFrac < 0.45;
  const hardInk =
    graphiteLike &&
    s.mean > 150 &&
    (s.paperFrac > 0.35 || s.nearPaperFrac > 0.4) &&
    s.inkFrac > 0.008 &&
    s.inkFrac < 0.62 &&
    s.thr < 230 &&
    (s.thr > 70 || (s.paperFrac > 0.7 && s.inkFrac < 0.25) || s.nearPaperFrac > 0.55);

  // Color marker / crayon / tinted scans on paper — posterized, stroke-like,
  // or paper-heavy with minority ink (JPEG chroma noise can kill strokeLike).
  const colorOnPaper =
    s.paperBacked &&
    !photoOnPaper &&
    (s.vividFrac > 0.04 || s.meanChroma > 12) &&
    (s.posterized ||
      (s.strokeLikeTex && s.midInkFrac < 0.4) ||
      (s.paperFrac > 0.55 &&
        s.midInkFrac > 0.04 &&
        s.midInkFrac < 0.5 &&
        s.inkFrac < 0.4));

  // Flat-fill illustrations / comics (little blank paper, posterized palette).
  const flatIllustration =
    s.posterized &&
    (s.strokeLikeTex || s.flatFrac > 0.55) &&
    s.inkFrac < 0.55 &&
    s.mean > 90;

  return (
    softPencil ||
    softGraphiteScan ||
    hardInk ||
    colorOnPaper ||
    flatIllustration ||
    // Generic paper-backed graphite stroke art
    (s.paperBacked && graphiteLike && s.strokeLikeTex && s.inkFrac < 0.45)
  );
}

/** Ink = 1 where darker than threshold (after optional invert). */
export function thresholdInk(gray: Float32Array, thr: number): Uint8Array {
  const out = new Uint8Array(gray.length);
  for (let i = 0; i < gray.length; i++) out[i] = gray[i] < thr ? 1 : 0;
  return out;
}

/** One iteration of Zhang–Suen thinning (binary ink). */
function zhangSuenIteration(
  bin: Uint8Array,
  w: number,
  h: number,
  step: 0 | 1,
): boolean {
  const mark: number[] = [];
  const at = (x: number, y: number) =>
    x >= 0 && y >= 0 && x < w && y < h ? bin[y * w + x] : 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      if (!bin[y * w + x]) continue;
      const p2 = at(x, y - 1);
      const p3 = at(x + 1, y - 1);
      const p4 = at(x + 1, y);
      const p5 = at(x + 1, y + 1);
      const p6 = at(x, y + 1);
      const p7 = at(x - 1, y + 1);
      const p8 = at(x - 1, y);
      const p9 = at(x - 1, y - 1);
      const neighbors = [p2, p3, p4, p5, p6, p7, p8, p9];
      const B = neighbors.reduce((a, b) => a + b, 0);
      if (B < 2 || B > 6) continue;
      let A = 0;
      for (let i = 0; i < 8; i++) {
        if (neighbors[i] === 0 && neighbors[(i + 1) % 8] === 1) A++;
      }
      if (A !== 1) continue;
      if (step === 0) {
        if (p2 * p4 * p6 !== 0) continue;
        if (p4 * p6 * p8 !== 0) continue;
      } else {
        if (p2 * p4 * p8 !== 0) continue;
        if (p2 * p6 * p8 !== 0) continue;
      }
      mark.push(y * w + x);
    }
  }
  for (const i of mark) bin[i] = 0;
  return mark.length > 0;
}

/** Zhang–Suen skeleton until stable (cap iterations). */
export function thinSkeleton(
  ink: Uint8Array,
  w: number,
  h: number,
  maxIter = 64,
): Uint8Array {
  const bin = ink.slice();
  for (let i = 0; i < maxIter; i++) {
    const a = zhangSuenIteration(bin, w, h, 0);
    const b = zhangSuenIteration(bin, w, h, 1);
    if (!a && !b) break;
  }
  return bin;
}

/**
 * Euclidean distance transform on a binary mask (1 = foreground).
 * Each foreground pixel stores the radius to the nearest background pixel.
 * Used to recover variable stroke weight after Zhang–Suen centerlining
 * (avoids hollow perimeter strokes from contour-tracing thick ink).
 */
export function distanceTransform(
  bin: Uint8Array,
  w: number,
  h: number,
): Float32Array {
  const dist = new Float32Array(w * h);
  const INF = 1e9;
  for (let i = 0; i < dist.length; i++) dist[i] = bin[i] ? INF : 0;

  // Forward pass
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!bin[i]) continue;
      let d = dist[i];
      if (x > 0) d = Math.min(d, dist[i - 1] + 1);
      if (y > 0) d = Math.min(d, dist[i - w] + 1);
      if (x > 0 && y > 0) d = Math.min(d, dist[i - w - 1] + Math.SQRT2);
      if (x + 1 < w && y > 0) d = Math.min(d, dist[i - w + 1] + Math.SQRT2);
      dist[i] = d;
    }
  }
  // Backward pass
  for (let y = h - 1; y >= 0; y--) {
    for (let x = w - 1; x >= 0; x--) {
      const i = y * w + x;
      if (!bin[i]) continue;
      let d = dist[i];
      if (x + 1 < w) d = Math.min(d, dist[i + 1] + 1);
      if (y + 1 < h) d = Math.min(d, dist[i + w] + 1);
      if (x + 1 < w && y + 1 < h)
        d = Math.min(d, dist[i + w + 1] + Math.SQRT2);
      if (x > 0 && y + 1 < h) d = Math.min(d, dist[i + w - 1] + Math.SQRT2);
      dist[i] = d;
    }
  }
  return dist;
}

/** Mean distance-transform radius sampled along a polyline (stroke half-width). */
export function sampleStrokeRadius(
  pts: Pt[],
  dist: Float32Array,
  w: number,
  h: number,
): number {
  if (!pts.length) return 1;
  let sum = 0;
  let n = 0;
  for (const p of pts) {
    const x = Math.round(p.x);
    const y = Math.round(p.y);
    if (x < 0 || y < 0 || x >= w || y >= h) continue;
    const r = dist[y * w + x];
    if (r > 0 && r < 1e8) {
      sum += r;
      n++;
    }
  }
  return n ? sum / n : 1;
}

function neighborCount(bin: Uint8Array, w: number, h: number, x: number, y: number): number {
  let n = 0;
  for (const [dx, dy] of N8) {
    const xx = x + dx;
    const yy = y + dy;
    if (xx >= 0 && yy >= 0 && xx < w && yy < h && bin[yy * w + xx]) n++;
  }
  return n;
}

/**
 * Walk 1-px skeleton into polylines. Prefer endpoints, then leftover chains.
 * Through-junctions mode keeps face/brick loops intact (PRO-38).
 */
export function traceSkeleton(
  skel: Uint8Array,
  w: number,
  h: number,
  opts?: { throughJunctions?: boolean },
): Pt[][] {
  const used = new Uint8Array(w * h);
  const strokes: Pt[][] = [];
  const through = !!opts?.throughJunctions;

  const isInk = (x: number, y: number) =>
    x >= 0 && y >= 0 && x < w && y < h && skel[y * w + x] === 1 && !used[y * w + x];

  const walk = (sx: number, sy: number) => {
    const pts: Pt[] = [];
    let x = sx;
    let y = sy;
    let guard = 0;
    const max = w * h;
    while (guard++ < max) {
      used[y * w + x] = 1;
      pts.push({ x, y });
      const optsN: [number, number][] = [];
      for (const [dx, dy] of N8) {
        const nx = x + dx;
        const ny = y + dy;
        if (isInk(nx, ny)) optsN.push([nx, ny]);
      }
      if (!optsN.length) break;
      // Prefer continuing straight-ish: pick closest to previous direction.
      let best = optsN[0];
      if (pts.length >= 2 && optsN.length > 1) {
        const prev = pts[pts.length - 2];
        const vx = x - prev.x;
        const vy = y - prev.y;
        let bestDot = -Infinity;
        for (const [nx, ny] of optsN) {
          const wx = nx - x;
          const wy = ny - y;
          const dot = vx * wx + vy * wy;
          if (dot > bestDot) {
            bestDot = dot;
            best = [nx, ny];
          }
        }
      }
      const deg = neighborCount(skel, w, h, best[0], best[1]);
      x = best[0];
      y = best[1];
      // Default: split at junctions. Through mode continues (branches picked later).
      if (!through && deg >= 3 && pts.length > 2) {
        used[y * w + x] = 1;
        pts.push({ x, y });
        break;
      }
    }
    return pts;
  };

  const endpoints: [number, number][] = [];
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      if (!skel[y * w + x]) continue;
      if (neighborCount(skel, w, h, x, y) === 1) endpoints.push([x, y]);
    }
  }
  for (const [x, y] of endpoints) {
    if (used[y * w + x]) continue;
    const pts = walk(x, y);
    if (pts.length >= 4) strokes.push(pts);
  }
  // Remaining loops / fragments
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      if (!skel[y * w + x] || used[y * w + x]) continue;
      const pts = walk(x, y);
      if (pts.length >= 4) strokes.push(pts);
    }
  }
  return strokes;
}

export function sobelMagnitude(
  gray: Float32Array,
  w: number,
  h: number,
): { mag: Float32Array; gx: Float32Array; gy: Float32Array } {
  const mag = new Float32Array(w * h);
  const gx = new Float32Array(w * h);
  const gy = new Float32Array(w * h);
  const at = (x: number, y: number) =>
    gray[Math.min(h - 1, Math.max(0, y)) * w + Math.min(w - 1, Math.max(0, x))];
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const gxv =
        -at(x - 1, y - 1) +
        at(x + 1, y - 1) -
        2 * at(x - 1, y) +
        2 * at(x + 1, y) -
        at(x - 1, y + 1) +
        at(x + 1, y + 1);
      const gyv =
        -at(x - 1, y - 1) -
        2 * at(x, y - 1) -
        at(x + 1, y - 1) +
        at(x - 1, y + 1) +
        2 * at(x, y + 1) +
        at(x + 1, y + 1);
      const i = y * w + x;
      gx[i] = gxv;
      gy[i] = gyv;
      mag[i] = Math.hypot(gxv, gyv);
    }
  }
  return { mag, gx, gy };
}

export function nonMaxSuppress(
  mag: Float32Array,
  gx: Float32Array,
  gy: Float32Array,
  w: number,
  h: number,
): Float32Array {
  const out = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const m = mag[i];
      if (m < 1e-3) continue;
      const angle = Math.atan2(gy[i], gx[i]);
      const a = ((angle + Math.PI) * 4) / Math.PI;
      const bin = Math.round(a) % 4;
      let n1 = 0;
      let n2 = 0;
      if (bin === 0) {
        n1 = mag[i - 1];
        n2 = mag[i + 1];
      } else if (bin === 1) {
        n1 = mag[(y - 1) * w + (x + 1)];
        n2 = mag[(y + 1) * w + (x - 1)];
      } else if (bin === 2) {
        n1 = mag[(y - 1) * w + x];
        n2 = mag[(y + 1) * w + x];
      } else {
        n1 = mag[(y - 1) * w + (x - 1)];
        n2 = mag[(y + 1) * w + (x + 1)];
      }
      if (m >= n1 && m >= n2) out[i] = m;
    }
  }
  return out;
}

export function hysteresisThreshold(
  mag: Float32Array,
  w: number,
  h: number,
  lowRatio = 0.06,
  highRatio = 0.16,
): Uint8Array {
  let max = 0;
  for (let i = 0; i < mag.length; i++) if (mag[i] > max) max = mag[i];
  const hi = max * highRatio;
  const lo = max * lowRatio;
  const edge = new Uint8Array(w * h);
  for (let i = 0; i < mag.length; i++) {
    if (mag[i] >= hi) edge[i] = 2;
    else if (mag[i] >= lo) edge[i] = 1;
  }
  const stack: number[] = [];
  for (let i = 0; i < edge.length; i++) if (edge[i] === 2) stack.push(i);
  while (stack.length) {
    const i = stack.pop()!;
    const x = i % w;
    const y = (i / w) | 0;
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        if (!dx && !dy) continue;
        const xx = x + dx;
        const yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
        const j = yy * w + xx;
        if (edge[j] === 1) {
          edge[j] = 2;
          stack.push(j);
        }
      }
    }
  }
  const bin = new Uint8Array(w * h);
  for (let i = 0; i < edge.length; i++) bin[i] = edge[i] === 2 ? 1 : 0;
  return bin;
}

export function findContours(bin: Uint8Array, w: number, h: number): Pt[][] {
  const visited = new Uint8Array(w * h);
  const contours: Pt[][] = [];
  const isOn = (x: number, y: number) =>
    x >= 0 && y >= 0 && x < w && y < h && bin[y * w + x] === 1;

  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const start = y * w + x;
      if (!bin[start] || visited[start]) continue;
      let boundary = false;
      for (const [dx, dy] of N8) {
        if (!isOn(x + dx, y + dy)) {
          boundary = true;
          break;
        }
      }
      if (!boundary) continue;

      const contour: Pt[] = [];
      let cx = x;
      let cy = y;
      let dir = 0;
      let guard = 0;
      const maxGuard = w * h * 2;
      do {
        const idx = cy * w + cx;
        if (!visited[idx]) {
          visited[idx] = 1;
          contour.push({ x: cx, y: cy });
        }
        let found = false;
        for (let k = 0; k < 8; k++) {
          const nd = (dir + 6 + k) % 8;
          const [dx, dy] = N8[nd];
          const nx = cx + dx;
          const ny = cy + dy;
          if (isOn(nx, ny)) {
            cx = nx;
            cy = ny;
            dir = nd;
            found = true;
            break;
          }
        }
        if (!found) break;
        guard++;
      } while ((cx !== x || cy !== y) && guard < maxGuard);

      if (contour.length >= 12) contours.push(contour);
    }
  }
  return contours;
}

function perpDist(p: Pt, a: Pt, b: Pt): number {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const len2 = dx * dx + dy * dy;
  if (len2 < 1e-9) return Math.hypot(p.x - a.x, p.y - a.y);
  const t = Math.max(
    0,
    Math.min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / len2),
  );
  return Math.hypot(p.x - (a.x + t * dx), p.y - (a.y + t * dy));
}

export function rdp(points: Pt[], epsilon: number): Pt[] {
  if (points.length < 3) return points.slice();
  let maxD = 0;
  let idx = 0;
  const end = points.length - 1;
  for (let i = 1; i < end; i++) {
    const d = perpDist(points[i], points[0], points[end]);
    if (d > maxD) {
      maxD = d;
      idx = i;
    }
  }
  if (maxD > epsilon) {
    const left = rdp(points.slice(0, idx + 1), epsilon);
    const right = rdp(points.slice(idx), epsilon);
    return left.slice(0, -1).concat(right);
  }
  return [points[0], points[end]];
}

export function perimeter(pts: Pt[]): number {
  let len = 0;
  for (let i = 1; i < pts.length; i++) {
    len += Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y);
  }
  return len;
}

/** Merge strokes whose endpoints nearly touch and are roughly collinear. */
export function mergeNearCollinear(strokes: Pt[][], joinDist = 3.5): Pt[][] {
  const alive = strokes.map((s) => s.slice());
  let merged = true;
  while (merged) {
    merged = false;
    outer: for (let i = 0; i < alive.length; i++) {
      for (let j = i + 1; j < alive.length; j++) {
        const a = alive[i];
        const b = alive[j];
        if (!a.length || !b.length) continue;
        const pairs: [number, number, boolean, boolean][] = [
          [0, 0, true, false],
          [0, b.length - 1, true, true],
          [a.length - 1, 0, false, false],
          [a.length - 1, b.length - 1, false, true],
        ];
        for (const [ai, bi, revA, revB] of pairs) {
          const pa = a[ai];
          const pb = b[bi];
          if (Math.hypot(pa.x - pb.x, pa.y - pb.y) > joinDist) continue;
          let A = revA ? a.slice().reverse() : a.slice();
          let B = revB ? b.slice().reverse() : b.slice();
          // A ends where B starts
          if (Math.hypot(A[A.length - 1].x - B[0].x, A[A.length - 1].y - B[0].y) > joinDist) {
            B = B.slice().reverse();
          }
          if (Math.hypot(A[A.length - 1].x - B[0].x, A[A.length - 1].y - B[0].y) > joinDist) {
            continue;
          }
          alive[i] = A.concat(B.slice(1));
          alive.splice(j, 1);
          merged = true;
          break outer;
        }
      }
    }
  }
  return alive.filter((s) => s.length >= 3);
}

/**
 * Downsample for CV work. Soft pencil lines are mid-gray hairlines —
 * area-averaging washes them out. Prefer darkest pixel in each block
 * (ink-preserving) so graphite ridges survive.
 */
export function downsample(
  img: RgbaImage,
  maxSide: number,
  mode: "area" | "ink-min" = "ink-min",
): RgbaImage {
  const scale = Math.min(1, maxSide / Math.max(img.width, img.height));
  if (scale >= 0.999) {
    return {
      width: img.width,
      height: img.height,
      data:
        img.data instanceof Uint8ClampedArray
          ? img.data
          : new Uint8ClampedArray(img.data),
    };
  }
  const w = Math.max(2, Math.round(img.width * scale));
  const h = Math.max(2, Math.round(img.height * scale));
  const data = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    const sy0 = Math.floor(y / scale);
    const sy1 = Math.min(img.height, Math.ceil((y + 1) / scale));
    for (let x = 0; x < w; x++) {
      const sx0 = Math.floor(x / scale);
      const sx1 = Math.min(img.width, Math.ceil((x + 1) / scale));
      let r = 0;
      let g = 0;
      let b = 0;
      let n = 0;
      let best = 9999;
      let br = 255;
      let bg = 255;
      let bb = 255;
      for (let sy = sy0; sy < sy1; sy++) {
        for (let sx = sx0; sx < sx1; sx++) {
          const si = (sy * img.width + sx) * 4;
          const rv = img.data[si];
          const gv = img.data[si + 1];
          const bv = img.data[si + 2];
          r += rv;
          g += gv;
          b += bv;
          n++;
          const luma = 0.299 * rv + 0.587 * gv + 0.114 * bv;
          if (luma < best) {
            best = luma;
            br = rv;
            bg = gv;
            bb = bv;
          }
        }
      }
      const di = (y * w + x) * 4;
      if (mode === "ink-min") {
        data[di] = br;
        data[di + 1] = bg;
        data[di + 2] = bb;
      } else {
        const d = Math.max(1, n);
        data[di] = r / d;
        data[di + 1] = g / d;
        data[di + 2] = b / d;
      }
      data[di + 3] = 255;
    }
  }
  return { width: w, height: h, data };
}

export function boardMapping(srcW: number, srcH: number): BoardRect & { scale: number } {
  const availW = VIEW_W - PAD * 2;
  const availH = VIEW_H - PAD * 2;
  const scale = Math.min(availW / srcW, availH / srcH);
  const width = srcW * scale;
  const height = srcH * scale;
  return {
    x: (VIEW_W - width) / 2,
    y: (VIEW_H - height) / 2,
    width,
    height,
    scale,
  };
}

function mapToView(pts: Pt[], board: BoardRect & { scale: number }): Pt[] {
  return pts.map((p) => ({
    x: Math.round((p.x * board.scale + board.x) * 10) / 10,
    y: Math.round((p.y * board.scale + board.y) * 10) / 10,
  }));
}

function pathStroke(
  pts: Pt[],
  width: number,
  color = GRAPHITE,
  opacity = 1,
): GuideStroke {
  return { kind: "path", d: ptsToPathD(pts), color, width, opacity };
}

function grayStats(gray: Float32Array): { mean: number; std: number } {
  let mean = 0;
  for (let i = 0; i < gray.length; i++) mean += gray[i];
  mean /= gray.length || 1;
  let varSum = 0;
  for (let i = 0; i < gray.length; i++) {
    const d = gray[i] - mean;
    varSum += d * d;
  }
  return { mean, std: Math.sqrt(varSum / (gray.length || 1)) };
}

/**
 * Convert an RGBA bitmap into multi-step stroke guides via the canonical
 * offline CV pipeline (XDoG+ETF → skeleton → geometry → hierarchy).
 */
export function rgbaToStrokeGuide(
  input: RgbaImage,
  options: PipelineOptions = {},
): PhotoGuideResult {
  if (!input.width || !input.height || input.data.length < 16) {
    return { ok: false, reason: "Could not read that image.", attemptLabel: options.label };
  }

  const maxWork = options.maxWork ?? MAX_WORK;
  const maxStrokes = options.maxStrokes ?? MAX_STROKES;
  const work = downsample(input, maxWork);
  const gray = toGray(work);
  const { std } = grayStats(gray);
  if (std < 6) {
    return {
      ok: false,
      reason:
        "Almost a lesson — this frame looks too flat. Try a clearer subject with stronger edges.",
      attemptLabel: options.label,
    };
  }

  // Scale Vision hints from input pixels → work pixels when downsampled.
  let vision = options.vision;
  if (vision && (work.width !== input.width || work.height !== input.height)) {
    const sx = work.width / input.width;
    const sy = work.height / input.height;
    const mapHint = (p: Pt): Pt => ({ x: p.x * sx, y: p.y * sy });
    let foregroundMask = vision.foregroundMask;
    if (foregroundMask && foregroundMask.length === input.width * input.height) {
      const resized = new Uint8Array(work.width * work.height);
      for (let y = 0; y < work.height; y++) {
        const iy = Math.min(input.height - 1, Math.floor(y / sy));
        for (let x = 0; x < work.width; x++) {
          const ix = Math.min(input.width - 1, Math.floor(x / sx));
          resized[y * work.width + x] = foregroundMask[iy * input.width + ix];
        }
      }
      foregroundMask = resized;
    }
    vision = {
      foregroundMask,
      engines: vision.engines,
      face: vision.face
        ? {
            leftEye: mapHint(vision.face.leftEye),
            rightEye: mapHint(vision.face.rightEye),
            nose: vision.face.nose ? mapHint(vision.face.nose) : undefined,
            mouth: vision.face.mouth ? mapHint(vision.face.mouth) : undefined,
            chin: vision.face.chin ? mapHint(vision.face.chin) : undefined,
          }
        : undefined,
      pose: vision.pose
        ? {
            neck: vision.pose.neck ? mapHint(vision.pose.neck) : undefined,
            leftShoulder: vision.pose.leftShoulder
              ? mapHint(vision.pose.leftShoulder)
              : undefined,
            rightShoulder: vision.pose.rightShoulder
              ? mapHint(vision.pose.rightShoulder)
              : undefined,
            midHip: vision.pose.midHip ? mapHint(vision.pose.midHip) : undefined,
          }
        : undefined,
      animal: vision.animal
        ? {
            nose: vision.animal.nose ? mapHint(vision.animal.nose) : undefined,
            leftEye: vision.animal.leftEye
              ? mapHint(vision.animal.leftEye)
              : undefined,
            rightEye: vision.animal.rightEye
              ? mapHint(vision.animal.rightEye)
              : undefined,
            leftEarTip: vision.animal.leftEarTip
              ? mapHint(vision.animal.leftEarTip)
              : undefined,
            rightEarTip: vision.animal.rightEarTip
              ? mapHint(vision.animal.rightEarTip)
              : undefined,
            neck: vision.animal.neck ? mapHint(vision.animal.neck) : undefined,
          }
        : undefined,
      contours: vision.contours?.map((poly) => poly.map(mapHint)),
    };
  }

  const staged = runCanonicalStages(work, {
    forceMode: options.forceMode,
    inkThrBias: options.inkThrBias ?? options.policy?.inkThrBias ?? 8,
    simplifyScale:
      options.simplifyScale ?? options.policy?.simplifyScale ?? 1,
    useGeometry: options.useGeometry === true,
    maxStrokes: options.policy?.maxStrokes
      ? Math.min(maxStrokes, options.policy.maxStrokes)
      : maxStrokes,
    vision,
    policy: options.policy,
  });

  if (!staged.strokes.length) {
    return {
      ok: false,
      reason:
        "Almost a lesson — no clear outlines yet. Try higher contrast or a simpler subject filling the frame.",
      attemptLabel: options.label,
    };
  }

  return assembleGuideFromWork(
    work,
    staged.mode,
    staged.workStrokes,
    staged.tonalStrokes,
    options.label,
    staged.strokeRadii,
  );
}

/** Build a playable guide from work-space strokes (used by iterative refine). */
export function assembleGuideFromWork(
  work: RgbaImage,
  mode: "line-art" | "photo",
  workStrokes: Pt[][],
  tonalStrokes: Pt[][],
  label?: string,
  strokeRadii?: number[],
): Extract<PhotoGuideResult, { ok: true }> {
  const board = boardMapping(work.width, work.height);
  const toGuide = (pts: Pt[], width: number, opacity = 1): GuideStroke =>
    pathStroke(mapToView(pts, board), width, GRAPHITE, opacity);

  const orderedWork = [...workStrokes, ...tonalStrokes];
  const contourCount = workStrokes.length;
  const continuousDraw: GuideStroke[] = orderedWork.map((pts, i) => {
    const isTone = i >= contourCount;
    if (isTone) return toGuide(pts, 1.0, 0.38);
    const t = contourCount <= 1 ? 0 : i / (contourCount - 1);
    // Distance-transform radius modulates pen weight (centerline ≠ hollow loop).
    const r = strokeRadii?.[i] ?? 1;
    // Line-art lessons must visibly cover the underlay pencil marks — too-thin
    // strokes looked like "missed penciling" even when the path was correct.
    const minW = mode === "line-art" ? 1.45 : 1.15;
    const maxW = mode === "line-art" ? 3.6 : 3.2;
    const width = Math.min(maxW, Math.max(minW, 1.1 + r * 0.55 - t * 0.35));
    const opacity = 0.98 - t * 0.08;
    return toGuide(pts, width, opacity);
  });

  return {
    ok: true,
    steps: [
      {
        stepNo: 1,
        caption: "Draw the subject",
        prior: [],
        draw: continuousDraw,
      },
    ],
    strokeCount: orderedWork.length,
    workSize: { w: work.width, h: work.height },
    mode,
    board: { x: board.x, y: board.y, width: board.width, height: board.height },
    workStrokes,
    tonalWorkStrokes: tonalStrokes,
    workImage: work,
    attemptLabel: label,
  };
}

/** Kept for index/generate compatibility — geometry now runs inside stages.ts. */
export function bindGeometryFitter(
  _fn: (contours: Pt[][], tol: number) => Pt[][],
): void {
  /* no-op: stages.ts imports geometry directly */
}
