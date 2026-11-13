/**
 * Canonical offline Photo Draw CV pipeline (PRO-38).
 *
 * 1 Preprocess — grayscale + CIE Lab + CLAHE local contrast
 * 2 Edges — adaptive ridges ∪ XDoG/ETF ∪ Lab color gradients
 * 3 Skeleton — Zhang–Suen + junction spur pruning
 * 4 Vectorize — 8-connect → least-cost merge → RDP → noise filter
 * 5 Least-path order + subordinate multi-coat tonal hatching
 */

import type { Pt } from "../strokeGuides/pathMath";
import {
  analyzeColorStructure,
  colorEdgeBinary,
} from "./colorStructure";
import { contoursToGeometryStrokes } from "./geometry";
import { morphologicalEnvelope, type EnvelopeHint } from "./envelope";
import {
  estimateForeground,
  foregroundBoundary,
  grabCutLite,
  maskEdgesWithForeground,
} from "./foreground";
import type { DrawPolicy } from "./subjectClass";
import { stylizePhotoToInk } from "./photoToInk";
import { leastPathOrder } from "./leastPath";
import {
  adaptiveInkMask,
  estimateInkSubjectMask,
  extractAdaptiveRidges,
  extractDeepCompactRings,
  extractFineOrnamentInk,
  recoverOrnamentsNearStrokes,
  gateEdgesByGradient,
  gateEdgesByInkDepth,
  pruneSpeckEdgesKeepFineDetail,
} from "./localContrast";
import {
  findContours,
  isLineArt,
  mergeNearCollinear,
  perimeter,
  rdp,
  distanceTransform,
  sampleStrokeRadius,
  thinSkeleton,
  toGray,
  traceSkeleton,
  type RgbaImage,
} from "./pipeline";
import {
  buildPortraitGraphiteCoats,
  buildTonalCoats,
  type TonalCoatSet,
} from "./tonalCoats";
import { mergeByTangent, pruneSkeletonSpurs } from "./topology";
import { computeGradients, xdogEtfEdges } from "./xdogEtf";

export type StrokeTier = "gesture" | "outer" | "secondary" | "detail" | "tone";

export type RankedStroke = {
  pts: Pt[];
  length: number;
  tier: StrokeTier;
  depth: number;
  bbox: { minX: number; minY: number; maxX: number; maxY: number };
  /** Half-width from distance transform along the skeleton (px). */
  radius?: number;
};

export type StagePipelineResult = {
  mode: "line-art" | "photo";
  strokes: RankedStroke[];
  workStrokes: Pt[][];
  /** Distance-transform half-widths aligned with workStrokes (variable ink). */
  strokeRadii: number[];
  /** Multi-coat hatching for dark value/color regions. */
  tonalCoats: TonalCoatSet;
  tonalStrokes: Pt[][];
  envelope: EnvelopeHint | null;
  usedVisionFg: boolean;
};

export type VisionHints = {
  /** Work-resolution FG mask (1=subject), same size as work image. */
  foregroundMask?: Uint8Array;
  /** Normalized face landmarks in work pixels (optional Loomis). */
  face?: {
    leftEye: Pt;
    rightEye: Pt;
    nose?: Pt;
    mouth?: Pt;
    chin?: Pt;
  };
  /** Body pose joints in work pixels. */
  pose?: {
    neck?: Pt;
    leftShoulder?: Pt;
    rightShoulder?: Pt;
    midHip?: Pt;
  };
  /** Cat/dog animal pose (iOS 17+ Vision). */
  animal?: {
    nose?: Pt;
    leftEye?: Pt;
    rightEye?: Pt;
    leftEarTip?: Pt;
    rightEarTip?: Pt;
    neck?: Pt;
  };
  /**
   * Optional contour polylines in the same pixel space as the image
   * these hints were built for (input or work after pipeline remap).
   */
  contours?: Pt[][];
  /** Which engines produced hints (ios engines or web-twin). */
  engines?: string[];
};

export function stage1Preprocess(img: RgbaImage): {
  gray: Float32Array;
  w: number;
  h: number;
} {
  return { gray: toGray(img), w: img.width, h: img.height };
}

/** Stamp Vision/web-twin contour polylines onto an edge mask (OR). */
export function rasterizeContoursOntoEdges(
  edges: Uint8Array,
  contours: Pt[][],
  w: number,
  h: number,
): Uint8Array {
  if (!contours.length) return edges;
  const out = edges.slice();
  const put = (x: number, y: number) => {
    const xx = Math.max(0, Math.min(w - 1, Math.round(x)));
    const yy = Math.max(0, Math.min(h - 1, Math.round(y)));
    out[yy * w + xx] = 1;
  };
  for (const poly of contours) {
    if (!poly || poly.length < 2) continue;
    for (let i = 1; i < poly.length; i++) {
      const a = poly[i - 1];
      const b = poly[i];
      const n = Math.max(2, Math.hypot(b.x - a.x, b.y - a.y) | 0);
      for (let t = 0; t <= n; t++) {
        const u = t / n;
        put(a.x + (b.x - a.x) * u, a.y + (b.y - a.y) * u);
      }
    }
  }
  return out;
}

/** Stage 2 — line-art: adaptive ink + selective ridges; photo: XDoG ∪ Lab ∪ ridges. */
export function stage2Edges(
  gray: Float32Array,
  w: number,
  h: number,
  mode: "line-art" | "photo",
  inkThrBias = 8,
  colorEdges?: Uint8Array,
  policy?: DrawPolicy,
  photoStylized = false,
): Uint8Array {
  const portraitLike =
    policy?.silhouetteFirst === true ||
    policy?.subjectClass === "portrait" ||
    !!policy?.tags?.includes("portrait");
  let merged: Uint8Array;
  if (mode === "line-art") {
    if (photoStylized && portraitLike) {
      // Synthetic portrait/animal ink from stylizePhotoToInk already consists of
      // thin 1–2px silhouette, eye/pupil rings, and NMS structural creases on white paper.
      // Avoid estimateInkSubjectMask (which drops eyes/noses < 0.6% frame area) and
      // refineInkFillsForLineArt (which hollows strokes into double-walled loops).
      const rawInk = new Uint8Array(w * h);
      const thr = Math.min(160, 125 + inkThrBias);
      for (let i = 0; i < rawInk.length; i++) {
        if (gray[i] <= thr) rawInk[i] = 1;
      }
      return pruneSpeckEdgesKeepFineDetail(
        rawInk,
        gray,
        w,
        h,
        Math.max(8, Math.round(Math.min(w, h) * 0.007)),
        4,
      );
    }
    // Trace pencil ridges + dark ink (never shade-fill refine → cell soup).
    // Goal: strokes sit ON the underlay lines the user already sees.
    const depthFloor = Math.max(30, 40 - inkThrBias);
    let subject = estimateInkSubjectMask(gray, w, h, depthFloor);
    // Widen so protruding ears / outer crown / wall edges aren't clipped before
    // ridges run — those show in underlay but were left unpenciled.
    {
      const grow = Math.max(10, Math.round(Math.min(w, h) * 0.04));
      const grown = subject.slice();
      for (let y = grow; y < h - grow; y++) {
        for (let x = grow; x < w - grow; x++) {
          if (!subject[y * w + x]) continue;
          for (let dy = -grow; dy <= grow; dy += 2) {
            for (let dx = -grow; dx <= grow; dx += 2) {
              grown[(y + dy) * w + (x + dx)] = 1;
            }
          }
        }
      }
      subject = grown;
    }
    const { ridges } = extractAdaptiveRidges(gray, w, h, { softPencil: true });
    const k = Math.max(10, 16 - inkThrBias * 0.1);
    const deepInk = adaptiveInkMask(
      gray,
      w,
      h,
      Math.max(12, Math.round(Math.min(w, h) / 32)),
      k,
      Math.max(depthFloor, 48),
    );
    const { edges: xdog } = xdogEtfEdges(gray, w, h, "line-art", inkThrBias);

    merged = new Uint8Array(w * h);
    for (let i = 0; i < merged.length; i++) {
      if (!subject[i]) continue;
      // Ridges / deep ink for graphite; XDoG only where it agrees with ink
      if (ridges[i] || deepInk[i] || (xdog[i] && deepInk[i])) merged[i] = 1;
    }
    merged = gateEdgesByInkDepth(merged, gray, Math.max(22, depthFloor - 8));
    merged = gateEdgesByGradient(merged, gray, w, h, 9);
    const fine = extractFineOrnamentInk(gray, subject, w, h, 14);
    for (let i = 0; i < merged.length; i++) {
      if (fine[i]) merged[i] = 1;
    }
    const rings = extractDeepCompactRings(gray, w, h, 70);
    for (let i = 0; i < merged.length; i++) {
      if (rings[i] && subject[i]) merged[i] = 1;
    }
    merged = gateEdgesByGradient(merged, gray, w, h, 7);
    // Marker/comic: Lab color edges + local ink when graphite detectors starve
    if (colorEdges) {
      for (let i = 0; i < merged.length; i++) {
        if (subject[i] && colorEdges[i]) merged[i] = 1;
      }
      const edgeCount = merged.reduce((a, b) => a + b, 0);
      if (edgeCount < w * h * 0.015) {
        for (let i = 0; i < merged.length; i++) {
          if (subject[i] && deepInk[i]) merged[i] = 1;
        }
        merged = gateEdgesByGradient(merged, gray, w, h, 4);
      }
    }
    return pruneSpeckEdgesKeepFineDetail(
      merged,
      gray,
      w,
      h,
      Math.max(8, Math.round(Math.min(w, h) * 0.009)),
      3,
    );
  }

  let meanGray = 0;
  for (let i = 0; i < gray.length; i++) meanGray += gray[i];
  meanGray /= gray.length || 1;
  const darkScene = meanGray < 95;

  // Portrait/animal close-ups: silhouette + strong features only.
  // CLAHE ridges turn fur into scribble (kitten failure).
  if (portraitLike) {
    const fg = grabCutLite(gray, w, h, 3);
    const boundary = foregroundBoundary(fg, w, h);
    const { enhanced } = extractAdaptiveRidges(gray, w, h, {
      softPencil: false,
    });
    const { edges } = xdogEtfEdges(
      enhanced,
      w,
      h,
      mode,
      Math.max(inkThrBias, 10),
    );
    const { mag } = computeGradients(enhanced, w, h);
    let magThr = 0;
    for (let i = 0; i < mag.length; i++) magThr += mag[i];
    // 1.55 was so high that muzzle/ear folds dropped; 1.2 keeps structure without fur flood
    magThr = (magThr / mag.length) * 1.2;
    merged = boundary.slice();
    // Dilate silhouette slightly so ear/cheek outline stays connected
    const sil = boundary.slice();
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const i = y * w + x;
        if (!fg[i] || sil[i]) continue;
        if (sil[i - 1] || sil[i + 1] || sil[i - w] || sil[i + w]) merged[i] = 1;
      }
    }
    // Strong feature edges near the subject (eyes/nose/ear folds) — not fur grain
    for (let i = 0; i < merged.length; i++) {
      if (!fg[i]) continue;
      if (edges[i] && mag[i] >= magThr) merged[i] = 1;
    }
    const rings = extractDeepCompactRings(gray, w, h);
    for (let i = 0; i < merged.length; i++) {
      if (rings[i] && fg[i]) merged[i] = 1;
    }
    // Mild close on silhouette only
    const closedP = merged.slice();
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const i = y * w + x;
        if (merged[i] || !fg[i]) continue;
        let n = 0;
        if (merged[i - 1]) n++;
        if (merged[i + 1]) n++;
        if (merged[i - w]) n++;
        if (merged[i + w]) n++;
        if (n >= 3) closedP[i] = 1;
      }
    }
    return prunePhotoEdgeSpecks(
      closedP,
      w,
      h,
      Math.max(12, Math.round(Math.min(w, h) * 0.01)),
    );
  }

  const { enhanced, ridges } = extractAdaptiveRidges(gray, w, h, {
    softPencil: false,
  });
  const { edges } = xdogEtfEdges(enhanced, w, h, mode, inkThrBias);
  merged = edges.slice();
  const skipRidges = policy?.skipRidges === true;
  // Soft faces/fur: ridges flood scribble — skip when policy says so.
  // Night scenes still need ridges for building silhouettes.
  if (!skipRidges) {
    for (let i = 0; i < merged.length; i++) {
      if (ridges[i]) merged[i] = 1;
    }
  }
  if (colorEdges && colorEdges.length === merged.length) {
    if (!darkScene && !skipRidges) {
      for (let i = 0; i < merged.length; i++) {
        if (colorEdges[i]) merged[i] = 1;
      }
    } else if (darkScene || skipRidges) {
      // Color edges only when near an existing structure hit
      for (let i = 0; i < merged.length; i++) {
        if (!colorEdges[i] || merged[i]) continue;
        const x = i % w;
        const y = (i / w) | 0;
        let near = false;
        for (let dy = -2; dy <= 2 && !near; dy++) {
          for (let dx = -2; dx <= 2; dx++) {
            const xx = x + dx;
            const yy = y + dy;
            if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
            if (edges[yy * w + xx] || (!skipRidges && ridges[yy * w + xx])) {
              near = true;
              break;
            }
          }
        }
        if (near) merged[i] = 1;
      }
    }
  }

  // Morphological close to bridge hairline gaps (photo path)
  const closed = merged.slice();
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
  const closeNeed = darkScene ? 6 : 5;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      if (merged[y * w + x]) continue;
      let n = 0;
      for (const [dx, dy] of N8) {
        if (merged[(y + dy) * w + (x + dx)]) n++;
      }
      if (n >= closeNeed) closed[y * w + x] = 1;
    }
  }
  return prunePhotoEdgeSpecks(
    closed,
    w,
    h,
    darkScene
      ? Math.max(10, Math.round(Math.min(w, h) * 0.008))
      : Math.max(8, Math.round(Math.min(w, h) * 0.006)),
  );
}

/** Drop tiny edge islands (light sparkles / sensor grain) while keeping long structure. */
export function prunePhotoEdgeSpecks(
  edges: Uint8Array,
  w: number,
  h: number,
  minArea: number,
): Uint8Array {
  const seen = new Uint8Array(w * h);
  const out = edges.slice();
  const qx: number[] = [];
  const qy: number[] = [];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const start = y * w + x;
      if (!edges[start] || seen[start]) continue;
      qx.length = 0;
      qy.length = 0;
      qx.push(x);
      qy.push(y);
      seen[start] = 1;
      let head = 0;
      const members: number[] = [start];
      while (head < qx.length) {
        const cx = qx[head];
        const cy = qy[head];
        head++;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (!dx && !dy) continue;
            const nx = cx + dx;
            const ny = cy + dy;
            if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
            const ni = ny * w + nx;
            if (!edges[ni] || seen[ni]) continue;
            seen[ni] = 1;
            qx.push(nx);
            qy.push(ny);
            members.push(ni);
          }
        }
      }
      if (members.length < minArea) {
        for (const i of members) out[i] = 0;
      }
    }
  }
  return out;
}

/**
 * Drop micro-ticks / L-noise. Keep short strokes only inside dense detail clusters.
 */
export function filterNoiseStrokes(
  polys: Pt[][],
  w: number,
  h: number,
  minLenView = 14,
): Pt[][] {
  const diag = Math.hypot(w, h);
  const minLen = Math.max(minLenView, diag * 0.016);
  const dens = new Uint16Array(w * h);
  const mark = (pts: Pt[]) => {
    for (const p of pts) {
      const x = Math.min(w - 1, Math.max(0, Math.round(p.x)));
      const y = Math.min(h - 1, Math.max(0, Math.round(p.y)));
      dens[y * w + x]++;
    }
  };
  for (const p of polys) mark(p);

  const densityNear = (pts: Pt[]) => {
    let hits = 0;
    for (const p of pts) {
      const x = Math.min(w - 1, Math.max(0, Math.round(p.x)));
      const y = Math.min(h - 1, Math.max(0, Math.round(p.y)));
      for (let dy = -3; dy <= 3; dy++) {
        for (let dx = -3; dx <= 3; dx++) {
          const xx = x + dx;
          const yy = y + dy;
          if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
          hits += dens[yy * w + xx];
        }
      }
    }
    return hits / Math.max(1, pts.length);
  };

  const isRightAngleTick = (pts: Pt[]) => {
    if (pts.length < 3 || pts.length > 4) return false;
    const len = perimeter(pts);
    if (len > minLen * 1.4) return false;
    // Two short orthogonal segments
    let orth = 0;
    for (let i = 1; i < pts.length - 1; i++) {
      const ax = pts[i].x - pts[i - 1].x;
      const ay = pts[i].y - pts[i - 1].y;
      const bx = pts[i + 1].x - pts[i].x;
      const by = pts[i + 1].y - pts[i].y;
      const la = Math.hypot(ax, ay);
      const lb = Math.hypot(bx, by);
      if (la < 1 || lb < 1) continue;
      const dot = (ax * bx + ay * by) / (la * lb);
      if (Math.abs(dot) < 0.25) orth++;
    }
    return orth >= 1;
  };

  return polys.filter((pts) => {
    if (pts.length < 2) return false;
    const len = perimeter(pts);
    const densN = densityNear(pts);
    // Lonely right-angle ticks are usually soft-fill medial junk (hands).
    // Clustered ticks are real ornaments (trunk hatch, filigree corners).
    if (isRightAngleTick(pts) && densN < 6) return false;
    if (len >= minLen) return true;
    // Short ornament fragments in dense clusters (crown jewels, hatch)
    if (densN >= 6 && len >= minLen * 0.35 && pts.length >= 2) return true;
    return densN >= 8 && len >= minLen * 0.5;
  });
}

/**
 * Photo-only: drop high-frequency scribble polylines (night lights, fur noise)
 * while keeping long coherent silhouettes and feature contours.
 */
export function filterPhotoScribbleStrokes(
  polys: Pt[][],
  w: number,
  h: number,
  gray: Float32Array,
): Pt[][] {
  if (polys.length < 6) return polys;
  const diag = Math.hypot(w, h);
  let meanGray = 0;
  for (let i = 0; i < gray.length; i++) meanGray += gray[i];
  meanGray /= gray.length || 1;
  const darkScene = meanGray < 95;
  // Daytime portraits need short smooth feature marks (eyes, nose V, whiskers)
  const minKeep = darkScene ? diag * 0.055 : diag * 0.014;

  const jagginess = (pts: Pt[]) => {
    if (pts.length < 4) return 0;
    let turns = 0;
    let segs = 0;
    for (let i = 1; i < pts.length - 1; i++) {
      const ax = pts[i].x - pts[i - 1].x;
      const ay = pts[i].y - pts[i - 1].y;
      const bx = pts[i + 1].x - pts[i].x;
      const by = pts[i + 1].y - pts[i].y;
      const la = Math.hypot(ax, ay);
      const lb = Math.hypot(bx, by);
      if (la < 0.8 || lb < 0.8) continue;
      segs++;
      const dot = (ax * bx + ay * by) / (la * lb);
      if (dot < 0.4) turns++;
    }
    return segs ? turns / segs : 0;
  };

  const isTinyClosedLoop = (pts: Pt[]) => {
    if (pts.length < 6) return false;
    const first = pts[0];
    const last = pts[pts.length - 1];
    if (Math.hypot(first.x - last.x, first.y - last.y) > diag * 0.02) return false;
    let minX = Infinity,
      minY = Infinity,
      maxX = -Infinity,
      maxY = -Infinity;
    for (const p of pts) {
      minX = Math.min(minX, p.x);
      minY = Math.min(minY, p.y);
      maxX = Math.max(maxX, p.x);
      maxY = Math.max(maxY, p.y);
    }
    const bw = maxX - minX;
    const bh = maxY - minY;
    // Night window sparkles → tiny closed blobs
    return bw < diag * 0.035 && bh < diag * 0.035;
  };

  return polys.filter((pts) => {
    const len = perimeter(pts);
    const jag = jagginess(pts);
    if (darkScene && isTinyClosedLoop(pts)) return false;
    // Long but frantic polylines are night-light scribble — drop them
    if (jag > 0.55 && len < minKeep * 2.2) return false;
    if (len >= minKeep && jag <= 0.55) return true;
    // Keep short but smooth feature marks (eyes, windows); drop jittery ticks
    if (jag > 0.4 && len < minKeep * 1.6) return false;
    if (darkScene && len < minKeep * 0.95) return false;
    return len >= minKeep * 0.55 && pts.length >= 3 && jag <= 0.45;
  });
}

/**
 * Drop soft-shade / fleck strokes. Subject mass is inferred from long strokes
 * and optionally from deep-ink on the original gray — works for any upload,
 * not a particular character or scene.
 */
export function filterBackgroundShadeStrokes(
  polys: Pt[][],
  w: number,
  h: number,
  gray?: Float32Array,
): Pt[][] {
  if (polys.length < 8) return polys;
  const diag = Math.hypot(w, h);
  const lengths = polys.map((p) => perimeter(p));
  const sorted = lengths.slice().sort((a, b) => b - a);
  const anchorLen = Math.max(40, sorted[Math.min(10, sorted.length - 1)] ?? 40);

  // Seed mask from long structural strokes
  const mask = new Uint8Array(w * h);
  const stamp = (pts: Pt[], r: number) => {
    for (const p of pts) {
      const x = Math.round(p.x);
      const y = Math.round(p.y);
      for (let dy = -r; dy <= r; dy++) {
        for (let dx = -r; dx <= r; dx++) {
          const xx = x + dx;
          const yy = y + dy;
          if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
          mask[yy * w + xx] = 1;
        }
      }
    }
  };
  for (let i = 0; i < polys.length; i++) {
    if (lengths[i] >= anchorLen) stamp(polys[i], 2);
  }
  // Subject halo from long structural strokes (tight — shade flecks stay out).
  // Eyes/pupils are kept later via closed-ring / high-grad rules, not this halo.
  const dilateR = Math.max(6, Math.round(diag * 0.016));
  const subject = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!mask[y * w + x]) continue;
      for (let dy = -dilateR; dy <= dilateR; dy += 2) {
        for (let dx = -dilateR; dx <= dilateR; dx += 2) {
          const xx = x + dx;
          const yy = y + dy;
          if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
          subject[yy * w + xx] = 1;
        }
      }
    }
  }

  let paper = 255;
  let localContrast: Float32Array | null = null;
  if (gray && gray.length === w * h) {
    const sortedG = Float32Array.from(gray);
    sortedG.sort();
    paper = sortedG[Math.min(sortedG.length - 1, Math.floor(sortedG.length * 0.9))];
    // Local |gray - blur| — real contours are high; soft shade / fill interiors are low
    localContrast = new Float32Array(w * h);
    const r = Math.max(3, Math.round(Math.min(w, h) / 80));
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let sum = 0;
        let n = 0;
        for (let dy = -r; dy <= r; dy++) {
          for (let dx = -r; dx <= r; dx++) {
            const xx = x + dx;
            const yy = y + dy;
            if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
            sum += gray[yy * w + xx];
            n++;
          }
        }
        const mean = sum / Math.max(1, n);
        localContrast[y * w + x] = Math.abs(gray[y * w + x] - mean);
      }
    }
  }

  const fracOnSubject = (pts: Pt[]) => {
    let hit = 0;
    for (const p of pts) {
      const x = Math.min(w - 1, Math.max(0, Math.round(p.x)));
      const y = Math.min(h - 1, Math.max(0, Math.round(p.y)));
      if (subject[y * w + x]) hit++;
    }
    return hit / Math.max(1, pts.length);
  };

  const medianDepth = (pts: Pt[]) => {
    if (!gray || gray.length !== w * h) return 999;
    const ds: number[] = [];
    for (const p of pts) {
      const x = Math.min(w - 1, Math.max(0, Math.round(p.x)));
      const y = Math.min(h - 1, Math.max(0, Math.round(p.y)));
      ds.push(paper - gray[y * w + x]);
    }
    ds.sort((a, b) => a - b);
    return ds[(ds.length / 2) | 0] ?? 0;
  };

  const medianContrast = (pts: Pt[]) => {
    if (!localContrast) return 999;
    const cs: number[] = [];
    for (const p of pts) {
      const x = Math.min(w - 1, Math.max(0, Math.round(p.x)));
      const y = Math.min(h - 1, Math.max(0, Math.round(p.y)));
      cs.push(localContrast[y * w + x]);
    }
    cs.sort((a, b) => a - b);
    return cs[(cs.length / 2) | 0] ?? 0;
  };

  // Sobel-ish gradient on original gray — interior fill ticks score low
  let gradMag: Float32Array | null = null;
  if (gray && gray.length === w * h) {
    gradMag = new Float32Array(w * h);
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const i = y * w + x;
        const gx = Math.abs(gray[i + 1] - gray[i - 1]) * 0.5;
        const gy = Math.abs(gray[i + w] - gray[i - w]) * 0.5;
        gradMag[i] = gx + gy;
      }
    }
  }
  const medianGrad = (pts: Pt[]) => {
    if (!gradMag) return 999;
    const gs: number[] = [];
    for (const p of pts) {
      const x = Math.min(w - 1, Math.max(0, Math.round(p.x)));
      const y = Math.min(h - 1, Math.max(0, Math.round(p.y)));
      gs.push(gradMag[y * w + x]);
    }
    gs.sort((a, b) => a - b);
    return gs[(gs.length / 2) | 0] ?? 0;
  };

  const isGuideCrosshair = (pts: Pt[], len: number) => {
    // Long nearly-axis-aligned strokes that span much of the frame (photo guides)
    if (pts.length < 2 || len < diag * 0.35) return false;
    let minX = Infinity,
      minY = Infinity,
      maxX = -Infinity,
      maxY = -Infinity;
    for (const p of pts) {
      minX = Math.min(minX, p.x);
      minY = Math.min(minY, p.y);
      maxX = Math.max(maxX, p.x);
      maxY = Math.max(maxY, p.y);
    }
    const bw = maxX - minX;
    const bh = maxY - minY;
    const horiz = bh < 6 && bw > w * 0.45;
    const vert = bw < 6 && bh > h * 0.45;
    return horiz || vert;
  };

  const minKeepLen = Math.max(20, diag * 0.02);

  // Local stroke density — facial filigree is short but clustered; shade flecks are lonely
  const dens = new Uint16Array(w * h);
  for (const pts of polys) {
    for (const p of pts) {
      const x = Math.min(w - 1, Math.max(0, Math.round(p.x)));
      const y = Math.min(h - 1, Math.max(0, Math.round(p.y)));
      dens[y * w + x]++;
    }
  }
  const densityNear = (pts: Pt[]) => {
    let hits = 0;
    for (const p of pts) {
      const x = Math.min(w - 1, Math.max(0, Math.round(p.x)));
      const y = Math.min(h - 1, Math.max(0, Math.round(p.y)));
      for (let dy = -4; dy <= 4; dy++) {
        for (let dx = -4; dx <= 4; dx++) {
          const xx = x + dx;
          const yy = y + dy;
          if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
          hits += dens[yy * w + xx];
        }
      }
    }
    return hits / Math.max(1, pts.length);
  };

  const endpointGap = (pts: Pt[]) => {
    if (pts.length < 2) return Infinity;
    const a = pts[0];
    const b = pts[pts.length - 1];
    return Math.hypot(a.x - b.x, a.y - b.y);
  };
  const nearlyClosed = (pts: Pt[], len: number) => {
    if (pts.length < 4) return false;
    const gap = endpointGap(pts);
    // Pupils/iris rings often leave a skeleton gap at the join
    return gap <= Math.max(6, Math.min(minKeepLen * 0.55, len * 0.28));
  };
  const semiClosed = (pts: Pt[], len: number) => {
    if (pts.length < 4) return false;
    return endpointGap(pts) <= Math.max(10, len * 0.5);
  };

  return polys.filter((pts, i) => {
    const len = lengths[i];
    if (isGuideCrosshair(pts, len)) return false;
    const on = fracOnSubject(pts);
    const depth = medianDepth(pts);
    const contrast = medianContrast(pts);
    const densN = densityNear(pts);
    const grad = medianGrad(pts);
    const closed = nearlyClosed(pts, len);
    const almost = semiClosed(pts, len);
    // Soft graphite wash: shallow AND low local contrast (fine hatch can be shallow)
    if (depth < 16 && contrast < 10) return false;
    if (depth < 28 && contrast < 8 && grad < 10) return false;
    // Fill-interior ticks (shaded skin/hands) — low image gradient
    if (grad < 7) return false;
    if (grad < 10 && len < minKeepLen * 1.8 && !closed && densN < 11) {
      return false;
    }
    // Lonely short open shade flecks (not strong eye arcs / ornament clusters)
    if (
      len < minKeepLen * 1.3 &&
      !closed &&
      !almost &&
      densN < 5 &&
      grad < 18 &&
      contrast < 14
    ) {
      return false;
    }
    if (len < minKeepLen && densN < 3 && grad < 18 && contrast < 12) {
      return false;
    }
    // Long structural strokes on/near subject
    if (len >= anchorLen * 0.75 && on >= 0.2 && grad >= 8) return true;
    // Closed / near-closed detail (pupils, beads, iris) — no halo required
    if (
      (closed || almost) &&
      grad >= 8 &&
      depth >= 22 &&
      len >= minKeepLen * 0.25
    ) {
      return true;
    }
    // Fine ornaments: short, high local-contrast, clustered (crown jewels,
    // trunk hatch, filigree) — scene-agnostic; soft wash fails contrast/densN
    if (
      len >= 5 &&
      len < minKeepLen * 2.2 &&
      contrast >= 13 &&
      grad >= 11 &&
      densN >= 5 &&
      depth >= 16 &&
      on >= 0.4
    ) {
      return true;
    }
    // Strong dark contour arcs (eyelids, lashes, brows, pupil shards)
    if (len >= 12 && grad >= 16 && depth >= 70 && densN >= 2.5) return true;
    if (len >= minKeepLen * 0.45 && grad >= 12 && depth >= 32 && densN >= 3) {
      return true;
    }
    // Other detail must sit on the long-stroke subject halo
    if (on >= 0.5 && densN >= 7 && grad >= 9 && depth >= 28) return true;
    if (
      on >= 0.5 &&
      grad >= 10 &&
      (contrast >= 10 || densN >= 6) &&
      depth >= 28
    ) {
      return true;
    }
    return false;
  });
}

/** Stage 3 — Zhang–Suen + junction spur pruning. */
export function stage3Skeleton(
  edges: Uint8Array,
  w: number,
  h: number,
): Uint8Array {
  const skel = thinSkeleton(edges, w, h, 72);
  // Minimal spur prune — trunk hatch ticks are short true spurs off the
  // centerline; pruning ≥3px erased them.
  const spur = Math.max(1, Math.round(Math.min(w, h) * 0.002));
  return pruneSkeletonSpurs(skel, w, h, spur);
}

/** Stage 4 — trace → tangent merge → RDP → selective geometry → noise filter. */
export function stage4Vectorize(
  skel: Uint8Array,
  edges: Uint8Array,
  w: number,
  h: number,
  simplifyScale: number,
  useGeometry: boolean,
): Pt[][] {
  const diag = Math.hypot(w, h);
  // Keep short junction fragments (grid cells / facial loops / ornaments /
  // trunk hatch ticks). Too-high minPerim erased crown jewels + cross-lines.
  const minKeep = Math.max(3, diag * 0.002);
  const minPerim = Math.max(5, diag * 0.004);
  let traced = traceSkeleton(skel, w, h, { throughJunctions: true });
  if (traced.length < 8) {
    traced = traced.concat(findContours(edges, w, h));
  }
  // Tangent merge across nearby endpoints so natural contours become longer
  // strokes — keep join modest so we don't snake the whole drawing.
  traced = mergeByTangent(
    traced.filter((c) => c.length >= 2 && perimeter(c) >= minKeep),
    Math.max(4.0, diag / 160),
    0.78,
  );
  const eps = Math.max(0.35, diag / 1400) * simplifyScale;
  let polys = mergeNearCollinear(
    traced
      .map((c) => rdp(c, eps))
      .filter((c) => c.length >= 2 && perimeter(c) >= minPerim),
    Math.max(2.6, diag / 220),
  );
  // Second tangent pass after simplify
  polys = mergeByTangent(polys, Math.max(3.5, diag / 180), 0.8);
  // PRO-38: skip aggressive circle/ellipse replacement — soft pencil needs
  // faithful polylines, not invented "block-in" ovals that erase the subject.
  if (useGeometry && polys.length) {
    const long = polys.filter((p) => perimeter(p) >= diag * 0.12);
    const short = polys.filter((p) => perimeter(p) < diag * 0.12);
    if (long.length) {
      const fitTol = Math.max(0.9, diag / 480) * simplifyScale;
      const { strokes } = contoursToGeometryStrokes(long, fitTol);
      const keptFits: Pt[][] = [];
      for (let i = 0; i < strokes.length; i++) {
        const s = strokes[i];
        const src = long[Math.min(i, long.length - 1)];
        const box = bboxOf(s);
        const bw = box.maxX - box.minX;
        const bh = box.maxY - box.minY;
        // Reject near-circular fits that balloon vs source length
        const lenRatio = perimeter(s) / Math.max(1, perimeter(src));
        const circular = Math.abs(bw - bh) / Math.max(bw, bh, 1) < 0.2;
        if (circular && (bw > w * 0.35 || bh > h * 0.35 || lenRatio > 1.35)) {
          keptFits.push(src); // keep original polyline
          continue;
        }
        if (bw < w * 0.75 && bh < h * 0.75 && perimeter(s) >= minPerim) {
          keptFits.push(s);
        } else {
          keptFits.push(src);
        }
      }
      polys = mergeNearCollinear([...keptFits, ...short], Math.max(2.0, diag / 300));
    }
  }
  polys = filterNoiseStrokes(polys, w, h, 14);
  return polys.sort((a, b) => perimeter(b) - perimeter(a));
}

function bboxOf(pts: Pt[]) {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const p of pts) {
    minX = Math.min(minX, p.x);
    minY = Math.min(minY, p.y);
    maxX = Math.max(maxX, p.x);
    maxY = Math.max(maxY, p.y);
  }
  return { minX, minY, maxX, maxY };
}

function contains(
  a: { minX: number; minY: number; maxX: number; maxY: number },
  b: { minX: number; minY: number; maxX: number; maxY: number },
  pad = 2,
): boolean {
  return (
    a.minX <= b.minX + pad &&
    a.minY <= b.minY + pad &&
    a.maxX >= b.maxX - pad &&
    a.maxY >= b.maxY - pad &&
    a.maxX - a.minX > b.maxX - b.minX + 4 &&
    a.maxY - a.minY > b.maxY - b.minY + 4
  );
}

export function stage5HierarchicalSort(polys: Pt[][]): RankedStroke[] {
  const items = polys.map((pts) => {
    const box = bboxOf(pts);
    const length = perimeter(pts);
    return { pts, length, bbox: box, depth: 0 };
  });
  for (let i = 0; i < items.length; i++) {
    let depth = 0;
    for (let j = 0; j < items.length; j++) {
      if (i === j) continue;
      if (contains(items[j].bbox, items[i].bbox)) depth++;
    }
    items[i].depth = depth;
  }
  const maxLen = items[0]?.length || 1;
  const ranked: RankedStroke[] = items.map((it) => {
    let tier: StrokeTier = "detail";
    if (it.depth === 0 && it.length >= maxLen * 0.35) tier = "gesture";
    else if (it.depth === 0) tier = "outer";
    else if (it.depth <= 2 && it.length >= maxLen * 0.08) tier = "secondary";
    else tier = "detail";
    return { pts: it.pts, length: it.length, tier, depth: it.depth, bbox: it.bbox };
  });
  const tierOrder: Record<StrokeTier, number> = {
    gesture: 0,
    outer: 1,
    secondary: 2,
    detail: 3,
    tone: 4,
  };
  ranked.sort((a, b) => {
    const td = tierOrder[a.tier] - tierOrder[b.tier];
    if (td) return td;
    if (a.depth !== b.depth) return a.depth - b.depth;
    return b.length - a.length;
  });
  // Pencil travel order is applied later via leastPathOrder in runCanonicalStages.
  return ranked;
}

/** Run stages 1–5 on a work-resolution RGBA image. */
export function runCanonicalStages(
  work: RgbaImage,
  opts: {
    forceMode?: "line-art" | "photo" | "auto";
    inkThrBias?: number;
    simplifyScale?: number;
    useGeometry?: boolean;
    maxStrokes?: number;
    vision?: VisionHints;
    policy?: DrawPolicy;
  } = {},
): StagePipelineResult {
  const { gray, w, h } = stage1Preprocess(work);
  const color = analyzeColorStructure(work);
  const autoLine = isLineArt(gray, work);
  const mode: "line-art" | "photo" =
    opts.forceMode === "line-art"
      ? "line-art"
      : opts.forceMode === "photo"
        ? "photo"
        : autoLine
          ? "line-art"
          : "photo";

  const policy = opts.policy;
  // Real photos: first convert to a clean ink drawing, then extract strokes
  // like line-art. Tracing raw photo texture (fur/water/neon) scribbles.
  let edgeGray = gray;
  let photoStylized = false;
  if (mode === "photo") {
    const ink = stylizePhotoToInk(work, policy);
    edgeGray = ink.inkGray;
    photoStylized = true;
  }
  // Color line-art (marker/comic) needs Lab edges — graphite ridges alone miss
  // flat fills. Do NOT inject Lab edges into graphite scans (Ganesha): they
  // steal stroke budget with shade noise and leave underlay lines unpenciled.
  let meanChroma = 0;
  {
    let chroma = 0;
    let samples = 0;
    const step = Math.max(1, Math.floor((w * h) / 4000));
    for (let i = 0; i < w * h; i += step) {
      const o = i * 4;
      const r = work.data[o];
      const g = work.data[o + 1];
      const b = work.data[o + 2];
      chroma += Math.max(r, g, b) - Math.min(r, g, b);
      samples++;
    }
    meanChroma = chroma / Math.max(1, samples);
  }
  const colorBin =
    !photoStylized &&
    ((mode === "line-art" && meanChroma > 20) ||
      (mode === "photo" && !policy?.skipRidges))
      ? colorEdgeBinary(color.edgeMag, w, h, mode === "line-art" ? 0.32 : 0.38)
      : undefined;
  let edges = stage2Edges(
    edgeGray,
    w,
    h,
    // Stylized ink is graphite-on-paper — use the line-art ink extractor.
    photoStylized ? "line-art" : mode,
    opts.inkThrBias ?? policy?.inkThrBias ?? (photoStylized ? 10 : 8),
    colorBin,
    policy,
    photoStylized,
  );
  // Portrait stylize already embeds silhouette; keep silhouetteFirst policy
  // from also re-running fur-prone photo branch (mode was photo but edges
  // came from line-art path above).

  let usedVisionFg = false;
  const twinOnly =
    !!opts.vision?.engines?.length &&
    opts.vision.engines.every((e) => e.startsWith("web-twin"));
  const realVisionFg =
    !twinOnly &&
    opts.vision?.foregroundMask &&
    opts.vision.foregroundMask.length === w * h;
  if (realVisionFg) {
    edges = maskEdgesWithForeground(edges, opts.vision!.foregroundMask!, w, h);
    usedVisionFg = true;
  } else if (!photoStylized && mode !== "line-art") {
    // Soft FG for photos only. Line-art already subject-masked in stage2 —
    // a second FG pass was clipping ears / crown / trunk strokes that the
    // underlay still shows (Ganesha "missed penciling" report).
    const fg = estimateForeground(work, mode);
    const before = edges.reduce((a, b) => a + b, 0);
    const masked = maskEdgesWithForeground(edges, fg, w, h);
    const after = masked.reduce((a, b) => a + b, 0);
    if (after >= before * 0.45 && after >= 80) {
      edges = masked;
    }
  }

  // Real iOS Vision contours only — twin contour rings inflated fur scribbles.
  if (!twinOnly && opts.vision?.contours?.length) {
    edges = rasterizeContoursOntoEdges(edges, opts.vision.contours, w, h);
  }

  const skel = stage3Skeleton(edges, w, h);
  // Distance map on thick edges BEFORE skeleton collapse → variable stroke weight
  const edgeDist = distanceTransform(edges, w, h);
  let polys = stage4Vectorize(
    skel,
    edges,
    w,
    h,
    opts.simplifyScale ?? policy?.simplifyScale ?? 1,
    // Default OFF — geometry ovals were inventing "block-in" circles (PRO-38).
    opts.useGeometry === true,
  );
  // Least-cost merge of fragments (color-aware join cost)
  const pathCtx = { L: color.L, w, h };
  // Do not MST-merge after vectorize — it snakes grids into sparse chords.
  // Line-art keeps shorter ornament ticks; photos drop scribble ticks harder.
  const scribbleStrict = policy?.strictScribble === true;
  polys = filterNoiseStrokes(
    polys,
    w,
    h,
    mode === "line-art" ? 10 : scribbleStrict ? 22 : 16,
  );
  if (mode === "photo") {
    polys = filterPhotoScribbleStrokes(polys, w, h, gray);
    if (scribbleStrict) {
      // Second pass: drop lonely ticks, but keep short facial marks (eyes,
      // nostrils, whisker roots). Portraits used to wipe those at 0.028*diag.
      const diag = Math.hypot(w, h);
      const portraitPass =
        policy?.subjectClass === "portrait" ||
        !!policy?.tags?.includes("portrait") ||
        !!policy?.silhouetteFirst;
      const minLen = diag * (portraitPass ? 0.012 : 0.028);
      const minPts = portraitPass ? 5 : 8;
      polys = polys.filter((p) => perimeter(p) >= minLen || p.length >= minPts);
    }
  }
  const maxStrokes =
    opts.maxStrokes ?? policy?.maxStrokes ?? 420;
  // Drop soft-shade / fleck strokes (any subject — depth + ink mass)
  if (mode === "line-art") {
    polys = filterBackgroundShadeStrokes(polys, w, h, gray);
    // Recover light oriented ornaments (trunk hatch, filigree stems) that
    // sit near kept structural strokes — soft wash away from structure stays out.
    const recovered = recoverOrnamentsNearStrokes(gray, w, h, polys, 14);
    if (recovered.length) {
      polys = polys.concat(recovered);
      // Re-filter lightly so lonely recovered grain still drops
      polys = filterNoiseStrokes(polys, w, h, 6);
    }
  }

  const rankedAll = stage5HierarchicalSort(polys).map((s) => ({
    ...s,
    radius: sampleStrokeRadius(s.pts, edgeDist, w, h),
  }));
  // Budget AFTER ranking so detail/ornament aren't silently dropped when
  // outer contours already fill a low maxStrokes cap (dense line-art miss).
  let ranked = rankedAll;
  if (rankedAll.length > maxStrokes) {
    if (mode === "line-art") {
      const primary = rankedAll.filter(
        (s) => s.tier === "gesture" || s.tier === "outer" || s.tier === "secondary",
      );
      const detail = rankedAll.filter(
        (s) => s.tier === "detail" || s.tier === "tone",
      );
      // Keep ~half the budget for ornament (ears, jewelry, trunk folds) that
      // the underlay still shows after outer contours fill the cap.
      const primaryKeep = Math.min(primary.length, Math.floor(maxStrokes * 0.52));
      const detailKeep = Math.min(detail.length, maxStrokes - primaryKeep);
      ranked = [
        ...primary.slice(0, primaryKeep),
        ...detail.slice(0, detailKeep),
      ];
      // If detail was scarce, fill remainder with leftover primary
      if (ranked.length < maxStrokes) {
        ranked = ranked.concat(
          primary.slice(primaryKeep, primaryKeep + (maxStrokes - ranked.length)),
        );
      }
    } else {
      ranked = rankedAll.slice(0, maxStrokes);
    }
  }
  // Order for continuous animation; pathCtx only affects travel cost, not geometry
  const workOrdered = leastPathOrder(
    ranked.map((s) => s.pts),
    pathCtx,
  );
  // Sample AFTER order (leastPath may reverse polylines)
  const strokeRadii = workOrdered.map((pts) =>
    sampleStrokeRadius(pts, edgeDist, w, h),
  );

  // Soft gray paper shade ≠ hatching. Portraits get dense ETF graphite coats
  // on the photo subject so a cat reads as a cat without underlay.
  const deepInkFrac =
    color.deepMask.reduce((a, b) => a + b, 0) / Math.max(1, color.deepMask.length);
  const portraitGraphite =
    mode === "photo" &&
    (policy?.subjectClass === "portrait" ||
      !!policy?.tags?.includes("portrait") ||
      !!policy?.silhouetteFirst);
  const wantTonal =
    portraitGraphite || mode === "photo" || deepInkFrac >= 0.02;
  let tonalCoats: TonalCoatSet = {
    mid: [] as Pt[][],
    dark: [] as Pt[][],
    deep: [] as Pt[][],
  };
  if (wantTonal) {
    if (portraitGraphite) {
      const fg = grabCutLite(gray, w, h, 3);
      tonalCoats = buildPortraitGraphiteCoats(color, fg, { budget: 560 });
    } else {
      tonalCoats = buildTonalCoats(color, {
        budget: mode === "line-art" ? 24 : 70,
        clearBorderFrac: 0.05,
        minSegLen: 18,
      });
    }
  }
  const tonalMerged = [
    ...tonalCoats.mid,
    ...tonalCoats.dark,
    ...tonalCoats.deep,
  ];
  const tonalRaw = !wantTonal
    ? []
    : portraitGraphite
      ? // Keep short fur ticks — shade wipe was killing tone (UI tone≈0.01)
        filterNoiseStrokes(tonalMerged, w, h, 6)
      : filterBackgroundShadeStrokes(
          filterNoiseStrokes(tonalMerged, w, h, 16),
          w,
          h,
          gray,
        );
  const tonalStrokes = leastPathOrder(tonalRaw, pathCtx);
  const envelope = morphologicalEnvelope(edges, w, h);

  return {
    mode,
    strokes: ranked,
    workStrokes: workOrdered,
    strokeRadii,
    tonalCoats: {
      mid: tonalCoats.mid,
      dark: tonalCoats.dark,
      deep: tonalCoats.deep,
    },
    tonalStrokes,
    envelope,
    usedVisionFg,
  };
}
