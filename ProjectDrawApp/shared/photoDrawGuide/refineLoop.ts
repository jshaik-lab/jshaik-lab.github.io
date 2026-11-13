/**
 * Internal generate → score → keep-correct → refine-misses loop.
 *
 * Bounded by rounds + wall-clock so we never spin forever. If confidence
 * never clears the gate, fail closed ("needs more training") instead of
 * showing a weak animation.
 */

import type { Pt } from "../strokeGuides/pathMath";
import {
  assembleGuideFromWork,
  perimeter,
  rgbaToStrokeGuide,
  type PhotoGuideResult,
  type PipelineOptions,
  type RgbaImage,
} from "./pipeline";
import { leastPathOrder } from "./leastPath";
import type { VisionHints } from "./stages";
import {
  buildEdgeSkeleton,
  dilate,
  rasterizePolylines,
  validateWorkStrokes,
  type ValidationScores,
} from "./validate";

export type RefineBudget = {
  maxRounds: number;
  maxMs: number;
  /** Accept animation only at/above this confidence. */
  minConfidence: number;
  /** Stop refining when improvement is tinier than this. */
  plateauEps: number;
};

export type RefineProgress = {
  phase: string;
  attempt: number;
  confidence?: number;
  round?: number;
};

export type RefineCandidate = {
  result: Extract<PhotoGuideResult, { ok: true }>;
  validation: ValidationScores;
  confidence: number;
  label: string;
};

const DEFAULT_PHOTO_BUDGET: RefineBudget = {
  maxRounds: 4,
  maxMs: 14_000,
  minConfidence: 0.38,
  plateauEps: 0.008,
};

const DEFAULT_LINEART_BUDGET: RefineBudget = {
  maxRounds: 3,
  maxMs: 10_000,
  minConfidence: 0.38,
  plateauEps: 0.008,
};

export function defaultRefineBudget(
  mode: "line-art" | "photo",
): RefineBudget {
  return mode === "line-art" ? { ...DEFAULT_LINEART_BUDGET } : { ...DEFAULT_PHOTO_BUDGET };
}

/** Single scalar in [0,1] for accept / compare decisions. */
export function matchConfidence(v: ValidationScores): number {
  const q = Math.max(0, Math.min(1, v.quadrantCoverage || 0));
  // Precision weighted harder — high-recall fur scribble must not look “confident”
  const base =
    0.35 * (v.dice || 0) +
    0.25 * (v.edgeRecall || 0) +
    0.4 * (v.edgePrecision || 0);
  return Math.max(0, Math.min(1, base * (0.72 + 0.28 * q)));
}

function strokePrecisionOnRef(
  pts: Pt[],
  ref: Uint8Array,
  w: number,
  h: number,
): number {
  const raw = rasterizePolylines([pts], w, h, 1.4);
  const stroke = dilate(raw, w, h, 2);
  let ink = 0;
  let hit = 0;
  for (let i = 0; i < stroke.length; i++) {
    if (!stroke[i]) continue;
    ink++;
    if (ref[i]) hit++;
  }
  return ink ? hit / ink : 0;
}

function strokeCoversMiss(
  pts: Pt[],
  miss: Uint8Array,
  w: number,
  h: number,
): boolean {
  const raw = rasterizePolylines([pts], w, h, 1.6);
  for (let i = 0; i < raw.length; i++) {
    if (raw[i] && miss[i]) return true;
  }
  return false;
}

/**
 * Keep strokes that already align with the source; drop noisy false positives.
 * Long structural strokes get a slightly softer keep threshold.
 */
export function keepCorrectStrokes(
  strokes: Pt[][],
  work: RgbaImage,
  mode: "line-art" | "photo",
): Pt[][] {
  const { skeleton } = buildEdgeSkeleton(work, mode);
  const w = work.width;
  const h = work.height;
  const ref = dilate(skeleton, w, h, mode === "line-art" ? 3 : 2);
  const diag = Math.hypot(w, h);
  return strokes.filter((pts) => {
    if (pts.length < 2) return false;
    const len = perimeter(pts);
    const prec = strokePrecisionOnRef(pts, ref, w, h);
    if (len >= diag * 0.08) return prec >= 0.38;
    if (len >= diag * 0.03) return prec >= 0.48;
    return prec >= 0.58;
  });
}

/** Pixels in the GT that current strokes still miss. */
export function buildMissMask(
  work: RgbaImage,
  strokes: Pt[][],
  mode: "line-art" | "photo",
): Uint8Array {
  const { skeleton } = buildEdgeSkeleton(work, mode);
  const w = work.width;
  const h = work.height;
  const ref = dilate(skeleton, w, h, mode === "line-art" ? 3 : 2);
  const stroke = dilate(
    rasterizePolylines(strokes, w, h, mode === "line-art" ? 1.7 : 1.5),
    w,
    h,
    mode === "line-art" ? 4 : 3,
  );
  const miss = new Uint8Array(w * h);
  for (let i = 0; i < miss.length; i++) {
    if (ref[i] && !stroke[i]) miss[i] = 1;
  }
  return miss;
}

function nearDuplicate(a: Pt[], b: Pt[], tol: number): boolean {
  if (Math.abs(perimeter(a) - perimeter(b)) > tol * 4) return false;
  const n = Math.min(a.length, b.length, 8);
  let err = 0;
  for (let i = 0; i < n; i++) {
    const ai = a[Math.floor((i * (a.length - 1)) / Math.max(1, n - 1))];
    const bi = b[Math.floor((i * (b.length - 1)) / Math.max(1, n - 1))];
    err += Math.hypot(ai.x - bi.x, ai.y - bi.y);
  }
  return err / n < tol;
}

function mergeUnique(base: Pt[][], extra: Pt[][], tol: number): Pt[][] {
  const out = base.slice();
  for (const s of extra) {
    if (out.some((b) => nearDuplicate(b, s, tol))) continue;
    out.push(s);
  }
  return out;
}

const REFINE_VARIANTS: PipelineOptions[] = [
  { label: "refine-soft", maxWork: 1100, useGeometry: false, inkThrBias: 14, simplifyScale: 1.15, maxStrokes: 900 },
  { label: "refine-detail", maxWork: 1024, useGeometry: false, inkThrBias: 2, simplifyScale: 1.05, maxStrokes: 850 },
  { label: "refine-clean", maxWork: 960, useGeometry: false, inkThrBias: 8, simplifyScale: 1.55, maxStrokes: 700 },
  { label: "refine-geo", maxWork: 1024, useGeometry: true, inkThrBias: 6, simplifyScale: 1.35, maxStrokes: 650 },
];

export type RefineLoopResult = {
  ok: boolean;
  best?: RefineCandidate;
  rounds: number;
  elapsedMs: number;
  reason: string;
};

/**
 * Iteratively improve a seed candidate: keep correct strokes, regenerate for
 * misses, re-score, stop on confidence / plateau / time / rounds.
 */
export function refineMatchLoop(
  work: RgbaImage,
  seed: RefineCandidate,
  opts: {
    vision?: VisionHints;
    budget?: RefineBudget;
    focused?: boolean;
    /** Reject candidates below this edge precision (portrait scribble guard). */
    minPrecision?: number;
    policy?: import("./subjectClass").DrawPolicy;
    onProgress?: (p: RefineProgress) => void;
    signal?: AbortSignal;
    now?: () => number;
  } = {},
): RefineLoopResult {
  const mode = seed.result.mode;
  const budget = opts.budget ?? defaultRefineBudget(mode);
  const now = opts.now ?? (() => Date.now());
  const t0 = now();
  let best = seed;
  let rounds = 0;
  let stagnant = 0;

  const variants = REFINE_VARIANTS.map((v) => ({
    ...v,
    forceMode: mode as "line-art" | "photo",
    policy: opts.policy,
    maxStrokes: opts.policy?.maxStrokes
      ? Math.min(v.maxStrokes ?? 400, opts.policy.maxStrokes + 20)
      : v.maxStrokes,
    simplifyScale: Math.max(
      v.simplifyScale ?? 1,
      opts.policy?.simplifyScale ?? 1,
    ),
  }));

  while (rounds < budget.maxRounds) {
    if (opts.signal?.aborted) break;
    const elapsed = now() - t0;
    if (elapsed >= budget.maxMs) break;
    if (best.confidence >= budget.minConfidence && best.validation.passed) break;

    const miss = buildMissMask(
      work,
      best.result.workStrokes,
      mode,
    );
    const missCount = miss.reduce((a, b) => a + b, 0);
    if (missCount < 40 && best.validation.edgePrecision >= 0.55) {
      // Nothing meaningful left to recover
      break;
    }

    const kept = keepCorrectStrokes(best.result.workStrokes, work, mode);
    const variant = variants[rounds % variants.length];
    const label = `${seed.label}+${variant.label}-r${rounds + 1}`;
    opts.onProgress?.({
      phase: label,
      attempt: rounds + 1,
      confidence: best.confidence,
      round: rounds + 1,
    });

    const fresh = rgbaToStrokeGuide(work, {
      ...variant,
      vision: opts.vision,
      label,
    });
    rounds++;
    if (!fresh.ok) {
      stagnant++;
      if (stagnant >= 2) break;
      continue;
    }

    // Prefer new strokes that cover misses or are themselves high-precision
    const { skeleton } = buildEdgeSkeleton(work, mode);
    const ref = dilate(
      skeleton,
      work.width,
      work.height,
      mode === "line-art" ? 3 : 2,
    );
    const useful = fresh.workStrokes.filter((pts) => {
      if (strokeCoversMiss(pts, miss, work.width, work.height)) return true;
      return strokePrecisionOnRef(pts, ref, work.width, work.height) >= 0.5;
    });

    const merged = leastPathOrder(
      mergeUnique(kept, useful, Math.max(3, Math.hypot(work.width, work.height) / 220)),
    );
    const tonal = best.result.tonalWorkStrokes;
    const assembled = assembleGuideFromWork(
      work,
      mode,
      merged.slice(0, variant.maxStrokes ?? 400),
      tonal,
      label,
    );
    const validation = validateWorkStrokes(
      work,
      assembled.workStrokes,
      mode,
      {
        focusedSubject: !!opts.focused,
        tonalStrokes: assembled.tonalWorkStrokes,
      },
    );
    const confidence = matchConfidence(validation);
    const improved = confidence > best.confidence + budget.plateauEps;
    if (improved || (confidence >= best.confidence && validation.dice >= best.validation.dice)) {
      best = { result: assembled, validation, confidence, label };
      stagnant = 0;
    } else {
      stagnant++;
      if (stagnant >= 2) break;
    }
  }

  const elapsedMs = now() - t0;
  const minPrec = opts.minPrecision ?? 0.32;
  const prec = best.validation.edgePrecision ?? 0;
  const recall = best.validation.edgeRecall ?? 0;
  const strokes = best.result.workStrokes.length;
  const isPortrait =
    opts.policy?.subjectClass === "portrait" ||
    !!opts.policy?.tags?.includes("portrait");
  const isStructure =
    opts.policy?.subjectClass === "structure" ||
    !!opts.policy?.tags?.includes("structure") ||
    !!opts.policy?.keepBackgroundStructure;
  // Macro-form Match Gate (not raw fur edges): silhouette lessons ≠ fur recall
  const minRecall = isPortrait ? 0.12 : isStructure ? 0.22 : 0.26;
  // Animal ink drawings score mid confidence vs furry photo edges (~0.40);
  // still accept when precision/form are solid (block only true scribble).
  const minConfAccept = isPortrait ? 0.28 : Math.max(budget.minConfidence, 0.4);
  const precFloor = Math.max(minPrec, isPortrait ? 0.36 : isStructure ? 0.48 : 0.42);
  const quad = best.validation.quadrantCoverage ?? 0;
  // Structure lessons must not be 100 window-sparkle loops
  let meanStrokeLen = 0;
  if (isStructure && strokes > 0) {
    for (const s of best.result.workStrokes) {
      for (let i = 1; i < s.length; i++) {
        meanStrokeLen += Math.hypot(s[i].x - s[i - 1].x, s[i].y - s[i - 1].y);
      }
    }
    meanStrokeLen /= strokes;
  }
  const diag = Math.hypot(best.result.workSize.w, best.result.workSize.h);
  const structureOk =
    !isStructure ||
    (isPortrait
      ? meanStrokeLen >= diag * 0.018 && strokes <= 190
      : meanStrokeLen >= diag * 0.035 && strokes <= 140);
  // Portrait silhouette lessons often dilate wider than sparse form GT →
  // precision looks "low" even when validation.passed and recall is healthy.
  const portraitFormPass =
    isPortrait &&
    recall >= 0.35 &&
    quad >= 0.5 &&
    prec >= 0.22 &&
    (best.validation.dice ?? 0) >= 0.16 &&
    // Soft fur often fails edgePrecision floor while form is still teachable
    (best.validation.passed || recall >= 0.5);
  // Night architecture: accept form-valid outlines even when sparkle GT fights us
  const structureFormPass =
    isStructure &&
    best.validation.passed &&
    recall >= 0.18 &&
    prec >= 0.35 &&
    quad >= 0.4 &&
    structureOk;
  const preciseEnough =
    prec >= minPrec || portraitFormPass || structureFormPass;
  // Form-first accept: solid precision/recall beats a low composite conf
  // (macro GT + silhouette lessons score ~0.30–0.36 vs fur photos).
  const formOk =
    (prec >= precFloor || portraitFormPass || structureFormPass) &&
    recall >= minRecall &&
    quad >= (isPortrait ? 0.3 : 0.4);
  const strokeCap = Math.min(
    240,
    (opts.policy?.maxStrokes ?? 220) + (portraitFormPass ? 100 : 20),
  );
  const teachableOutline =
    strokes >= (isPortrait ? 8 : 10) &&
    strokes <= strokeCap &&
    formOk &&
    structureOk &&
    (best.confidence >= minConfAccept ||
      portraitFormPass ||
      (isPortrait && prec >= 0.42));
  const strong =
    preciseEnough &&
    best.confidence >= (isPortrait ? minConfAccept : budget.minConfidence) &&
    (best.validation.passed ||
      (recall >= minRecall &&
        prec >= minPrec &&
        best.validation.quadrantCoverage >= 0.45));
  const solidSeed =
    seed.validation.passed &&
    seed.confidence >= minConfAccept &&
    seed.validation.quadrantCoverage >= 0.4 &&
    ((seed.validation.edgePrecision ?? 0) >= precFloor ||
      (isPortrait && (seed.validation.edgePrecision ?? 0) >= 0.14)) &&
    (seed.validation.edgeRecall ?? 0) >= minRecall;
  // Portrait/structure form GT + healthy scores: accept teachable outlines
  // even when Chamfer precision is diluted (fur width / night GT mismatch).
  const accept =
    teachableOutline ||
    ((strong || (solidSeed && best.confidence >= seed.confidence - 0.01)) &&
      preciseEnough) ||
    portraitFormPass ||
    structureFormPass;

  if (!accept) {
    return {
      ok: false,
      best,
      rounds,
      elapsedMs,
      reason: !preciseEnough
        ? "Almost a lesson — the lines chased texture more than form. Bring a clearer, closer subject photo and we’ll invent a cleaner training path."
        : "Almost a lesson — the coach needs a brighter crop with one clear focus before the strokes feel true enough to teach. Try again and we’ll invent the path.",
    };
  }

  return { ok: true, best, rounds, elapsedMs, reason: "" };
}
