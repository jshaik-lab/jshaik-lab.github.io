/**
 * Photo → clean ink drawing (intermediate) before stroke extraction.
 *
 * Self-tuning stylize (any natural photo):
 *   scale-normalized kernels → exposure stretch → HFER texture diagnosis →
 *   rolling-guidance macro smooth → multi-scale XDoG with quantile ε
 *
 * Mixed scenes keep silhouette subjects AND background structure when policy says so.
 */

import {
  toGray,
  boxBlur,
  sobelMagnitude,
  nonMaxSuppress,
  hysteresisThreshold,
  type RgbaImage,
} from "./pipeline";
import {
  estimateForeground,
  estimatePhotoForeground,
  fillInteriorHoles,
  smoothBinaryMask,
  foregroundBoundary,
  grabCutLite,
} from "./foreground";
import type { DrawPolicy } from "./subjectClass";
import {
  generalizedPhotoToInk,
  macroToGray255,
  type AdaptiveInkStats,
} from "./adaptiveStylize";

export type PhotoInkResult = {
  inkGray: Float32Array;
  w: number;
  h: number;
  inkRgba: RgbaImage;
  /** Texture-free macro gray (0..255) for Match Gate GT. */
  macroGray?: Float32Array;
  adaptive?: AdaptiveInkStats;
};

export function stylizePhotoToInk(
  work: RgbaImage,
  policy?: DrawPolicy,
): PhotoInkResult {
  const w = work.width;
  const h = work.height;
  const gray = toGray(work);
  const tags = policy?.tags ?? (policy ? [policy.subjectClass] : []);
  const hasPortrait = tags.includes("portrait");
  const hasStructure = tags.includes("structure");
  const hasNature = tags.includes("nature");
  const mixed =
    !!policy?.mixed || tags.filter((t) => t !== "general").length > 1;

  const portraitOnly =
    (hasPortrait || !!policy?.silhouetteFirst) && !hasStructure && !hasNature;

  // Soft FG for silhouette subjects (mask guides quantile ε + HFER).
  // Close-up animals on flat gray often collapse grayscale grabCut — use color + warm FG.
  let fgRough =
    policy?.silhouetteFirst || hasPortrait || hasNature
      ? estimatePhotoForeground(work, gray)
      : estimateForeground(work, "photo");
  if (portraitOnly || hasPortrait) {
    const warm = warmSubjectMask(work, w, h);
    let warmN = 0;
    let grabN = 0;
    for (let i = 0; i < warm.length; i++) {
      if (warm[i]) warmN++;
      if (fgRough[i]) grabN++;
    }
    const warmFrac = warmN / (w * h);
    // Whole-frame warm tint (synth / studio wash) is useless as FG — keep photo FG.
    if (warmFrac > 0.08 && warmFrac < 0.96) {
      if (warmN > 40) {
        for (let i = 0; i < fgRough.length; i++) {
          if (warm[i]) fgRough[i] = 1;
        }
      }
      fgRough = smoothBinaryMask(
        fillInteriorHoles(fgRough, w, h),
        w,
        h,
        Math.max(2, Math.round(Math.min(w, h) * 0.005)),
      );
    }
  }

  // Night / dark architecture: build outlines from bright masses, not window soup
  let meanL = 0;
  for (let i = 0; i < gray.length; i++) meanL += gray[i];
  meanL /= gray.length || 1;
  const nightStructure =
    (hasStructure || !!policy?.keepBackgroundStructure) && meanL < 100;

  // Target ink density: portraits need facial structure (eyes/nose/muzzle/ears),
  // not fur grain — denser than the old 0.065 silhouette-only target.
  const targetInkRatio = portraitOnly
    ? 0.11
    : nightStructure
      ? 0.07
      : hasStructure
        ? 0.11
        : hasPortrait
          ? 0.1
          : 0.09;
  const textureBias = portraitOnly || hasNature ? 0.08 : 0;

  // Structure / night skyline: quantile over the whole frame (FG often misses
  // buildings). Portraits/nature: mask to the subject so sky isn't inked.
  const useFgMask =
    (hasPortrait || hasNature || portraitOnly) &&
    !hasStructure &&
    !policy?.keepBackgroundStructure;
  const adaptive = generalizedPhotoToInk(gray, w, h, {
    targetInkRatio,
    mask: useFgMask ? fgRough : undefined,
    textureBias,
    nightStructure,
  });

  // Re-estimate FG combining macro-smoothed luminance + color-aware FG
  const macro255 = macroToGray255(adaptive.macroSmoothed);
  let fg =
    policy?.silhouetteFirst || hasPortrait || hasNature
      ? estimatePhotoForeground(work, macro255)
      : fgRough;
  // Union with initial photo FG so neither half of a close-up face drops out
  if (portraitOnly || hasPortrait) {
    for (let i = 0; i < fg.length; i++) {
      if (fgRough[i]) fg[i] = 1;
    }
    const warm = warmSubjectMask(work, w, h);
    let warmN = 0;
    for (let i = 0; i < warm.length; i++) {
      if (warm[i]) warmN++;
    }
    const warmFrac = warmN / (w * h);
    if (warmFrac > 0.08 && warmFrac < 0.96) {
      for (let i = 0; i < fg.length; i++) if (warm[i]) fg[i] = 1;
      fg = expandWarmForeground(work, fg, w, h);
    }
    fg = smoothBinaryMask(
      fillInteriorHoles(fg, w, h),
      w,
      h,
      Math.max(3, Math.round(Math.min(w, h) * 0.006)),
    );
  }

  let sil = foregroundBoundary(fg, w, h);
  // Close-ups that touch the frame lose ear/cheek outline if we only delete
  // border sil pixels. Inset the FG slightly and redraw the boundary so the
  // head still gets a continuous graphite silhouette inside the paper.
  const edgeMargin = Math.max(3, Math.round(Math.min(w, h) * 0.012));
  let borderFg = 0;
  let borderN = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const onBorder =
        x < edgeMargin ||
        x >= w - edgeMargin ||
        y < edgeMargin ||
        y >= h - edgeMargin;
      if (!onBorder) continue;
      borderN++;
      if (fg[y * w + x]) borderFg++;
    }
  }
  const touchesFrame = borderN > 0 && borderFg / borderN > 0.2;
  if (touchesFrame && (portraitOnly || hasPortrait || policy?.silhouetteFirst)) {
    const inset = new Uint8Array(fg.length);
    for (let y = edgeMargin; y < h - edgeMargin; y++) {
      for (let x = edgeMargin; x < w - edgeMargin; x++) {
        const i = y * w + x;
        if (!fg[i]) continue;
        let keep = true;
        for (let dy = -edgeMargin; dy <= edgeMargin && keep; dy++) {
          for (let dx = -edgeMargin; dx <= edgeMargin; dx++) {
            if (!fg[(y + dy) * w + (x + dx)]) {
              keep = false;
              break;
            }
          }
        }
        if (keep) inset[i] = 1;
      }
    }
    const insetSil = foregroundBoundary(inset, w, h);
    let insetN = 0;
    for (let i = 0; i < insetSil.length; i++) if (insetSil[i]) insetN++;
    if (insetN > 40) sil = insetSil;
  } else {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        if (
          x < 3 ||
          x >= w - 3 ||
          y < 3 ||
          y >= h - 3
        ) {
          sil[y * w + x] = 0;
        }
      }
    }
  }

  const features =
    hasPortrait || policy?.silhouetteFirst
      ? extractPortraitFeatureLines(work, gray, macro255, fg, sil, w, h)
      : null;

  const inkGray = new Float32Array(w * h);
  inkGray.fill(248);
  const allowBg =
    !portraitOnly && (!!policy?.keepBackgroundStructure || hasStructure);

  for (let i = 0; i < inkGray.length; i++) {
    const onFg = fg[i] !== 0;
    if (!onFg && !allowBg) continue;

    let ink = false;
    if (sil[i]) ink = true;
    if (features && features[i]) ink = true;

    // Quantile XDoG / macro-gradient ink
    if (adaptive.inkMask[i] && (onFg || allowBg)) {
      if (onFg && (portraitOnly || hasPortrait || policy?.silhouetteFirst)) {
        // Prefer silhouette + anatomical features so fur grain does not flood.
        // Also keep the darkest adaptive ridges (jaw, ear pinna, muzzle bridge)
        // so close-ups remain recognizable without the photo underlay.
        if (sil[i] || (features && features[i])) ink = true;
        else if (adaptive.inkLuma[i] <= 40) ink = true;
      } else {
        ink = true;
      }
    }

    if (ink) {
      // Soft XDoG luma where available; silhouette/features stay darkest
      if (sil[i] || (features && features[i])) {
        inkGray[i] = 28;
      } else {
        inkGray[i] = Math.min(55, adaptive.inkLuma[i]);
      }
    } else if (onFg && !portraitOnly) {
      // Soft wash from macro (no posterize bands)
      inkGray[i] = 210 + adaptive.macroSmoothed[i] * 30;
    } else if (onFg && portraitOnly) {
      inkGray[i] = 232;
    }
  }

  dilateDarkInk(
    inkGray,
    w,
    h,
    portraitOnly ? 1 : hasPortrait ? 1 : nightStructure ? 1 : 1,
    40,
  );

  return {
    inkGray,
    w,
    h,
    inkRgba: inkGrayToRgba(inkGray, w, h),
    macroGray: macro255,
    adaptive: adaptive.stats,
  };
}

export function morphCloseBright(
  img: Float32Array,
  w: number,
  h: number,
  radius: number,
): void {
  if (radius <= 0) return;
  const dilate = new Float32Array(img.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let m = -Infinity;
      for (let dy = -radius; dy <= radius; dy++) {
        const yy = Math.min(h - 1, Math.max(0, y + dy));
        for (let dx = -radius; dx <= radius; dx++) {
          const xx = Math.min(w - 1, Math.max(0, x + dx));
          m = Math.max(m, img[yy * w + xx]);
        }
      }
      dilate[y * w + x] = m;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let m = Infinity;
      for (let dy = -radius; dy <= radius; dy++) {
        const yy = Math.min(h - 1, Math.max(0, y + dy));
        for (let dx = -radius; dx <= radius; dx++) {
          const xx = Math.min(w - 1, Math.max(0, x + dx));
          m = Math.min(m, dilate[yy * w + xx]);
        }
      }
      img[y * w + x] = m;
    }
  }
}

/**
 * Extract 1px anatomical & structural lines inside the foreground mask for
 * single or multiple animals/portraits:
 *  1. Illumination-invariant local contrast basins (eyes, irises, pupils, nose tips)
 *     emitted as 1px outer contours + 1px inner pupil/nostril core contours.
 *  2. Macro-smoothed luminance + opponent-color gradient Non-Maximum Suppression (NMS)
 *     for crisp eyelids, nose triangle, ear folds, chin/jaw, and multi-subject overlap edges.
 *  3. 1D dark crease centerlines for mouth/muzzle clefts and eye corner creases.
 */
export function extractPortraitFeatureLines(
  work: RgbaImage,
  gray: Float32Array,
  macro255: Float32Array,
  fg: Uint8Array,
  sil: Uint8Array,
  w: number,
  h: number,
): Uint8Array {
  const out = new Uint8Array(w * h);
  let fgMean = 0;
  let fgN = 0;
  for (let i = 0; i < gray.length; i++) {
    if (!fg[i]) continue;
    fgMean += gray[i];
    fgN++;
  }
  if (fgN < 60) return out;
  fgMean /= fgN;

  const minSide = Math.min(w, h);
  const smoothGray = boxBlur(
    gray,
    w,
    h,
    Math.max(1, Math.round(minSide * 0.003)),
  );
  const localMed = boxBlur(
    smoothGray,
    w,
    h,
    Math.max(10, Math.round(minSide * 0.045)),
  );

  // Build an interior FG mask eroded slightly from the outer silhouette & image frame
  const interior = new Uint8Array(w * h);
  const pad = Math.max(4, Math.round(minSide * 0.008));
  for (let y = pad; y < h - pad; y++) {
    for (let x = pad; x < w - pad; x++) {
      const i = y * w + x;
      if (!fg[i]) continue;
      let nearBg = false;
      for (let dy = -3; dy <= 3 && !nearBg; dy++) {
        for (let dx = -3; dx <= 3; dx++) {
          if (!fg[(y + dy) * w + (x + dx)] || sil[(y + dy) * w + (x + dx)]) {
            nearBg = true;
            break;
          }
        }
      }
      if (!nearBg) interior[i] = 1;
    }
  }

  const rg = new Float32Array(w * h);
  const yb = new Float32Array(w * h);
  const green = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const o = i * 4;
    const r = work.data[o];
    const g = work.data[o + 1];
    const b = work.data[o + 2];
    rg[i] = r - g;
    yb[i] = (r + g) * 0.5 - b;
    green[i] = g;
  }
  const greenBlur = boxBlur(
    green,
    w,
    h,
    Math.max(2, Math.round(minSide * 0.004)),
  );
  // Coarse muzzle-scale background (so a ~90px nose pad contrasts against the surrounding muzzle)
  const rMuzzleBg = Math.max(22, Math.round(minSide * 0.095));
  const localGreenMed = boxBlur(greenBlur, w, h, rMuzzleBg);
  const rChroma = Math.max(3, Math.round(minSide * 0.007));
  const rgBlur = boxBlur(rg, w, h, rChroma);
  const ybBlur = boxBlur(yb, w, h, rChroma);
  const localRgMed = boxBlur(rgBlur, w, h, rMuzzleBg);

  // --- 1A. Compact dark eye basins (irises + pupils across all subjects) ---
  const darkBasin = new Uint8Array(w * h);
  for (let i = 0; i < darkBasin.length; i++) {
    if (!interior[i]) continue;
    const dip = localMed[i] - smoothGray[i];
    if (dip >= 15 && smoothGray[i] <= Math.min(125, fgMean * 0.82)) {
      darkBasin[i] = 1;
    }
  }

  const basinRingZone = new Uint8Array(w * h);
  const muzzleZone = new Uint8Array(w * h);
  const seen = new Uint8Array(w * h);
  const minEyeArea = Math.max(24, Math.floor(w * h * 0.0003));
  const maxEyeArea = Math.floor(w * h * 0.015);
  const stack: number[] = [];
  type EyeCenter = { cx: number; cy: number; r: number };
  const detectedEyes: EyeCenter[] = [];

  for (let i = 0; i < darkBasin.length; i++) {
    if (!darkBasin[i] || seen[i]) continue;
    stack.length = 0;
    stack.push(i);
    seen[i] = 1;
    const comp: number[] = [];
    let x0 = w,
      y0 = h,
      x1 = 0,
      y1 = 0;
    let sumL = 0;
    let minL = 255;
    let maxDip = 0;
    while (stack.length) {
      const cur = stack.pop()!;
      comp.push(cur);
      const x = cur % w;
      const y = (cur / w) | 0;
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
      const lv = smoothGray[cur];
      const dip = localMed[cur] - lv;
      sumL += lv;
      if (lv < minL) minL = lv;
      if (dip > maxDip) maxDip = dip;
      for (const [dx, dy] of [
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
      ] as const) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const ni = ny * w + nx;
        if (seen[ni] || !darkBasin[ni]) continue;
        seen[ni] = 1;
        stack.push(ni);
      }
    }
    if (comp.length < minEyeArea || comp.length > maxEyeArea) continue;
    // True eyes/pupils have a deep dark pupil core and strong contrast dip;
    // forehead tabby fur stripes and ear shadows do not.
    if (minL > Math.min(74, fgMean * 0.54) || maxDip < 30) continue;
    const bw = x1 - x0 + 1;
    const bh = y1 - y0 + 1;
    const aspect = Math.max(bw, bh) / Math.max(1, Math.min(bw, bh));
    const fill = comp.length / Math.max(1, bw * bh);
    if (aspect > 2.35 || fill < 0.34) continue;

    detectedEyes.push({
      cx: (x0 + x1) * 0.5,
      cy: (y0 + y1) * 0.5,
      r: Math.max(bw, bh) * 0.5,
    });

    const compMask = new Uint8Array(w * h);
    for (const idx of comp) compMask[idx] = 1;
    const smoothedComp =
      comp.length >= 36 ? smoothBinaryMask(compMask, w, h, 2) : compMask;

    const halo = Math.max(6, Math.round(Math.max(bw, bh) * 0.24));
    const topHalo = Math.max(8, Math.round(Math.max(bw, bh) * 0.58));
    for (
      let yy = Math.max(0, y0 - topHalo);
      yy <= Math.min(h - 1, y1 + halo);
      yy++
    ) {
      for (let xx = Math.max(0, x0 - halo); xx <= Math.min(w - 1, x1 + halo); xx++) {
        basinRingZone[yy * w + xx] = 1;
      }
    }

    for (const idx of comp) {
      const x = idx % w;
      const y = (idx / w) | 0;
      if (x <= 0 || y <= 0 || x >= w - 1 || y >= h - 1) continue;
      if (!smoothedComp[idx]) continue;
      if (
        !smoothedComp[idx - 1] ||
        !smoothedComp[idx + 1] ||
        !smoothedComp[idx - w] ||
        !smoothedComp[idx + w]
      ) {
        out[idx] = 1;
      }
    }

    const meanCompL = sumL / comp.length;
    if (comp.length >= 45 && meanCompL - minL >= 10) {
      const coreThr = minL + (meanCompL - minL) * 0.46;
      const coreMask = new Uint8Array(w * h);
      let coreCount = 0;
      for (const idx of comp) {
        if (smoothGray[idx] <= coreThr) {
          coreMask[idx] = 1;
          coreCount++;
        }
      }
      if (coreCount >= 8 && coreCount <= comp.length * 0.72) {
        const smoothCore =
          coreCount >= 28 ? smoothBinaryMask(coreMask, w, h, 1) : coreMask;
        for (const idx of comp) {
          if (!smoothCore[idx]) continue;
          const x = idx % w;
          const y = (idx / w) | 0;
          if (x <= 0 || y <= 0 || x >= w - 1 || y >= h - 1) continue;
          if (
            !smoothCore[idx - 1] ||
            !smoothCore[idx + 1] ||
            !smoothCore[idx - w] ||
            !smoothCore[idx + w]
          ) {
            out[idx] = 1;
          }
        }
      }
    }
  }

  // Pair eyes greedily (1-to-1) across each animal/human face in the frame to locate its nose & muzzle ROI
  const candidatePairs: { i: number; j: number; dist: number; score: number }[] = [];
  for (let i = 0; i < detectedEyes.length; i++) {
    for (let j = i + 1; j < detectedEyes.length; j++) {
      const a = detectedEyes[i];
      const b = detectedEyes[j];
      const dx = Math.abs(b.cx - a.cx);
      const dy = Math.abs(b.cy - a.cy);
      const dist = Math.hypot(dx, dy);
      const maxR = Math.max(a.r, b.r);
      const minR = Math.min(a.r, b.r);
      if (
        dist >= maxR * 1.6 &&
        dist <= w * 0.72 &&
        dy <= dx * 0.48 &&
        minR >= maxR * 0.42
      ) {
        const score = dy / Math.max(1, dx) + (maxR - minR) / Math.max(1, maxR) + dist / w;
        candidatePairs.push({ i, j, dist, score });
      }
    }
  }
  candidatePairs.sort((a, b) => a.score - b.score);
  const usedEye = new Set<number>();
  const matchedPairs: { emx: number; emy: number; dist: number }[] = [];
  for (const p of candidatePairs) {
    if (usedEye.has(p.i) || usedEye.has(p.j)) continue;
    usedEye.add(p.i);
    usedEye.add(p.j);
    const a = detectedEyes[p.i];
    const b = detectedEyes[p.j];
    const emx = (a.cx + b.cx) * 0.5;
    const emy = (a.cy + b.cy) * 0.5;
    matchedPairs.push({ emx, emy, dist: p.dist });
    const mx0 = Math.max(0, Math.round(emx - p.dist * 0.48));
    const mx1 = Math.min(w - 1, Math.round(emx + p.dist * 0.48));
    const my0 = Math.max(0, Math.round(emy + p.dist * 0.22));
    const my1 = Math.min(h - 1, Math.round(emy + p.dist * 1.15));
    for (let yy = my0; yy <= my1; yy++) {
      for (let xx = mx0; xx <= mx1; xx++) {
        if (interior[yy * w + xx]) {
          muzzleZone[yy * w + xx] = 1;
          basinRingZone[yy * w + xx] = 1;
        }
      }
    }
    // Full-face structural ROI (ears → chin / cheeks) so jaw arcs and ear
    // folds use fine green+chroma NMS instead of fur-suppressing macro only.
    const fx0 = Math.max(0, Math.round(emx - p.dist * 1.15));
    const fx1 = Math.min(w - 1, Math.round(emx + p.dist * 1.15));
    const fy0 = Math.max(0, Math.round(emy - p.dist * 1.35));
    const fy1 = Math.min(h - 1, Math.round(emy + p.dist * 1.55));
    for (let yy = fy0; yy <= fy1; yy++) {
      for (let xx = fx0; xx <= fx1; xx++) {
        if (interior[yy * w + xx]) basinRingZone[yy * w + xx] = 1;
      }
    }
  }
  // Unmatched dark basins situated below a matched eye pair are dark noses (e.g. dog black nose pad)
  for (let idx = 0; idx < detectedEyes.length; idx++) {
    if (usedEye.has(idx)) continue;
    const c = detectedEyes[idx];
    for (const mp of matchedPairs) {
      if (
        Math.abs(c.cx - mp.emx) <= mp.dist * 0.55 &&
        c.cy >= mp.emy + mp.dist * 0.18 &&
        c.cy <= mp.emy + mp.dist * 1.25
      ) {
        const nx0 = Math.max(0, Math.round(c.cx - c.r * 2.5));
        const nx1 = Math.min(w - 1, Math.round(c.cx + c.r * 2.5));
        const ny0 = Math.max(0, Math.round(c.cy - c.r * 0.6));
        const ny1 = Math.min(h - 1, Math.round(c.cy + c.r * 3.2));
        for (let yy = ny0; yy <= ny1; yy++) {
          for (let xx = nx0; xx <= nx1; xx++) {
            if (interior[yy * w + xx]) {
              muzzleZone[yy * w + xx] = 1;
              basinRingZone[yy * w + xx] = 1;
            }
          }
        }
      }
    }
  }

  // --- 1B. Nose pad & nostril basin (pink/terracotta or dark nose triangle) ---
  const noseMask = new Uint8Array(w * h);
  const noseScoreArr = new Float32Array(w * h);
  for (let i = 0; i < noseMask.length; i++) {
    if (!interior[i]) continue;
    const gDip = Math.max(0, localGreenMed[i] - greenBlur[i]);
    const rgLift = Math.max(0, rgBlur[i] - localRgMed[i]);
    const score = gDip + 0.85 * rgLift;
    noseScoreArr[i] = score;
    const inMuzzle = muzzleZone[i] === 1;
    const scoreThr = inMuzzle ? 24 : 34;
    if (score >= scoreThr && greenBlur[i] <= fgMean * 0.92) {
      noseMask[i] = 1;
    }
  }

  const noseSeen = new Uint8Array(w * h);
  const minNoseArea = Math.max(24, Math.floor(w * h * 0.00025));
  const maxNoseArea = Math.floor(w * h * 0.014);
  for (let i = 0; i < noseMask.length; i++) {
    if (!noseMask[i] || noseSeen[i]) continue;
    stack.length = 0;
    stack.push(i);
    noseSeen[i] = 1;
    const comp: number[] = [];
    let x0 = w,
      y0 = h,
      x1 = 0,
      y1 = 0;
    let maxScore = 0;
    let minG = 255;
    let sumG = 0;
    let inMuzzleCount = 0;
    while (stack.length) {
      const cur = stack.pop()!;
      comp.push(cur);
      if (muzzleZone[cur]) inMuzzleCount++;
      const x = cur % w;
      const y = (cur / w) | 0;
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
      if (noseScoreArr[cur] > maxScore) maxScore = noseScoreArr[cur];
      const gv = greenBlur[cur];
      sumG += gv;
      if (gv < minG) minG = gv;
      for (const [dx, dy] of [
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
      ] as const) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const ni = ny * w + nx;
        if (noseSeen[ni] || !noseMask[ni]) continue;
        noseSeen[ni] = 1;
        stack.push(ni);
      }
    }
    if (comp.length < minNoseArea || comp.length > maxNoseArea) continue;
    const mostlyInMuzzle = inMuzzleCount >= comp.length * 0.5;
    if (maxScore < (mostlyInMuzzle ? 34 : 46)) continue;
    const bw = x1 - x0 + 1;
    const bh = y1 - y0 + 1;
    const aspect = Math.max(bw, bh) / Math.max(1, Math.min(bw, bh));
    const fill = comp.length / Math.max(1, bw * bh);
    if (aspect > (mostlyInMuzzle ? 3.8 : 2.4) || fill < 0.26) continue;

    const compMask = new Uint8Array(w * h);
    for (const idx of comp) compMask[idx] = 1;
    const smoothedNose = smoothBinaryMask(fillInteriorHoles(compMask, w, h), w, h, 2);
    for (const idx of comp) {
      const x = idx % w;
      const y = (idx / w) | 0;
      if (x <= 0 || y <= 0 || x >= w - 1 || y >= h - 1) continue;
      if (!smoothedNose[idx]) continue;
      if (
        !smoothedNose[idx - 1] ||
        !smoothedNose[idx + 1] ||
        !smoothedNose[idx - w] ||
        !smoothedNose[idx + w]
      ) {
        out[idx] = 1;
      }
    }

    // Also trace inner nostril cores inside the nose triangle
    const meanG = sumG / comp.length;
    if (comp.length >= 60 && meanG - minG >= 12) {
      const nostrilThr = minG + (meanG - minG) * 0.42;
      const nostrilMask = new Uint8Array(w * h);
      let nostrilCount = 0;
      for (const idx of comp) {
        if (greenBlur[idx] <= nostrilThr) {
          nostrilMask[idx] = 1;
          nostrilCount++;
        }
      }
      if (nostrilCount >= 8 && nostrilCount <= comp.length * 0.55) {
        for (const idx of comp) {
          if (!nostrilMask[idx]) continue;
          const x = idx % w;
          const y = (idx / w) | 0;
          if (x <= 0 || y <= 0 || x >= w - 1 || y >= h - 1) continue;
          if (
            !nostrilMask[idx - 1] ||
            !nostrilMask[idx + 1] ||
            !nostrilMask[idx - w] ||
            !nostrilMask[idx + w]
          ) {
            out[idx] = 1;
          }
        }
      }
    }

    // Activate muzzle/mouth cleft zone around & below the nose
    const mx0 = Math.max(0, Math.round(x0 - bw * 0.95));
    const mx1 = Math.min(w - 1, Math.round(x1 + bw * 0.95));
    const my0 = Math.max(0, Math.round(y0 - bh * 0.25));
    const my1 = Math.min(h - 1, Math.round(y1 + bh * 1.75));
    for (let yy = my0; yy <= my1; yy++) {
      for (let xx = mx0; xx <= mx1; xx++) {
        if (interior[yy * w + xx]) {
          muzzleZone[yy * w + xx] = 1;
          basinRingZone[yy * w + xx] = 1;
        }
      }
    }
  }

  // --- 2. Macro-smoothed luminance + opponent-color gradient NMS ---
  const macroBlur = boxBlur(
    macro255,
    w,
    h,
    Math.max(5, Math.round(minSide * 0.014)),
  );
  const { mag: macroMag } = sobelMagnitude(macroBlur, w, h);
  const {
    mag: greenMag,
    gx: greenGx,
    gy: greenGy,
  } = sobelMagnitude(
    boxBlur(greenBlur, w, h, Math.max(2, Math.round(minSide * 0.005))),
    w,
    h,
  );
  const { mag: rgMag } = sobelMagnitude(rgBlur, w, h);
  const { mag: ybMag } = sobelMagnitude(ybBlur, w, h);

  const structMag = new Float32Array(w * h);
  const magSamples: number[] = [];
  const sampleStep = Math.max(1, Math.floor((w * h) / 5000));
  for (let i = 0; i < structMag.length; i++) {
    if (!interior[i]) continue;
    const cm = Math.hypot(rgMag[i], ybMag[i]);
    // Inside eye/muzzle zones use finer green+chroma gradient so nose bridge, mouth curves,
    // and chin arc are sharp; outside use coarse macro gradient to ignore fur/ear hairs.
    const v = basinRingZone[i]
      ? greenMag[i] * 0.85 + 0.65 * cm
      : macroMag[i] + 0.35 * cm;
    structMag[i] = v;
    if (i % sampleStep === 0 && v > 1) magSamples.push(v);
  }

  if (magSamples.length >= 20) {
    magSamples.sort((a, b) => a - b);
    const p90 = magSamples[Math.floor(magSamples.length * 0.9)] ?? 28;
    // Slightly lower than p90 so eyelid / muzzle / ear-fold ridges survive
    // without opening the floodgates to fur grain.
    const highThr = Math.max(18, p90 * 0.82);
    const lowThr = highThr * 0.48;
    const nms = nonMaxSuppress(structMag, greenGx, greenGy, w, h);
    let maxNms = 1;
    for (let i = 0; i < nms.length; i++) if (nms[i] > maxNms) maxNms = nms[i];
    const edges = hysteresisThreshold(
      nms,
      w,
      h,
      Math.min(0.9, lowThr / maxNms),
      Math.min(0.95, highThr / maxNms),
    );

    // --- 3. 1D dark crease centerlines (mouth cleft, nostril slits, chin curve, eye corners) ---
    const rCrease = Math.max(5, Math.round(minSide * 0.016));
    const creaseBase = boxBlur(greenBlur, w, h, rCrease);
    const crease = new Float32Array(w * h);
    for (let i = 0; i < crease.length; i++) {
      if (!interior[i]) continue;
      const gDip = Math.max(0, creaseBase[i] - greenBlur[i]);
      const rgBoost = muzzleZone[i]
        ? Math.max(0, rgBlur[i] - localRgMed[i]) * 0.35
        : 0;
      crease[i] = gDip + rgBoost;
    }

    const rawLines = new Uint8Array(w * h);
    for (let y = 2; y < h - 2; y++) {
      for (let x = 2; x < w - 2; x++) {
        const i = y * w + x;
        if (!interior[i]) continue;
        if (edges[i]) {
          rawLines[i] = 1;
          continue;
        }
        if (!basinRingZone[i]) continue;
        const c = crease[i];
        if (c < (muzzleZone[i] ? 7.5 : 9.5)) continue;
        const peakX =
          c >= crease[i - 1] &&
          c > crease[i + 1] &&
          c - Math.min(crease[i - 2], crease[i + 2]) >= 2.0;
        const peakY =
          c >= crease[i - w] &&
          c > crease[i + w] &&
          c - Math.min(crease[i - 2 * w], crease[i + 2 * w]) >= 2.0;
        if (peakX || peakY) rawLines[i] = 1;
      }
    }

    // Keep coherent structural lines: fine details near eyes/nose/muzzle, and only major
    // smooth macro-contours (multi-cat overlap boundaries, full ear pinna arcs) elsewhere.
    const lineSeen = new Uint8Array(w * h);
    const minMacroSpan = minSide * 0.12;
    const minNearFeatureLen = Math.max(6, Math.round(minSide * 0.01));
    for (let i = 0; i < rawLines.length; i++) {
      if (!rawLines[i] || lineSeen[i]) continue;
      stack.length = 0;
      stack.push(i);
      lineSeen[i] = 1;
      const comp: number[] = [];
      let nearBasin = false;
      let x0 = w,
        y0 = h,
        x1 = 0,
        y1 = 0;
      while (stack.length) {
        const cur = stack.pop()!;
        comp.push(cur);
        if (basinRingZone[cur]) nearBasin = true;
        const x = cur % w;
        const y = (cur / w) | 0;
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (dx === 0 && dy === 0) continue;
            const nx = x + dx;
            const ny = y + dy;
            if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
            const ni = ny * w + nx;
            if (lineSeen[ni] || !rawLines[ni]) continue;
            lineSeen[ni] = 1;
            stack.push(ni);
          }
        }
      }
      const span = Math.hypot(x1 - x0, y1 - y0);
      const straightness = span / Math.max(1, comp.length);
      if (
        (nearBasin &&
          comp.length >= minNearFeatureLen &&
          span >= minSide * 0.012) ||
        (span >= minMacroSpan && straightness >= 0.5)
      ) {
        for (const idx of comp) out[idx] = 1;
      }
    }
  }

  return out;
}

/** Warm animal/skin vs cool/gray backdrop (close-up pet faces & multi-pet photos). */
function warmSubjectMask(img: RgbaImage, w: number, h: number): Uint8Array {
  const out = new Uint8Array(w * h);
  // Sample lowest-variance border patches for backdrop color so subjects touching
  // top/bottom borders (like a close-up cat) don't contaminate the backdrop mean.
  const border = Math.max(3, Math.round(Math.min(w, h) * 0.04));
  const patchStep = Math.max(10, Math.round(Math.min(w, h) / 14));
  type BorderPatch = { r: number; g: number; b: number; varRGB: number; n: number };
  const patches: BorderPatch[] = [];
  const samplePatch = (x0: number, y0: number, x1: number, y1: number) => {
    let sr = 0,
      sg = 0,
      sb = 0,
      n = 0;
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        const o = (y * w + x) * 4;
        sr += img.data[o];
        sg += img.data[o + 1];
        sb += img.data[o + 2];
        n++;
      }
    }
    if (n < 6) return;
    const mr = sr / n;
    const mg = sg / n;
    const mb = sb / n;
    let v = 0;
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        const o = (y * w + x) * 4;
        const dr = img.data[o] - mr;
        const dg = img.data[o + 1] - mg;
        const db = img.data[o + 2] - mb;
        v += dr * dr + dg * dg + db * db;
      }
    }
    patches.push({ r: mr, g: mg, b: mb, varRGB: v / n, n });
  };
  for (let x = 0; x < w; x += patchStep) {
    samplePatch(x, 0, Math.min(w, x + patchStep), border);
    samplePatch(x, Math.max(0, h - border), Math.min(w, x + patchStep), h);
  }
  for (let y = border; y < h - border; y += patchStep) {
    samplePatch(0, y, border, Math.min(h - border, y + patchStep));
    samplePatch(Math.max(0, w - border), y, w, Math.min(h - border, y + patchStep));
  }
  const bgPenalty = (p: BorderPatch) =>
    p.varRGB + (p.r > p.g + 8 && p.r > p.b + 16 ? 260 : 0);
  patches.sort((a, b) => bgPenalty(a) - bgPenalty(b));
  let br = 0,
    bg = 0,
    bb = 0,
    bn = 0;
  if (patches.length) {
    const seed = patches[0];
    const maxVar = Math.max(30, seed.varRGB * 3.5);
    for (const p of patches) {
      const dist = Math.hypot(p.r - seed.r, p.g - seed.g, p.b - seed.b);
      if (dist <= 28 && p.varRGB <= maxVar) {
        br += p.r * p.n;
        bg += p.g * p.n;
        bb += p.b * p.n;
        bn += p.n;
      }
    }
  }
  br /= bn || 1;
  bg /= bn || 1;
  bb /= bn || 1;

  for (let i = 0; i < out.length; i++) {
    const o = i * 4;
    const r = img.data[o];
    const g = img.data[o + 1];
    const b = img.data[o + 2];
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const chroma = max - min;
    const dBorder = Math.hypot(r - br, g - bg, b - bb);
    const warm = r > g + 5 && r > b + 9 && r > 50 && r < 252;
    const grayish = chroma < 16 && max > 70 && max < 210;
    // Must differ from backdrop — avoids marking a uniformly warm synth frame
    if (warm && !grayish && dBorder > 24) out[i] = 1;
    // Cream muzzle / light fur still warm-ish
    if (
      !out[i] &&
      r > 135 &&
      g > 105 &&
      b > 85 &&
      r >= g - 2 &&
      r > b + 4 &&
      chroma > 10 &&
      dBorder > 20
    ) {
      out[i] = 1;
    }
  }
  // Fill small holes and interior eye/nose cavities
  const filled = out.slice();
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      if (out[i]) continue;
      let n = 0;
      if (out[i - 1]) n++;
      if (out[i + 1]) n++;
      if (out[i - w]) n++;
      if (out[i + w]) n++;
      if (n >= 3) filled[i] = 1;
    }
  }
  const multi = keepSubjectMaskComponents(filled, w, h);
  return fillInteriorHoles(multi, w, h);
}

/**
 * Keep all prominent connected subject components (supports multiple cats/subjects
 * in the same photo while dropping tiny background specks).
 */
function keepSubjectMaskComponents(
  mask: Uint8Array,
  w: number,
  h: number,
): Uint8Array {
  const seen = new Uint8Array(w * h);
  const comps: number[][] = [];
  let maxLen = 0;
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i] || seen[i]) continue;
    const stack = [i];
    const comp: number[] = [];
    seen[i] = 1;
    while (stack.length) {
      const cur = stack.pop()!;
      comp.push(cur);
      const x = cur % w;
      const y = (cur / w) | 0;
      for (const [dx, dy] of [
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
      ] as const) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const ni = ny * w + nx;
        if (seen[ni] || !mask[ni]) continue;
        seen[ni] = 1;
        stack.push(ni);
      }
    }
    comps.push(comp);
    if (comp.length > maxLen) maxLen = comp.length;
  }
  const out = new Uint8Array(w * h);
  if (!maxLen) return out;
  const minArea = Math.max(80, Math.floor(w * h * 0.012));
  for (const comp of comps) {
    if (comp.length === maxLen || (comp.length >= minArea && comp.length >= maxLen * 0.14)) {
      for (const c of comp) out[c] = 1;
    }
  }
  return out;
}

/** Grow under-segmented animal/face FG using warm chroma + macro continuity. */
function expandWarmForeground(
  img: RgbaImage,
  fg: Uint8Array,
  w: number,
  h: number,
): Uint8Array {
  const out = fg.slice();
  let fgN = 0;
  for (let i = 0; i < out.length; i++) if (out[i]) fgN++;
  if (fgN < 40 || fgN > w * h * 0.92) return out;

  for (let pass = 0; pass < 3; pass++) {
    const src = out.slice();
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const i = y * w + x;
        if (src[i]) continue;
        let near = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (src[(y + dy) * w + (x + dx)]) near++;
          }
        }
        if (near < 2) continue;
        const o = i * 4;
        const r = img.data[o];
        const g = img.data[o + 1];
        const b = img.data[o + 2];
        if (r > g + 6 && r > b + 10 && r > 55 && r < 250) out[i] = 1;
      }
    }
  }
  return out;
}

function dilateDarkInk(
  ink: Float32Array,
  w: number,
  h: number,
  radius: number,
  darkThr: number,
): void {
  if (radius <= 0) return;
  const src = ink.slice();
  for (let y = radius; y < h - radius; y++) {
    for (let x = radius; x < w - radius; x++) {
      const i = y * w + x;
      if (src[i] <= darkThr) continue;
      let dark = false;
      for (let dy = -radius; dy <= radius && !dark; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          if (src[(y + dy) * w + (x + dx)] <= darkThr) dark = true;
        }
      }
      if (dark) ink[i] = Math.min(ink[i], darkThr + 8);
    }
  }
}

function inkGrayToRgba(ink: Float32Array, w: number, h: number): RgbaImage {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    const v = Math.max(0, Math.min(255, Math.round(ink[i])));
    const o = i * 4;
    data[o] = v;
    data[o + 1] = v;
    data[o + 2] = v;
    data[o + 3] = 255;
  }
  return { width: w, height: h, data };
}
