/**
 * Offline subject-class router for AI Tutor.
 * Classical signals only (no cloud). Supports **mixed** scenes
 * (animal + buildings, nature + man-made) via multi-tag blended policy.
 */

import {
  isLineArt,
  measureArtworkSignals,
  toGray,
  type RgbaImage,
} from "./pipeline";
import {
  contrastSaliency,
  estimatePhotoForeground,
} from "./foreground";
import { computeGradients } from "./xdogEtf";

export type SubjectClass =
  | "line-art"
  | "portrait" // animals + humans close-up
  | "structure" // buildings / hard geometry / night skyline
  | "nature" // landscapes / foliage-heavy
  | "general";

export type SubjectClassResult = {
  /** Primary (highest) class — kept for compatibility. */
  class: SubjectClass;
  /** 0–1 confidence in the primary label. */
  confidence: number;
  scores: Record<SubjectClass, number>;
  /**
   * All strong classes present (primary first). Mixed photos get 2+.
   * Example: ["portrait", "structure"] for a cat in front of a skyline.
   */
  tags: SubjectClass[];
  /** True when more than one content tag is active. */
  mixed: boolean;
};

function meanChroma(img: RgbaImage): number {
  const n = img.width * img.height;
  const step = Math.max(1, Math.floor(n / 5000));
  let chroma = 0;
  let samples = 0;
  for (let i = 0; i < n; i += step) {
    const o = i * 4;
    const r = img.data[o];
    const g = img.data[o + 1];
    const b = img.data[o + 2];
    chroma += Math.max(r, g, b) - Math.min(r, g, b);
    samples++;
  }
  return chroma / Math.max(1, samples);
}

/**
 * Classify the upload so the stroke policy can specialize.
 * Emits multiple tags when several content types score strongly.
 */
export function classifySubject(img: RgbaImage): SubjectClassResult {
  const gray = toGray(img);
  const w = img.width;
  const h = img.height;
  const n = w * h;
  const scores: Record<SubjectClass, number> = {
    "line-art": 0,
    portrait: 0,
    structure: 0,
    nature: 0,
    general: 0.15,
  };

  if (isLineArt(gray, img)) {
    scores["line-art"] = 0.95;
    return {
      class: "line-art",
      confidence: 0.95,
      scores,
      tags: ["line-art"],
      mixed: false,
    };
  }

  // Artwork safety net (B&W or color): paper-backed / posterized drawings that
  // narrowly miss isLineArt must NOT become "animal/face + buildings".
  // Do NOT treat white-backdrop photos (kitten on seamless) as drawings.
  {
    const art = measureArtworkSignals(gray, img, w, h);
    const graphiteLike = art.meanChroma <= 18 || art.vividFrac <= 0.05;
    const photoOnPaper =
      art.paperBacked &&
      !art.posterized &&
      !graphiteLike &&
      art.vividFrac > 0.12 &&
      art.midToneFrac > 0.2 &&
      art.texMean >= 16;
    if (
      !photoOnPaper &&
      ((art.paperBacked && graphiteLike) ||
        (art.paperBacked && art.posterized) ||
        // Posterized comics still keep some paper/sky highlights; smooth
        // soft-color photo blobs (no paper) must stay on the photo path.
        (art.posterized &&
          art.flatFrac > 0.55 &&
          art.mean > 90 &&
          art.nearPaperFrac > 0.25))
    ) {
      scores["line-art"] = 0.9;
      return {
        class: "line-art",
        confidence: 0.9,
        scores,
        tags: ["line-art"],
        mixed: false,
      };
    }
  }

  let mean = 0;
  let dark = 0;
  let bright = 0;
  let mid = 0;
  for (let i = 0; i < n; i++) {
    const v = gray[i];
    mean += v;
    if (v < 70) dark++;
    else if (v > 200) bright++;
    else mid++;
  }
  mean /= n;
  const darkFrac = dark / n;
  const brightFrac = bright / n;
  const midFrac = mid / n;
  const chroma = meanChroma(img);

  const { mag } = computeGradients(gray, w, h);
  let magSum = 0;
  let strong = 0;
  let horiz = 0;
  let vert = 0;
  const thirdStrong = [0, 0, 0];
  const thirdStrictAxis = [0, 0, 0];
  for (let y = 2; y < h - 2; y += 2) {
    for (let x = 2; x < w - 2; x += 2) {
      const i = y * w + x;
      const m = mag[i];
      magSum += m;
      if (m < 40) continue;
      strong++;
      const gx =
        -gray[(y - 1) * w + (x - 1)] +
        gray[(y - 1) * w + (x + 1)] -
        2 * gray[y * w + (x - 1)] +
        2 * gray[y * w + (x + 1)] -
        gray[(y + 1) * w + (x - 1)] +
        gray[(y + 1) * w + (x + 1)];
      const gy =
        -gray[(y - 1) * w + (x - 1)] -
        2 * gray[(y - 1) * w + x] -
        gray[(y - 1) * w + (x + 1)] +
        gray[(y + 1) * w + (x - 1)] +
        2 * gray[(y + 1) * w + x] +
        gray[(y + 1) * w + (x + 1)];
      if (Math.abs(gx) > Math.abs(gy) * 1.35) vert++;
      else if (Math.abs(gy) > Math.abs(gx) * 1.35) horiz++;
      const tIdx = Math.min(2, Math.floor((x / w) * 3));
      if (m >= 50) {
        thirdStrong[tIdx]++;
        if (
          Math.abs(gx) > Math.abs(gy) * 1.85 ||
          Math.abs(gy) > Math.abs(gx) * 1.85
        ) {
          thirdStrictAxis[tIdx]++;
        }
      }
    }
  }
  const meanMag = magSum / n;
  const axisFrac = strong ? (horiz + vert) / strong : 0;
  const thirdSamples = Math.max(1, (w * h) / 12);
  let hasArchZone = false;
  for (let t = 0; t < 3; t++) {
    const f = thirdStrong[t] ? thirdStrictAxis[t] / thirdStrong[t] : 0;
    if (thirdStrong[t] >= thirdSamples * 0.022 && f >= 0.78) {
      hasArchZone = true;
    }
  }

  const fg = estimatePhotoForeground(img, gray);
  let fgN = 0;
  let fgCenter = 0;
  const cx = (w - 1) / 2;
  const cy = (h - 1) / 2;
  const r2max = (Math.min(w, h) * 0.42) ** 2;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!fg[y * w + x]) continue;
      fgN++;
      if ((x - cx) ** 2 + (y - cy) ** 2 <= r2max) fgCenter++;
    }
  }
  const fgFrac = fgN / n;
  const fgCenterFrac = fgN ? fgCenter / fgN : 0;

  let tex = 0;
  let texN = 0;
  for (let y = 2; y < h - 2; y += 3) {
    for (let x = 2; x < w - 2; x += 3) {
      if (!fg[y * w + x]) continue;
      const c = gray[y * w + x];
      const d =
        Math.abs(c - gray[y * w + (x + 2)]) +
        Math.abs(c - gray[(y + 2) * w + x]);
      tex += d;
      texN++;
    }
  }
  const texMean = texN ? tex / texN : 0;

  const sal = contrastSaliency(gray, w, h);
  let salMean = 0;
  for (let i = 0; i < n; i++) salMean += sal[i];
  salMean /= n;

  // Structure / night architecture / hard man-made geometry
  if (mean < 100 && darkFrac > 0.35) scores.structure += 0.45;
  if (axisFrac > 0.55 && meanMag > 25) scores.structure += 0.35;
  if (brightFrac < 0.08 && darkFrac > 0.25) scores.structure += 0.15;
  // Daytime buildings: strong axis edges even when not night, or distinct architectural zone in mixed scene
  if (axisFrac > 0.48 && meanMag > 32 && fgFrac > 0.35) scores.structure += 0.2;
  if (hasArchZone) scores.structure += 0.52;

  // Portrait (animal/human): centered soft blob, chroma, midtones, fur/skin.
  // Night architecture alone must NOT become "animal/face" (window lights).
  // A real warm animal mass beside buildings still should (mixed scenes).
  const nightArch = mean < 100 && darkFrac > 0.35 && axisFrac > 0.4;
  // Warm midtone mass inside FG — animals in night city, not yellow windows
  let warmFg = 0;
  let warmN = 0;
  for (let i = 0; i < n; i += Math.max(1, Math.floor(n / 6000))) {
    if (!fg[i]) continue;
    const o = i * 4;
    const r = img.data[o];
    const g = img.data[o + 1];
    const b = img.data[o + 2];
    const v = gray[i];
    warmN++;
    if (v > 80 && v < 220 && r > g + 12 && r > b + 20) warmFg++;
  }
  const warmFgFrac = warmN ? warmFg / warmN : 0;
  let deepIslands = 0;
  {
    const seen = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      if (!fg[i] || gray[i] > 70 || seen[i]) continue;
      let area = 0;
      const stack = [i];
      seen[i] = 1;
      while (stack.length) {
        const cur = stack.pop()!;
        area++;
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
          if (seen[ni] || !fg[ni] || gray[ni] > 70) continue;
          seen[ni] = 1;
          stack.push(ni);
        }
      }
      if (area >= 25 && area <= Math.floor(n * 0.02)) deepIslands++;
    }
  }

  // Close-up animal can fill most of the frame (fgFrac high); mixed cat+city is smaller.
  // Bright-backdrop pets often sit slightly off-center — warm FG mass or paired eye/nose islands is the signal.
  const nonWarmAnimal =
    !nightArch &&
    deepIslands >= 2 &&
    deepIslands <= 14 &&
    fgFrac > 0.08 &&
    fgFrac < 0.78 &&
    texMean > 5;
  const animalMass =
    (warmFgFrac > 0.18 &&
      fgFrac > 0.08 &&
      fgFrac < 0.7 &&
      (fgCenterFrac > 0.3 || warmFgFrac > 0.45)) ||
    nonWarmAnimal;
  const closeUpFace =
    warmFgFrac > 0.22 &&
    fgFrac > 0.2 &&
    !nightArch &&
    (fgCenterFrac > 0.45 || (warmFgFrac > 0.5 && fgFrac > 0.1));

  if (
    fgFrac > 0.1 &&
    (fgCenterFrac > 0.4 || animalMass || closeUpFace) &&
    (fgFrac < 0.92 || closeUpFace) &&
    (!nightArch || animalMass)
  ) {
    scores.portrait += 0.35;
  }
  if (chroma > 16 && (midFrac > 0.22 || animalMass || closeUpFace) && (mean > 70 || animalMass || closeUpFace))
    scores.portrait += 0.25;
  if (
    texMean > 8 &&
    texMean < 70 &&
    (mean > 70 || animalMass || closeUpFace) &&
    (fgCenterFrac > 0.3 || animalMass)
  ) {
    scores.portrait += 0.2;
  }
  if (salMean > 0.12 && (fgCenterFrac > 0.45 || animalMass) && (mean > 65 || animalMass || closeUpFace))
    scores.portrait += 0.15;
  if (animalMass) scores.portrait += 0.25;
  if (closeUpFace) scores.portrait += 0.3;
  if (deepIslands >= 2 && deepIslands <= 14) scores.portrait += 0.2;

  // Nature / foliage
  let greenBias = 0;
  let gbN = 0;
  const step = Math.max(1, Math.floor(n / 4000));
  for (let i = 0; i < n; i += step) {
    const o = i * 4;
    const r = img.data[o];
    const g = img.data[o + 1];
    const b = img.data[o + 2];
    if (g > r + 8 && g > b + 5) greenBias++;
    gbN++;
  }
  const greenFrac = greenBias / Math.max(1, gbN);
  if (texMean > 28 && greenFrac > 0.25) scores.nature += 0.4;
  if (fgFrac > 0.7 && axisFrac < 0.4) scores.nature += 0.2;
  if (greenFrac > 0.35 && midFrac > 0.45) scores.nature += 0.15;

  scores.general += 0.2;

  let best: SubjectClass = "general";
  let bestScore = -1;
  for (const k of Object.keys(scores) as SubjectClass[]) {
    if (scores[k] > bestScore) {
      bestScore = scores[k];
      best = k;
    }
  }
  if (bestScore < 0.35) best = "general";

  // Multi-tag: keep secondary classes only when they are genuinely close
  // to the winner (real mixed scenes). Loose floors false-tagged skylines
  // as "buildings + animal/face" from window chroma.
  const contentKeys: SubjectClass[] = [
    "portrait",
    "structure",
    "nature",
  ];
  const tags: SubjectClass[] = [];
  if (best === "line-art") {
    tags.push("line-art");
  } else {
    for (const k of contentKeys) {
      if (k === best) {
        tags.push(k);
        continue;
      }
      // Portrait as secondary: strict — needs warm animal mass on night cities
      if (k === "portrait") {
        const pFloor = Math.max(0.5, bestScore * 0.72);
        if (scores.portrait < pFloor) continue;
        if (nightArch && !animalMass) continue;
        tags.push(k);
        continue;
      }
      // Structure / nature as secondary: absolute presence beats relative
      // margin — a strong animal score shouldn't hide real buildings.
      if (scores[k] >= 0.5) tags.push(k);
    }
    tags.sort((a, b) => scores[b] - scores[a]);
    if (!tags.length) tags.push(best === "general" ? "general" : best);
    if (best === "general" && !tags.includes("general")) tags.push("general");
  }

  const mixed = tags.filter((t) => t !== "general" && t !== "line-art").length > 1;
  const confidence = Math.max(0.2, Math.min(0.98, bestScore));
  return { class: best, confidence, scores, tags, mixed };
}

/** Drawing policy knobs derived from subject class / mix. */
export type DrawPolicy = {
  subjectClass: SubjectClass;
  /** All active tags (primary first); length>1 ⇒ mixed scene. */
  tags: SubjectClass[];
  mixed: boolean;
  skipRidges: boolean;
  silhouetteFirst: boolean;
  strictScribble: boolean;
  maxStrokes: number;
  simplifyScale: number;
  inkThrBias: number;
  minPrecision: number;
  /** Keep background / skyline structure outside the soft FG blob. */
  keepBackgroundStructure: boolean;
};

export function labelForSubjectClass(cls: SubjectClass | undefined): string {
  switch (cls) {
    case "portrait":
      return "animal/face";
    case "structure":
      return "buildings";
    case "nature":
      return "nature";
    case "line-art":
      return "line art";
    case "general":
      return "photo";
    default:
      return "subject";
  }
}

/** User-facing label for primary + mix tags. */
export function labelForClassification(
  primary: SubjectClass | undefined,
  tags?: SubjectClass[],
): string {
  const list = (tags?.length ? tags : primary ? [primary] : [])
    .filter((t) => t !== "general")
    .map(labelForSubjectClass);
  if (!list.length) return labelForSubjectClass(primary);
  // de-dupe preserve order
  const seen = new Set<string>();
  const out: string[] = [];
  for (const s of list) {
    if (seen.has(s)) continue;
    seen.add(s);
    out.push(s);
  }
  return out.join(" + ");
}

function basePolicy(cls: SubjectClass): Omit<DrawPolicy, "tags" | "mixed"> {
  switch (cls) {
    case "line-art":
      return {
        subjectClass: cls,
        skipRidges: true,
        silhouetteFirst: false,
        strictScribble: false,
        // Dense graphite/marker drawings need headroom for ornament (eyes,
        // crown, jewelry). 400 truncated detail after outer contours.
        maxStrokes: 750,
        simplifyScale: 0.7,
        inkThrBias: 6,
        minPrecision: 0.28,
        keepBackgroundStructure: false,
      };
    case "portrait":
      return {
        subjectClass: cls,
        skipRidges: true,
        silhouetteFirst: true,
        strictScribble: true,
        // Structural contours + room for graphite coats (fur/value) so a
        // close-up cat reads as pencil art without underlay — 96 was outline-only.
        maxStrokes: 900,
        simplifyScale: 0.95,
        inkThrBias: 6,
        // Silhouette lessons cover less of photo fur edges — don't fail-closed
        minPrecision: 0.36,
        keepBackgroundStructure: false,
      };
    case "structure":
      return {
        subjectClass: cls,
        skipRidges: false,
        silhouetteFirst: false,
        strictScribble: true,
        maxStrokes: 180,
        simplifyScale: 1.45,
        inkThrBias: 4,
        // Night skylines score mid precision vs window-sparkle GT
        minPrecision: 0.42,
        keepBackgroundStructure: true,
      };
    case "nature":
      return {
        subjectClass: cls,
        skipRidges: true,
        silhouetteFirst: true,
        strictScribble: true,
        maxStrokes: 220,
        simplifyScale: 1.5,
        inkThrBias: 8,
        minPrecision: 0.48,
        keepBackgroundStructure: false,
      };
    default:
      return {
        subjectClass: "general",
        skipRidges: false,
        silhouetteFirst: false,
        strictScribble: false,
        maxStrokes: 360,
        simplifyScale: 1.25,
        inkThrBias: 6,
        minPrecision: 0.45,
        keepBackgroundStructure: false,
      };
  }
}

export function drawPolicyFor(cls: SubjectClass): DrawPolicy {
  const b = basePolicy(cls);
  return { ...b, tags: [cls], mixed: false };
}

/**
 * Blend policies when a photo mixes animals, buildings, nature, etc.
 * Example: cat + skyline → silhouette animals AND keep building structure.
 */
export function drawPolicyFromClassification(
  result: SubjectClassResult,
): DrawPolicy {
  const tags = result.tags.length ? result.tags : [result.class];
  if (tags.length === 1 || tags[0] === "line-art") {
    return { ...drawPolicyFor(tags[0]), tags, mixed: false };
  }

  const has = (c: SubjectClass) => tags.includes(c);
  const parts = tags.map((t) => basePolicy(t));
  const avg = (pick: (p: ReturnType<typeof basePolicy>) => number) =>
    parts.reduce((s, p) => s + pick(p), 0) / parts.length;

  // Mixed animal + buildings: keep animal silhouette AND man-made edges
  const silhouetteFirst = has("portrait") || has("nature");
  // Never skip ridges entirely if structure is present — buildings need them
  const skipRidges = has("structure") ? false : parts.every((p) => p.skipRidges);
  const keepBackgroundStructure =
    has("structure") || parts.some((p) => p.keepBackgroundStructure);

  return {
    subjectClass: result.class,
    tags,
    mixed: true,
    skipRidges,
    silhouetteFirst,
    strictScribble: true,
    maxStrokes: Math.round(
      Math.min(
        has("portrait") ? 900 : 360,
        Math.max(160, avg((p) => p.maxStrokes) * 1.15),
      ),
    ),
    simplifyScale: avg((p) => p.simplifyScale),
    inkThrBias: Math.round(avg((p) => p.inkThrBias)),
    minPrecision: Math.max(
      0.36,
      Math.min(...parts.map((p) => p.minPrecision)),
    ),
    keepBackgroundStructure,
  };
}
