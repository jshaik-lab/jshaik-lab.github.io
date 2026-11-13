/**
 * Validated photo→guide generation with subject auto-focus + match gate.
 * Prefer auto-cropping the main subject over asking the user to crop manually.
 * Async path yields between attempts so Web Workers can cancel / stay responsive.
 */

import { contoursToGeometryStrokes } from "./geometry";
import {
  bindGeometryFitter,
  isLineArt,
  rgbaToStrokeGuide,
  toGray,
  type PhotoGuideResult,
  type PipelineOptions,
  type RgbaImage,
  type BoardRect,
} from "./pipeline";
import { validateWorkStrokes, type ValidationScores } from "./validate";
import {
  defaultRefineBudget,
  matchConfidence,
  refineMatchLoop,
  type RefineCandidate,
} from "./refineLoop";
import {
  classifySubject,
  drawPolicyFor,
  drawPolicyFromClassification,
  type DrawPolicy,
  type SubjectClass,
} from "./subjectClass";
import type { VisionHints } from "./stages";
import type { StepStrokeGuide } from "../strokeGuides/pathMath";
import {
  cropRgba,
  detectSubjectFocus,
  visionInCrop,
  type SubjectFocus,
  type SubjectRect,
} from "./subjectDetect";

bindGeometryFitter((contours, tol) => {
  const { strokes } = contoursToGeometryStrokes(contours, tol);
  return strokes.length ? strokes : contours;
});

const FAIL_COPY =
  "Almost a lesson — this frame didn’t yet yield strokes true enough to teach from. Try a brighter photo with one subject filling the view, and we’ll invent the training path again.";

export type SubjectFocusInfo = {
  cropped: boolean;
  source: SubjectFocus["source"];
  rect: SubjectRect;
  coverage: number;
  /** Cropped working image (for preview / underlay when focused). */
  image?: RgbaImage;
};

export type ValidatedGuide =
  | {
      ok: true;
      steps: StepStrokeGuide[];
      strokeCount: number;
      workSize: { w: number; h: number };
      mode: "line-art" | "photo";
      board: BoardRect;
      validation: ValidationScores;
      attemptLabel: string;
      attempts: number;
      focus?: SubjectFocusInfo;
      /** 0–1 match confidence after internal refine. */
      confidence: number;
      refineRounds: number;
      subjectClass?: SubjectClass;
      subjectTags?: SubjectClass[];
    }
  | {
      ok: false;
      reason: string;
      validation?: ValidationScores;
      attemptLabel?: string;
      attempts: number;
      focus?: SubjectFocusInfo;
      confidence?: number;
      refineRounds?: number;
      subjectClass?: SubjectClass;
      subjectTags?: SubjectClass[];
    };

export type GenerateProgress = {
  phase: string;
  attempt: number;
  confidence?: number;
  round?: number;
};

export type GenerateOptions = {
  vision?: VisionHints;
  signal?: AbortSignal;
  onProgress?: (p: GenerateProgress) => void;
};

/**
 * Real photos: forceMode "photo" now stylizes → ink drawing → line-art strokes
 * inside runCanonicalStages. Do not feed raw colorful photos into line-art
 * attempts (that path is for already-inked scans only).
 */
const PHOTO_ATTEMPTS: PipelineOptions[] = [
  {
    label: "ink-photo-1024",
    maxWork: 1024,
    useGeometry: false,
    forceMode: "photo",
    inkThrBias: 10,
    simplifyScale: 1.35,
    maxStrokes: 720,
  },
  {
    label: "ink-photo-soft",
    maxWork: 1100,
    useGeometry: false,
    forceMode: "photo",
    inkThrBias: 14,
    simplifyScale: 1.2,
    maxStrokes: 800,
  },
  {
    label: "ink-photo-tight",
    maxWork: 960,
    useGeometry: false,
    forceMode: "photo",
    inkThrBias: 8,
    simplifyScale: 1.5,
    maxStrokes: 560,
  },
];

/** Soft pencil / dense line art: polylines first (no invented ovals). */
const LINEART_ATTEMPTS: PipelineOptions[] = [
  {
    label: "poly-lineart-1024",
    maxWork: 1024,
    useGeometry: false,
    forceMode: "line-art",
    inkThrBias: 6,
    simplifyScale: 0.75,
    maxStrokes: 750,
  },
  {
    label: "poly-soft-1100",
    maxWork: 1100,
    useGeometry: false,
    forceMode: "line-art",
    simplifyScale: 0.85,
    inkThrBias: 10,
    maxStrokes: 800,
  },
  {
    label: "poly-lineart-loose",
    maxWork: 960,
    useGeometry: false,
    forceMode: "line-art",
    simplifyScale: 1.0,
    inkThrBias: 14,
    maxStrokes: 700,
  },
  {
    label: "poly-auto-line",
    maxWork: 1024,
    useGeometry: false,
    forceMode: "auto",
    simplifyScale: 0.8,
    maxStrokes: 650,
  },
];

function attemptsForInput(
  input: RgbaImage,
  focused: boolean,
  policy?: DrawPolicy,
): PipelineOptions[] {
  const lineArt =
    policy?.subjectClass === "line-art" || isLineArt(toGray(input), input);
  if (lineArt) {
    const list = LINEART_ATTEMPTS.map((a) => ({
      ...a,
      policy: policy ?? drawPolicyFor("line-art"),
    }));
    return focused ? list.slice(0, 3) : list;
  }
  const pol = policy ?? drawPolicyFor("general");
  // Class-specialized first attempt, then general photo fallbacks
  const specialized: PipelineOptions = {
    label: `ink-${pol.subjectClass}`,
    maxWork: 1024,
    useGeometry: false,
    forceMode: "photo",
    inkThrBias: Math.max(8, pol.inkThrBias),
    simplifyScale: pol.simplifyScale,
    maxStrokes: pol.maxStrokes,
    policy: pol,
  };
  const list = [
    specialized,
    ...PHOTO_ATTEMPTS.map((a) => ({
      ...a,
      maxStrokes: Math.min(a.maxStrokes ?? 400, pol.maxStrokes + 40),
      simplifyScale: Math.max(a.simplifyScale ?? 1, pol.simplifyScale * 0.9),
      policy: pol,
    })),
  ];
  return focused ? list.slice(0, 3) : list;
}

const FOCUS_PASSES: { label: string; margin: number | null }[] = [
  { label: "full-frame", margin: null },
  { label: "subject-loose", margin: 0.2 },
  { label: "subject-mid", margin: 0.1 },
  { label: "subject-tight", margin: 0.04 },
];

function fromOk(
  result: Extract<PhotoGuideResult, { ok: true }>,
  validation: ValidationScores,
  attempts: number,
  focus?: SubjectFocusInfo,
  confidence = matchConfidence(validation),
  refineRounds = 0,
  subjectClass?: SubjectClass,
  subjectTags?: SubjectClass[],
): ValidatedGuide {
  return {
    ok: true,
    steps: result.steps,
    strokeCount: result.strokeCount,
    workSize: result.workSize,
    mode: result.mode,
    board: result.board,
    validation,
    attemptLabel: result.attemptLabel ?? "ok",
    attempts,
    focus,
    confidence,
    refineRounds,
    subjectClass,
    subjectTags,
  };
}

function throwIfAborted(signal?: AbortSignal) {
  if (signal?.aborted) {
    const err = new Error("cancelled");
    err.name = "AbortError";
    throw err;
  }
}

async function yieldTick() {
  await new Promise<void>((r) => setTimeout(r, 0));
}

async function tryPipelineAsync(
  input: RgbaImage,
  vision: VisionHints | undefined,
  labelPrefix: string,
  focused: boolean,
  opts?: GenerateOptions,
  lockedPolicy?: DrawPolicy,
): Promise<{
  ok: boolean;
  result?: Extract<PhotoGuideResult, { ok: true }>;
  validation?: ValidationScores;
  reason: string;
  label?: string;
  attempts: number;
  confidence?: number;
  refineRounds?: number;
  subjectClass?: SubjectClass;
  subjectTags?: SubjectClass[];
}> {
  let lastVal: ValidationScores | undefined;
  let lastReason = FAIL_COPY;
  let lastLabel: string | undefined;
  let attempts = 0;
  const policy =
    lockedPolicy ?? drawPolicyFromClassification(classifySubject(input));
  const list = attemptsForInput(input, focused, policy);
  let bestSeed: RefineCandidate | undefined;

  for (const pipeOpts of list) {
    throwIfAborted(opts?.signal);
    await yieldTick();
    attempts++;
    const label = `${labelPrefix}/${pipeOpts.label}`;
    opts?.onProgress?.({ phase: label, attempt: attempts });
    const result = rgbaToStrokeGuide(input, {
      ...pipeOpts,
      vision,
      label,
    });
    lastLabel = label;
    if (!result.ok) {
      lastReason = result.reason;
      continue;
    }
    const validation = validateWorkStrokes(
      result.workImage,
      result.workStrokes,
      result.mode,
      {
        focusedSubject: focused,
        tonalStrokes: result.tonalWorkStrokes,
      },
    );
    lastVal = validation;
    const confidence = matchConfidence(validation);
    opts?.onProgress?.({
      phase: label,
      attempt: attempts,
      confidence,
    });
    if (
      !bestSeed ||
      confidence > bestSeed.confidence + 0.004 ||
      (confidence >= bestSeed.confidence &&
        validation.dice >= bestSeed.validation.dice)
    ) {
      bestSeed = { result, validation, confidence, label };
    }
  }

  if (!bestSeed) {
    return {
      ok: false,
      validation: lastVal,
      reason: lastReason,
      label: lastLabel,
      attempts,
    };
  }

  // Internal score→keep-correct→retry-misses loop (time-bounded).
  opts?.onProgress?.({
    phase: `${bestSeed.label}/refine`,
    attempt: attempts,
    confidence: bestSeed.confidence,
    round: 0,
  });
  const budget = defaultRefineBudget(bestSeed.result.mode);
  // Portrait/structure must not pass on recall alone — enforce class precision.
  if (policy.minPrecision >= 0.45) {
    budget.minConfidence = Math.max(budget.minConfidence, 0.45);
  }
  const refined = refineMatchLoop(bestSeed.result.workImage, bestSeed, {
    vision,
    focused,
    budget,
    minPrecision: policy.minPrecision,
    policy,
    signal: opts?.signal,
    onProgress: (p) =>
      opts?.onProgress?.({
        phase: p.phase,
        attempt: attempts + (p.round ?? 0),
        confidence: p.confidence,
        round: p.round,
      }),
  });
  attempts += refined.rounds;

  if (refined.ok && refined.best) {
    return {
      ok: true,
      result: refined.best.result,
      validation: refined.best.validation,
      reason: "",
      label: `${policy.subjectClass}/${refined.best.label}·conf${refined.best.confidence.toFixed(2)}`,
      attempts,
      confidence: refined.best.confidence,
      refineRounds: refined.rounds,
      subjectClass: policy.subjectClass,
      subjectTags: policy.tags,
    };
  }

  return {
    ok: false,
    result: refined.best?.result,
    validation: refined.best?.validation ?? bestSeed.validation,
    reason: refined.reason || lastReason,
    label: refined.best?.label ?? bestSeed.label,
    attempts,
    confidence: refined.best?.confidence ?? bestSeed.confidence,
    refineRounds: refined.rounds,
    subjectClass: policy.subjectClass,
    subjectTags: policy.tags,
  };
}

async function runGenerate(
  input: RgbaImage,
  opts?: GenerateOptions,
): Promise<ValidatedGuide> {
  const vision = opts?.vision;
  let totalAttempts = 0;
  let lastVal: ValidationScores | undefined;
  let lastReason = FAIL_COPY;
  let lastLabel: string | undefined;
  let lastFocus: SubjectFocusInfo | undefined;

  throwIfAborted(opts?.signal);
  opts?.onProgress?.({ phase: "detect-subject", attempt: 0 });
  // Classify once on the original upload — crops must inherit this policy
  // or a kitten face crop can flip to "general" and re-enable fur scribble.
  const rootClass = classifySubject(input);
  const rootPolicy = drawPolicyFromClassification(rootClass);
  opts?.onProgress?.({
    phase: rootPolicy.mixed
      ? `class:mixed:${rootPolicy.tags.join("+")}`
      : `class:${rootClass.class}`,
    attempt: 0,
    confidence: rootClass.confidence,
  });
  // Soft pencil / line-art already fills the paper — auto-crop chops off
  // bricks/body and leaves face-only junk (PRO-38 Ganesha failure).
  const grayProbe = toGray(input);
  const lineArtInput =
    rootClass.class === "line-art" || isLineArt(grayProbe, input);
  const probe = detectSubjectFocus(input, vision, 0.16);
  const passes = lineArtInput
    ? [{ label: "full-frame", margin: null as number | null }]
    : !rootPolicy.keepBackgroundStructure &&
        !rootPolicy.mixed &&
        probe.source !== "full" &&
        probe.coverage < 0.72
      ? [
          { label: "subject-loose", margin: 0.18 },
          { label: "subject-mid", margin: 0.1 },
          { label: "full-frame", margin: null },
          { label: "subject-tight", margin: 0.04 },
        ]
      : FOCUS_PASSES;

  for (const pass of passes) {
    throwIfAborted(opts?.signal);
    await yieldTick();
    let work = input;
    let hints = vision;
    let focusMeta: SubjectFocusInfo = {
      cropped: false,
      source: "full",
      rect: { x: 0, y: 0, width: input.width, height: input.height },
      coverage: 1,
    };

    if (pass.margin != null) {
      const focus = detectSubjectFocus(input, vision, pass.margin);
      if (focus.source === "full" || focus.coverage > 0.92) {
        if (pass.label !== "subject-loose") continue;
      }
      if (focus.source !== "full" && focus.coverage <= 0.92) {
        work = cropRgba(input, focus.rect);
        hints = visionInCrop(vision, focus.rect, input.width, input.height);
        focusMeta = {
          cropped: true,
          source: focus.source,
          rect: focus.rect,
          coverage: focus.coverage,
          image: work,
        };
      }
    }

    lastFocus = focusMeta;
    opts?.onProgress?.({ phase: pass.label, attempt: totalAttempts });
    const trial = await tryPipelineAsync(
      work,
      hints,
      pass.label,
      focusMeta.cropped,
      opts,
      rootPolicy,
    );
    totalAttempts += trial.attempts;
    lastVal = trial.validation ?? lastVal;
    lastLabel = trial.label ?? lastLabel;
    if (trial.ok && trial.result && trial.validation) {
      return fromOk(
        trial.result,
        trial.validation,
        totalAttempts,
        focusMeta,
        trial.confidence,
        trial.refineRounds,
        trial.subjectClass,
        trial.subjectTags,
      );
    }
    lastReason = trial.reason || lastReason;
  }

  return {
    ok: false,
    reason:
      /coach|lesson|wandered|training|faithful|true enough/i.test(lastReason)
        ? lastReason
        : FAIL_COPY,
    validation: lastVal,
    attemptLabel: lastLabel,
    attempts: totalAttempts,
    focus: lastFocus,
    subjectClass: rootPolicy.subjectClass,
    subjectTags: rootPolicy.tags,
  };
}

/** Sync wrapper (tests / Node / worker fallback). Prefer async in UI. */
export function generatePhotoDrawGuide(
  input: RgbaImage,
  vision?: VisionHints,
): ValidatedGuide {
  return generatePhotoDrawGuideSync(input, vision);
}

function tryPipelineSync(
  input: RgbaImage,
  vision: VisionHints | undefined,
  labelPrefix: string,
  focused: boolean,
  lockedPolicy?: DrawPolicy,
): {
  ok: boolean;
  result?: Extract<PhotoGuideResult, { ok: true }>;
  validation?: ValidationScores;
  reason: string;
  label?: string;
  attempts: number;
  confidence?: number;
  refineRounds?: number;
  subjectClass?: SubjectClass;
  subjectTags?: SubjectClass[];
} {
  let lastVal: ValidationScores | undefined;
  let lastReason = FAIL_COPY;
  let lastLabel: string | undefined;
  let attempts = 0;
  const policy =
    lockedPolicy ?? drawPolicyFromClassification(classifySubject(input));
  const list = attemptsForInput(input, focused, policy);
  let bestSeed: RefineCandidate | undefined;
  for (const pipeOpts of list) {
    attempts++;
    const label = `${labelPrefix}/${pipeOpts.label}`;
    const result = rgbaToStrokeGuide(input, { ...pipeOpts, vision, label });
    lastLabel = label;
    if (!result.ok) {
      lastReason = result.reason;
      continue;
    }
    const validation = validateWorkStrokes(
      result.workImage,
      result.workStrokes,
      result.mode,
      {
        focusedSubject: focused,
        tonalStrokes: result.tonalWorkStrokes,
      },
    );
    lastVal = validation;
    const confidence = matchConfidence(validation);
    if (
      !bestSeed ||
      confidence > bestSeed.confidence + 0.004 ||
      (confidence >= bestSeed.confidence &&
        validation.dice >= bestSeed.validation.dice)
    ) {
      bestSeed = { result, validation, confidence, label };
    }
  }
  if (!bestSeed) {
    return {
      ok: false,
      validation: lastVal,
      reason: lastReason,
      label: lastLabel,
      attempts,
    };
  }
  const budget = defaultRefineBudget(bestSeed.result.mode);
  if (policy.minPrecision >= 0.45) {
    budget.minConfidence = Math.max(budget.minConfidence, 0.45);
  }
  const refined = refineMatchLoop(bestSeed.result.workImage, bestSeed, {
    vision,
    focused,
    budget,
    minPrecision: policy.minPrecision,
    policy,
  });
  attempts += refined.rounds;
  if (refined.ok && refined.best) {
    return {
      ok: true,
      result: refined.best.result,
      validation: refined.best.validation,
      reason: "",
      label: `${policy.subjectClass}/${refined.best.label}·conf${refined.best.confidence.toFixed(2)}`,
      attempts,
      confidence: refined.best.confidence,
      refineRounds: refined.rounds,
      subjectClass: policy.subjectClass,
      subjectTags: policy.tags,
    };
  }
  return {
    ok: false,
    result: refined.best?.result,
    validation: refined.best?.validation ?? bestSeed.validation,
    reason: refined.reason || lastReason,
    label: refined.best?.label ?? bestSeed.label,
    attempts,
    confidence: refined.best?.confidence ?? bestSeed.confidence,
    refineRounds: refined.rounds,
    subjectClass: policy.subjectClass,
    subjectTags: policy.tags,
  };
}

function generatePhotoDrawGuideSync(
  input: RgbaImage,
  vision?: VisionHints,
): ValidatedGuide {
  let totalAttempts = 0;
  let lastVal: ValidationScores | undefined;
  let lastLabel: string | undefined;
  let lastFocus: SubjectFocusInfo | undefined;
  let lastReason = FAIL_COPY;
  const rootClass = classifySubject(input);
  const rootPolicy = drawPolicyFromClassification(rootClass);
  const lineArtInput =
    rootClass.class === "line-art" || isLineArt(toGray(input), input);
  const probe = detectSubjectFocus(input, vision, 0.16);
  const passes = lineArtInput
    ? [{ label: "full-frame", margin: null as number | null }]
    : !rootPolicy.keepBackgroundStructure &&
        !rootPolicy.mixed &&
        probe.source !== "full" &&
        probe.coverage < 0.72
      ? [
          { label: "subject-loose", margin: 0.18 },
          { label: "subject-mid", margin: 0.1 },
          { label: "full-frame", margin: null },
          { label: "subject-tight", margin: 0.04 },
        ]
      : FOCUS_PASSES;

  for (const pass of passes) {
    let work = input;
    let hints = vision;
    let focusMeta: SubjectFocusInfo = {
      cropped: false,
      source: "full",
      rect: { x: 0, y: 0, width: input.width, height: input.height },
      coverage: 1,
    };
    if (pass.margin != null) {
      const focus = detectSubjectFocus(input, vision, pass.margin);
      if (focus.source === "full" || focus.coverage > 0.92) {
        if (pass.label !== "subject-loose") continue;
      }
      if (focus.source !== "full" && focus.coverage <= 0.92) {
        work = cropRgba(input, focus.rect);
        hints = visionInCrop(vision, focus.rect, input.width, input.height);
        focusMeta = {
          cropped: true,
          source: focus.source,
          rect: focus.rect,
          coverage: focus.coverage,
          image: work,
        };
      }
    }
    lastFocus = focusMeta;
    const trial = tryPipelineSync(
      work,
      hints,
      pass.label,
      focusMeta.cropped,
      rootPolicy,
    );
    totalAttempts += trial.attempts;
    lastVal = trial.validation ?? lastVal;
    lastLabel = trial.label ?? lastLabel;
    lastReason = trial.reason || lastReason;
    if (trial.ok && trial.result && trial.validation) {
      return fromOk(
        trial.result,
        trial.validation,
        totalAttempts,
        focusMeta,
        trial.confidence,
        trial.refineRounds,
        trial.subjectClass,
        trial.subjectTags,
      );
    }
  }
  return {
    ok: false,
    reason:
      /coach|lesson|wandered|training|faithful|true enough/i.test(lastReason)
        ? lastReason
        : FAIL_COPY,
    validation: lastVal,
    attemptLabel: lastLabel,
    attempts: totalAttempts,
    focus: lastFocus,
    subjectClass: rootPolicy.subjectClass,
    subjectTags: rootPolicy.tags,
  };
}

/** Async path for Web Worker / UI — yields between attempts, honors AbortSignal. */
export function generatePhotoDrawGuideAsync(
  input: RgbaImage,
  opts?: GenerateOptions | VisionHints,
): Promise<ValidatedGuide> {
  const normalized: GenerateOptions =
    opts &&
    typeof opts === "object" &&
    ("vision" in opts || "signal" in opts || "onProgress" in opts)
      ? (opts as GenerateOptions)
      : { vision: opts as VisionHints | undefined };
  return runGenerate(input, normalized);
}
