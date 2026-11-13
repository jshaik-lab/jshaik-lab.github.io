export {
  rgbaToStrokeGuide,
  bindGeometryFitter,
  toGray,
  boxBlur,
  sobelMagnitude,
  nonMaxSuppress,
  hysteresisThreshold,
  findContours,
  rdp,
  isLineArt,
  measureArtworkSignals,
  otsuThreshold,
  thinSkeleton,
  distanceTransform,
  sampleStrokeRadius,
  VIEW_W,
  VIEW_H,
  boardMapping,
  type RgbaImage,
  type ArtworkSignals,
  type PhotoGuideResult,
  type PipelineOptions,
  type BoardRect,
} from "./pipeline";

export {
  generatePhotoDrawGuide,
  generatePhotoDrawGuideAsync,
  type ValidatedGuide,
  type SubjectFocusInfo,
  type GenerateOptions,
  type GenerateProgress,
} from "./generate";

export {
  refineMatchLoop,
  matchConfidence,
  keepCorrectStrokes,
  buildMissMask,
  defaultRefineBudget,
  type RefineBudget,
  type RefineCandidate,
  type RefineLoopResult,
} from "./refineLoop";

export {
  classifySubject,
  drawPolicyFor,
  drawPolicyFromClassification,
  labelForSubjectClass,
  labelForClassification,
  type SubjectClass,
  type SubjectClassResult,
  type DrawPolicy,
} from "./subjectClass";

export {
  foregroundBoundary,
  estimateForeground,
  estimatePhotoForeground,
  fillInteriorHoles,
  smoothBinaryMask,
  grabCutLite,
} from "./foreground";

export {
  detectSubjectFocus,
  cropRgba,
  rgbaToPngDataUrl,
  type SubjectFocus,
  type SubjectRect,
} from "./subjectDetect";

export {
  validateWorkStrokes,
  type ValidationScores,
} from "./validate";

export {
  samplePrimitive,
  fitBestPrimitive,
  contoursToGeometryStrokes,
  type GeoPrimitive,
} from "./geometry";

export {
  runCanonicalStages,
  stage1Preprocess,
  stage2Edges,
  stage3Skeleton,
  stage4Vectorize,
  stage5HierarchicalSort,
  type RankedStroke,
  type StrokeTier,
  type StagePipelineResult,
  type VisionHints,
} from "./stages";

export {
  xdogEtfEdges,
  buildEtf,
  xdogResponse,
  licAlongEtf,
} from "./xdogEtf";
export { pruneSkeletonSpurs, mergeByTangent } from "./topology";
export { applyKinematics, orientStroke } from "./kinematics";
export { morphologicalEnvelope } from "./envelope";
export { buildLoomisBoard } from "./loomis";
export {
  synthesizeVisionHints,
  type VisionTwinOptions,
} from "./visionTwin";
export {
  stylizePhotoToInk,
  morphCloseBright,
  type PhotoInkResult,
} from "./photoToInk";
export {
  generalizedPhotoToInk,
  measureHfer,
  normalizeExposure,
  rollingGuidanceSmooth,
  type AdaptiveInkResult,
  type AdaptiveInkStats,
} from "./adaptiveStylize";
export { analyzeColorStructure, colorEdgeBinary } from "./colorStructure";
export {
  buildPortraitGraphiteCoats,
  buildTonalCoats,
  flattenCoats,
} from "./tonalCoats";
export { leastCostMerge, leastPathOrder } from "./leastPath";
export {
  clahe,
  extractAdaptiveRidges,
  multiScaleDogRidges,
  gateEdgesByInkDepth,
  gateEdgesByGradient,
  adaptiveInkMask,
  estimateInkSubjectMask,
  pruneSpeckEdges,
  pruneSpeckEdgesKeepFineDetail,
  extractFineOrnamentInk,
  recoverOrnamentsNearStrokes,
  inkFillToBoundary,
  punchThickInkInteriors,
  refineInkFillsForLineArt,
  extractDeepCompactRings,
} from "./localContrast";
export { filterNoiseStrokes, filterBackgroundShadeStrokes } from "./stages";
export {
  NEURAL_EDGES_AVAILABLE,
  getNeuralEdgeBackend,
  runNeuralEdgesStub,
} from "./neuralEdges";

// Geometry fitter historically bound here; stages.ts imports geometry directly.
import { bindGeometryFitter } from "./pipeline";
import { contoursToGeometryStrokes as _fitContours } from "./geometry";
bindGeometryFitter((contours, tol) => {
  const { strokes } = _fitContours(contours, tol);
  return strokes.length ? strokes : contours;
});

/** Decode an HTML ImageBitmap / HTMLImageElement into RGBA via canvas (web). */
export async function imageElementToRgba(
  source: CanvasImageSource,
  maxSide = 1400,
): Promise<{ width: number; height: number; data: Uint8ClampedArray }> {
  const w0 =
    "videoWidth" in source && (source as HTMLVideoElement).videoWidth
      ? (source as HTMLVideoElement).videoWidth
      : "naturalWidth" in source
        ? (source as HTMLImageElement).naturalWidth ||
          (source as HTMLImageElement).width
        : (source as ImageBitmap).width;
  const h0 =
    "videoHeight" in source && (source as HTMLVideoElement).videoHeight
      ? (source as HTMLVideoElement).videoHeight
      : "naturalHeight" in source
        ? (source as HTMLImageElement).naturalHeight ||
          (source as HTMLImageElement).height
        : (source as ImageBitmap).height;

  const scale = Math.min(1, maxSide / Math.max(w0, h0));
  const w = Math.max(2, Math.round(w0 * scale));
  const h = Math.max(2, Math.round(h0 * scale));

  const canvas =
    typeof OffscreenCanvas !== "undefined"
      ? new OffscreenCanvas(w, h)
      : Object.assign(document.createElement("canvas"), { width: w, height: h });
  if ("width" in canvas) {
    (canvas as HTMLCanvasElement | OffscreenCanvas).width = w;
    (canvas as HTMLCanvasElement | OffscreenCanvas).height = h;
  }
  const ctx = canvas.getContext("2d") as
    | CanvasRenderingContext2D
    | OffscreenCanvasRenderingContext2D
    | null;
  if (!ctx) throw new Error("Canvas 2D unavailable");
  ctx.drawImage(source as CanvasImageSource, 0, 0, w, h);
  const img = ctx.getImageData(0, 0, w, h);
  return { width: w, height: h, data: img.data };
}

export async function fileToRgba(file: Blob): Promise<{
  width: number;
  height: number;
  data: Uint8ClampedArray;
}> {
  const url = URL.createObjectURL(file);
  try {
    const img = await new Promise<HTMLImageElement>((resolve, reject) => {
      const el = new Image();
      el.onload = () => resolve(el);
      el.onerror = () => reject(new Error("Could not decode image"));
      el.src = url;
    });
    return imageElementToRgba(img);
  } finally {
    URL.revokeObjectURL(url);
  }
}
