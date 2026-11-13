/**
 * Cross-scene AI Tutor regression gate.
 *
 * Why this exists: fixing kitten/skyline must not silently wreck line-art,
 * soft shade, or dense ink drawings. Every class below is synthetic and
 * scene-agnostic — no single photo hardcoding.
 *
 *   cd web && npm run test:ai-tutor-regression
 */
import { mkdirSync, writeFileSync, existsSync, readFileSync } from "fs";
import { join } from "path";
import {
  generatePhotoDrawGuide,
  runCanonicalStages,
  isLineArt,
  toGray,
  type RgbaImage,
} from "../shared/photoDrawGuide";
import { perimeter } from "../shared/photoDrawGuide/pipeline";

const OUT = join(process.cwd(), "../tmp/ai-tutor-regression");
mkdirSync(OUT, { recursive: true });

type CaseResult = {
  id: string;
  class: "line-art" | "photo";
  ok: boolean;
  mode: string | null;
  isLineArtDetect: boolean;
  strokes: number;
  edgeRecall: number | null;
  precision: number | null;
  dice: number | null;
  fails: string[];
};

function rgba(W: number, H: number, paint: (set: (x: number, y: number, r: number, g?: number, b?: number) => void, W: number, H: number) => void): RgbaImage {
  const data = new Uint8ClampedArray(W * H * 4);
  const set = (x: number, y: number, r: number, g = r, b = r) => {
    if (x < 0 || y < 0 || x >= W || y >= H) return;
    const i = (y * W + x) * 4;
    data[i] = r;
    data[i + 1] = g;
    data[i + 2] = b;
    data[i + 3] = 255;
  };
  for (let i = 0; i < data.length; i += 4) {
    data[i] = 245;
    data[i + 1] = 245;
    data[i + 2] = 248;
    data[i + 3] = 255;
  }
  paint(set, W, H);
  return { width: W, height: H, data };
}

/** Classic graphite on paper — must stay line-art. */
function inkPortrait(): RgbaImage {
  return rgba(360, 420, (set, W, H) => {
    // paper already filled
    const oval = (cx: number, cy: number, rx: number, ry: number, v = 35) => {
      for (let y = -ry; y <= ry; y++) {
        for (let x = -rx; x <= rx; x++) {
          if ((x * x) / (rx * rx) + (y * y) / (ry * ry) <= 1) set(cx + x, cy + y, v);
        }
      }
    };
    oval(W * 0.5, H * 0.48, 90, 120, 40);
    oval(W * 0.38, H * 0.42, 14, 10, 20);
    oval(W * 0.62, H * 0.42, 14, 10, 20);
    // nose / smile lines
    for (let t = 0; t < 40; t++) {
      set(Math.round(W * 0.5), Math.round(H * 0.5 + t * 0.4), 25);
      set(Math.round(W * 0.5 - 20 + t), Math.round(H * 0.62 + Math.abs(t - 20) * 0.2), 30);
    }
    // soft shade wash on cheek — must NOT flood strokes
    for (let y = Math.round(H * 0.5); y < Math.round(H * 0.7); y++) {
      for (let x = Math.round(W * 0.55); x < Math.round(W * 0.78); x++) {
        const d = Math.hypot(x - W * 0.66, y - H * 0.58) / 50;
        if (d < 1) set(x, y, Math.round(210 - d * 25));
      }
    }
  });
}

/** Dense ornamental line-art (thick grid + filigree ticks). */
function denseOrnament(): RgbaImage {
  return rgba(400, 400, (set, W, H) => {
    const ink = (x: number, y: number) => {
      set(x, y, 28);
      set(x + 1, y, 28);
      set(x, y + 1, 28);
    };
    for (let y = 40; y < H - 40; y += 36) {
      for (let x = 40; x < W - 40; x++) ink(x, y);
    }
    for (let x = 40; x < W - 40; x += 36) {
      for (let y = 40; y < H - 40; y++) ink(x, y);
    }
    // outer silhouette so recall has a clear subject mass
    for (let x = 36; x < W - 36; x++) {
      ink(x, 36);
      ink(x, H - 38);
    }
    for (let y = 36; y < H - 36; y++) {
      ink(36, y);
      ink(W - 38, y);
    }
    // crown-like arc of short ticks
    for (let a = 0; a < Math.PI; a += 0.14) {
      const cx = W * 0.5 + Math.cos(a) * 110;
      const cy = H * 0.35 - Math.sin(a) * 70;
      for (let t = 0; t < 14; t++) {
        ink(Math.round(cx + Math.cos(a) * t), Math.round(cy + Math.sin(a) * t));
      }
    }
  });
}

/** Colorful soft animal-like photo — must use photo mode. */
function softColorPhoto(): RgbaImage {
  return rgba(420, 480, (set, W, H) => {
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) set(x, y, 78, 95, 70);
    }
    const cx = W * 0.5;
    const cy = H * 0.52;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const dx = (x - cx) / (W * 0.38);
        const dy = (y - cy) / (H * 0.42);
        const r2 = dx * dx + dy * dy;
        if (r2 < 1) {
          const shade = 200 - r2 * 55 + ((x * 17 + y * 13) % 7);
          set(x, y, shade + 25, shade - 5, shade - 45);
        }
      }
    }
    const eye = (ex: number, ey: number) => {
      for (let y = -18; y <= 18; y++) {
        for (let x = -22; x <= 22; x++) {
          if (x * x / 484 + y * y / 324 <= 1) set(ex + x, ey + y, 40, 110, 55);
          if (x * x / 64 + y * y / 100 <= 1) set(ex + x, ey + y, 20, 25, 20);
        }
      }
    };
    eye(Math.round(W * 0.37), Math.round(H * 0.48));
    eye(Math.round(W * 0.63), Math.round(H * 0.48));
  });
}

/** Night skyline photo — photo mode, no scribble flood. */
function nightSkyline(): RgbaImage {
  return rgba(640, 280, (set, W, H) => {
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) set(x, y, 12, 16, 28);
    }
    for (const [x0, y0, bw, bh] of [
      [80, 90, 70, 160],
      [200, 40, 90, 210],
      [340, 70, 120, 180],
      [500, 55, 80, 195],
    ] as number[][]) {
      for (let y = y0; y < Math.min(y0 + bh, H * 0.78); y++) {
        for (let x = x0; x < x0 + bw; x++) {
          set(x, y, 55, 62, 75);
          if ((x + y) % 11 === 0) set(x, y, 220, 200, 120);
        }
      }
      for (let x = x0; x < x0 + bw; x++) {
        for (let t = 0; t < 3; t++) set(x, y0 + t, 210, 220, 235);
      }
    }
    for (let i = 0; i < 120; i++) {
      const x = (i * 37) % W;
      const y = 20 + ((i * 91) % Math.round(H * 0.55));
      set(x, y, 255, 240, 180);
    }
  });
}

/**
 * Soft graphite wash beside a bold ink contour on bright paper.
 * Must classify as line-art and keep wash from flooding stroke count.
 */
function shadeWashOnly(): RgbaImage {
  return rgba(320, 320, (set, W, H) => {
    // bright paper (line-art prior)
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) set(x, y, 248, 248, 250);
    }
    // soft wash (should stay mostly ignored)
    for (let y = 40; y < 200; y++) {
      for (let x = 30; x < 150; x++) {
        const d = Math.hypot(x - 85, y - 110) / 70;
        if (d < 1) set(x, y, Math.round(228 - d * 28));
      }
    }
    // bold ink subject
    for (let t = 0; t < 160; t++) {
      const x = 200 + Math.round(Math.cos(t / 18) * 48);
      const y = 160 + Math.round(Math.sin(t / 18) * 62);
      set(x, y, 32);
      set(x + 1, y, 32);
      set(x, y + 1, 32);
    }
  });
}

function jagginess(strokes: { x: number; y: number }[][]): number {
  let turns = 0;
  let segs = 0;
  for (const pts of strokes) {
    for (let i = 1; i < pts.length - 1; i++) {
      const ax = pts[i].x - pts[i - 1].x;
      const ay = pts[i].y - pts[i - 1].y;
      const bx = pts[i + 1].x - pts[i].x;
      const by = pts[i + 1].y - pts[i].y;
      const la = Math.hypot(ax, ay);
      const lb = Math.hypot(bx, by);
      if (la < 0.8 || lb < 0.8) continue;
      segs++;
      if ((ax * bx + ay * by) / (la * lb) < 0.35) turns++;
    }
  }
  return segs ? turns / segs : 0;
}

function evaluate(
  id: string,
  cls: "line-art" | "photo",
  img: RgbaImage,
  gates: {
    expectMode: "line-art" | "photo";
    minRecall?: number;
    minPrecision?: number;
    minStrokes?: number;
    maxStrokes?: number;
    maxJagginess?: number;
    maxShadeFlood?: number;
    /** When false, skip isLineArt class assertion (still checks generate mode). */
    checkDetect?: boolean;
    /** Portrait policy may refuse weak synthetics instead of scribbling. */
    allowFailClosed?: boolean;
  },
): CaseResult {
  const detect = isLineArt(toGray(img), img);
  const guide = generatePhotoDrawGuide(img);
  const staged = runCanonicalStages(img, {
    forceMode: gates.expectMode,
    useGeometry: false,
    simplifyScale: gates.expectMode === "photo" ? 1.4 : 0.85,
    maxStrokes: 420,
  });
  const fails: string[] = [];
  const failClosedOk =
    !!gates.allowFailClosed &&
    !guide.ok &&
    staged.workStrokes.length <= (gates.maxStrokes ?? 200) &&
    (guide.reason || "").length > 0;
  if (!guide.ok && !failClosedOk) fails.push("generate failed");
  if (guide.ok && guide.mode !== gates.expectMode)
    fails.push(`mode=${guide.mode} expected ${gates.expectMode}`);
  const checkDetect = gates.checkDetect !== false;
  if (checkDetect && cls === "photo" && detect)
    fails.push("isLineArt true on photo");
  if (checkDetect && cls === "line-art" && !detect)
    fails.push("isLineArt false on line-art");
  const recall = guide.validation?.edgeRecall ?? null;
  const precision = guide.validation?.edgePrecision ?? null;
  const dice = guide.validation?.dice ?? null;
  const strokes = guide.ok ? guide.strokeCount : staged.workStrokes.length;
  if (!failClosedOk && gates.minRecall != null && (recall ?? 0) < gates.minRecall)
    fails.push(`recall ${recall}`);
  if (
    !failClosedOk &&
    gates.minPrecision != null &&
    (precision ?? 0) < gates.minPrecision
  )
    fails.push(`precision ${precision}`);
  if (!failClosedOk && gates.minStrokes != null && strokes < gates.minStrokes)
    fails.push(`strokes ${strokes} < ${gates.minStrokes}`);
  if (gates.maxStrokes != null && strokes > gates.maxStrokes)
    fails.push(`strokes ${strokes} > ${gates.maxStrokes}`);
  if (gates.maxJagginess != null) {
    const jag = jagginess(staged.workStrokes);
    if (jag > gates.maxJagginess) fails.push(`jagginess ${jag.toFixed(3)}`);
  }
  if (gates.maxShadeFlood != null) {
    // shade wash region proxy: short strokes fraction should stay bounded
    const short = staged.workStrokes.filter((s) => perimeter(s) < 40).length;
    if (short > gates.maxShadeFlood) fails.push(`shadeFlood short=${short}`);
  }
  return {
    id,
    class: cls,
    ok: fails.length === 0,
    mode: guide.ok ? guide.mode : null,
    isLineArtDetect: detect,
    strokes,
    edgeRecall: recall,
    precision,
    dice,
    fails,
  };
}

const results: CaseResult[] = [
  evaluate("ink-portrait", "line-art", inkPortrait(), {
    expectMode: "line-art",
    minRecall: 0.35,
    minPrecision: 0.35,
    minStrokes: 8,
    maxStrokes: 220,
    maxShadeFlood: 90,
  }),
  evaluate("dense-ornament", "line-art", denseOrnament(), {
    expectMode: "line-art",
    minRecall: 0.25,
    minPrecision: 0.3,
    minStrokes: 12,
    maxStrokes: 400,
  }),
  evaluate("soft-color-photo", "photo", softColorPhoto(), {
    expectMode: "photo",
    minRecall: 0.05,
    minStrokes: 2,
    maxStrokes: 200,
    // Portrait policy may fail-closed on weak synthetics; allow that if no flood
    allowFailClosed: true,
  }),
  evaluate("night-skyline", "photo", nightSkyline(), {
    expectMode: "photo",
    // Macro-form GT (not raw neon sparkles) — form recall, not texture
    minRecall: 0.28,
    minStrokes: 12,
    maxStrokes: 360,
    maxJagginess: 0.55,
  }),
];

// Soft-wash guard: property test on forced line-art path (shade must not flood).
{
  const img = shadeWashOnly();
  const staged = runCanonicalStages(img, {
    forceMode: "line-art",
    useGeometry: false,
    maxStrokes: 420,
    simplifyScale: 0.85,
    inkThrBias: 10,
  });
  const short = staged.workStrokes.filter((s) => perimeter(s) < 40).length;
  const fails: string[] = [];
  if (staged.workStrokes.length < 1) fails.push("no ink contour recovered");
  if (staged.workStrokes.length > 80) fails.push(`stroke flood ${staged.workStrokes.length}`);
  if (short > 35) fails.push(`shadeFlood short=${short}`);
  results.push({
    id: "shade-wash-guard",
    class: "line-art",
    ok: fails.length === 0,
    mode: staged.mode,
    isLineArtDetect: isLineArt(toGray(img), img),
    strokes: staged.workStrokes.length,
    edgeRecall: null,
    precision: null,
    dice: null,
    fails,
  });
}

// Optional: real user crops if present from earlier session
const cropDir = join(process.cwd(), "../tmp/photo-quality-proof");
for (const name of ["kitten-source-crop", "skyline-source-crop"] as const) {
  const metaPath = join(cropDir, `${name}.json`);
  if (!existsSync(metaPath)) continue;
  const meta = JSON.parse(readFileSync(metaPath, "utf8")) as {
    width: number;
    height: number;
    path: string;
  };
  if (!existsSync(meta.path)) continue;
  const buf = readFileSync(meta.path);
  const img: RgbaImage = {
    width: meta.width,
    height: meta.height,
    data: new Uint8ClampedArray(buf.buffer, buf.byteOffset, buf.byteLength),
  };
  const expect = name.startsWith("kitten") || name.startsWith("skyline") ? "photo" : "photo";
  results.push(
    evaluate(`user:${name}`, "photo", img, {
      expectMode: expect,
      // Macro Match Gate: silhouette/form recall, not fur/window-texture recall
      minRecall: name.startsWith("kitten") ? 0.15 : 0.28,
      minStrokes: name.startsWith("kitten") ? 180 : 30,
      maxStrokes: name.startsWith("kitten") ? 1200 : 400,
    }),
  );
}

const report = {
  generatedAt: new Date().toISOString(),
  passed: results.filter((r) => r.ok).length,
  failed: results.filter((r) => !r.ok).length,
  results,
};
writeFileSync(join(OUT, "report.json"), JSON.stringify(report, null, 2));
console.log(JSON.stringify(report, null, 2));

if (report.failed) {
  console.error(
    "REGRESSION FAIL",
    results.filter((r) => !r.ok).map((r) => ({ id: r.id, fails: r.fails })),
  );
  process.exit(1);
}
console.log(`AI TUTOR REGRESSION OK (${report.passed} cases) → ${OUT}`);
