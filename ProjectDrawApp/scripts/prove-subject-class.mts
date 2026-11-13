/**
 * Prove subject-class router + portrait anti-scribble policy.
 *   cd web && npx vite-node ../scripts/prove-subject-class.mts
 */
import { readFileSync, existsSync } from "fs";
import { join } from "path";
import {
  classifySubject,
  generatePhotoDrawGuide,
  runCanonicalStages,
  drawPolicyFor,
  type RgbaImage,
} from "../shared/photoDrawGuide";

function softKitten(): RgbaImage {
  const W = 420;
  const H = 480;
  const data = new Uint8ClampedArray(W * H * 4);
  const set = (x: number, y: number, r: number, g = r, b = r) => {
    if (x < 0 || y < 0 || x >= W || y >= H) return;
    const i = (y * W + x) * 4;
    data[i] = r;
    data[i + 1] = g;
    data[i + 2] = b;
    data[i + 3] = 255;
  };
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) set(x, y, 78, 95, 70);
  const cx = W * 0.5;
  const cy = H * 0.52;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const dx = (x - cx) / (W * 0.38);
      const dy = (y - cy) / (H * 0.42);
      const r2 = dx * dx + dy * dy;
      if (r2 < 1) {
        const shade = 200 - r2 * 55 + ((x * 17 + y * 13) % 7);
        // fur-like local noise
        const n = ((x * 13 + y * 7) % 9) - 4;
        set(x, y, shade + 25 + n, shade - 5 + n, shade - 45);
      }
    }
  }
  const eye = (ex: number, ey: number) => {
    for (let y = -18; y <= 18; y++)
      for (let x = -22; x <= 22; x++) {
        if (x * x / 484 + y * y / 324 <= 1) set(ex + x, ey + y, 40, 110, 55);
        if (x * x / 64 + y * y / 100 <= 1) set(ex + x, ey + y, 20, 25, 20);
      }
  };
  eye(Math.round(W * 0.37), Math.round(H * 0.48));
  eye(Math.round(W * 0.63), Math.round(H * 0.48));
  return { width: W, height: H, data };
}

function nightCity(): RgbaImage {
  const W = 640;
  const H = 280;
  const data = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < data.length; i += 4) {
    data[i] = 12;
    data[i + 1] = 16;
    data[i + 2] = 28;
    data[i + 3] = 255;
  }
  const set = (x: number, y: number, r: number, g = r, b = r) => {
    if (x < 0 || y < 0 || x >= W || y >= H) return;
    const i = (y * W + x) * 4;
    data[i] = r;
    data[i + 1] = g;
    data[i + 2] = b;
  };
  for (const [x0, y0, bw, bh] of [
    [80, 90, 70, 160],
    [200, 40, 90, 210],
    [340, 70, 120, 180],
  ] as number[][]) {
    for (let y = y0; y < Math.min(y0 + bh, H * 0.78); y++)
      for (let x = x0; x < x0 + bw; x++) {
        set(x, y, 55, 62, 75);
        if ((x + y) % 11 === 0) set(x, y, 220, 200, 120);
      }
    for (let x = x0; x < x0 + bw; x++) for (let t = 0; t < 3; t++) set(x, y0 + t, 210, 220, 235);
  }
  return { width: W, height: H, data };
}

const kitten = softKitten();
const city = nightCity();
const kClass = classifySubject(kitten);
const cClass = classifySubject(city);
const kPol = drawPolicyFor(kClass.class);
const kStaged = runCanonicalStages(kitten, {
  forceMode: "photo",
  policy: kPol,
  maxStrokes: kPol.maxStrokes,
  simplifyScale: kPol.simplifyScale,
});
const kGuide = generatePhotoDrawGuide(kitten);
const cGuide = generatePhotoDrawGuide(city);

const report: Record<string, unknown> = {
  kitten: {
    class: kClass,
    stagedStrokes: kStaged.workStrokes.length,
    guide: {
      ok: kGuide.ok,
      subjectClass: kGuide.subjectClass,
      strokes: kGuide.ok ? kGuide.strokeCount : 0,
      precision: kGuide.validation?.edgePrecision ?? null,
      recall: kGuide.validation?.edgeRecall ?? null,
      conf: kGuide.confidence ?? null,
      reason: kGuide.ok ? null : kGuide.reason,
    },
  },
  city: {
    class: cClass,
    guide: {
      ok: cGuide.ok,
      subjectClass: cGuide.subjectClass,
      strokes: cGuide.ok ? cGuide.strokeCount : 0,
      precision: cGuide.validation?.edgePrecision ?? null,
      conf: cGuide.confidence ?? null,
    },
  },
};

// Optional real crop
const cropMeta = join(process.cwd(), "../tmp/photo-quality-proof/kitten-source-crop.json");
if (existsSync(cropMeta)) {
  const meta = JSON.parse(readFileSync(cropMeta, "utf8")) as {
    width: number;
    height: number;
    path: string;
  };
  if (existsSync(meta.path)) {
    const buf = readFileSync(meta.path);
    const img: RgbaImage = {
      width: meta.width,
      height: meta.height,
      data: new Uint8ClampedArray(buf.buffer, buf.byteOffset, buf.byteLength),
    };
    const cls = classifySubject(img);
    const g = generatePhotoDrawGuide(img);
    report.userKitten = {
      class: cls.class,
      ok: g.ok,
      strokes: g.ok ? g.strokeCount : 0,
      precision: g.validation?.edgePrecision ?? null,
      recall: g.validation?.edgeRecall ?? null,
      conf: g.confidence ?? null,
      subjectClass: g.subjectClass,
      reason: g.ok ? null : g.reason?.slice(0, 120),
    };
  }
}

console.log(JSON.stringify(report, null, 2));

const fails: string[] = [];
if (kClass.class !== "portrait" && kClass.class !== "general")
  fails.push(`kitten class ${kClass.class}`);
if (kStaged.workStrokes.length > 1200)
  fails.push(`kitten staged scribble ${kStaged.workStrokes.length}`);
// Soft synthetic blob may fail-closed (good) — real user crop must draw a cat.
if (kGuide.ok && (kGuide.validation?.edgePrecision ?? 0) < 0.28)
  fails.push(`kitten precision ${kGuide.validation?.edgePrecision}`);
if (kGuide.ok && kGuide.strokeCount > 1200)
  fails.push(`kitten stroke flood ${kGuide.strokeCount}`);
// Soft synthetic kitten may stay sparse (good anti-scribble); density gate is on userKitten.
if (kGuide.ok && (kGuide.confidence ?? 0) < 0.28)
  fails.push(`kitten conf too low ${kGuide.confidence}`);
if (cClass.class !== "structure" && cClass.class !== "general")
  fails.push(`city class ${cClass.class}`);

const userKitten = report.userKitten as
  | {
      ok?: boolean;
      strokes?: number;
      precision?: number | null;
      conf?: number | null;
      reason?: string | null;
    }
  | undefined;
if (userKitten) {
  if (!userKitten.ok)
    fails.push(`user kitten hidden: ${userKitten.reason ?? "?"}`);
  if (userKitten.ok && (userKitten.strokes ?? 0) > 1200)
    fails.push(`user kitten scribble ${userKitten.strokes}`);
  if (userKitten.ok && (userKitten.strokes ?? 0) < 180)
    fails.push(`user kitten too sparse ${userKitten.strokes}`);
  if (userKitten.ok && (userKitten.precision ?? 0) < 0.28)
    fails.push(`user kitten precision ${userKitten.precision}`);
  if (userKitten.ok && (userKitten.conf ?? 0) < 0.28)
    fails.push(`user kitten conf ${userKitten.conf}`);
}

if (fails.length) {
  console.error("FAIL", fails);
  process.exit(1);
}
console.log("SUBJECT CLASS PROOF OK");
