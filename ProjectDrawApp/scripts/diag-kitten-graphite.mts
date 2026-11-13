/**
 * Dump portrait graphite densify stats + PPM for kitten-source-crop.
 *   cd web && npx vite-node ../scripts/diag-kitten-graphite.mts
 */
import { readFileSync, writeFileSync, existsSync } from "fs";
import { join } from "path";
import {
  classifySubject,
  drawPolicyFromClassification,
  generatePhotoDrawGuide,
  runCanonicalStages,
  type RgbaImage,
} from "../shared/photoDrawGuide";

const OUT = join(process.cwd(), "../tmp/photo-quality-proof");

function loadMeta(name: string): RgbaImage | null {
  const metaPath = join(OUT, `${name}.json`);
  if (!existsSync(metaPath)) return null;
  const meta = JSON.parse(readFileSync(metaPath, "utf8")) as {
    width: number;
    height: number;
    path: string;
  };
  if (!existsSync(meta.path)) return null;
  const buf = readFileSync(meta.path);
  return {
    width: meta.width,
    height: meta.height,
    data: new Uint8ClampedArray(buf.buffer, buf.byteOffset, buf.byteLength),
  };
}

function writePpm(path: string, w: number, h: number, rgb: Uint8Array) {
  writeFileSync(
    path,
    Buffer.concat([Buffer.from(`P6\n${w} ${h}\n255\n`), Buffer.from(rgb)]),
  );
}

function render(
  img: RgbaImage,
  strokes: { x: number; y: number }[][],
  path: string,
) {
  const w = img.width;
  const h = img.height;
  const rgb = new Uint8Array(w * h * 3);
  for (let i = 0, p = 0; i < w * h; i++, p += 4) {
    const g = Math.round(
      0.55 * img.data[p] + 0.35 * img.data[p + 1] + 0.1 * img.data[p + 2],
    );
    const v = Math.min(255, Math.round(242 + (g - 128) * 0.06));
    rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = v;
  }
  for (const s of strokes) {
    for (let i = 1; i < s.length; i++) {
      const x0 = s[i - 1].x;
      const y0 = s[i - 1].y;
      const x1 = s[i].x;
      const y1 = s[i].y;
      const n = Math.max(1, Math.hypot(x1 - x0, y1 - y0) | 0);
      for (let k = 0; k <= n; k++) {
        const t = k / n;
        const x = Math.round(x0 + (x1 - x0) * t);
        const y = Math.round(y0 + (y1 - y0) * t);
        if (x < 0 || y < 0 || x >= w || y >= h) continue;
        const o = (y * w + x) * 3;
        rgb[o] = 18;
        rgb[o + 1] = 18;
        rgb[o + 2] = 22;
      }
    }
  }
  writePpm(path, w, h, rgb);
}

const img =
  loadMeta("real-kitten-photo") ?? loadMeta("kitten-source-crop");
if (!img) {
  console.error("missing real-kitten-photo / kitten-source-crop");
  process.exit(1);
}
const cls = classifySubject(img);
const pol = drawPolicyFromClassification(cls);
const staged = runCanonicalStages(img, {
  forceMode: "photo",
  policy: pol,
  maxStrokes: pol.maxStrokes,
});
const g = generatePhotoDrawGuide(img);
const all = [...staged.workStrokes, ...staged.tonalStrokes];
render(img, all, join(OUT, "kitten-graphite-dense.ppm"));
const report = {
  class: cls.class,
  policyMax: pol.maxStrokes,
  structural: staged.workStrokes.length,
  tonal: staged.tonalStrokes.length,
  tonalMid: staged.tonalCoats.mid.length,
  tonalDark: staged.tonalCoats.dark.length,
  tonalDeep: staged.tonalCoats.deep.length,
  guideOk: g.ok,
  guideStrokes: g.ok ? g.strokeCount : 0,
  label: g.ok ? g.attemptLabel : g.reason,
  precision: g.validation?.edgePrecision ?? null,
  tone: g.validation?.toneCoverage ?? null,
};
writeFileSync(join(OUT, "kitten-graphite-dense.json"), JSON.stringify(report, null, 2));
console.log(JSON.stringify(report, null, 2));
