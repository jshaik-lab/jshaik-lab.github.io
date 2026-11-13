import { readFileSync, writeFileSync, existsSync } from "fs";
import { join } from "path";
import { classifySubject, generatePhotoDrawGuide, type RgbaImage } from "../shared/photoDrawGuide";

const OUT = join(process.cwd(), "../tmp/photo-quality-proof");

function loadMeta(name: string): RgbaImage | null {
  const metaPath = join(OUT, `${name}.json`);
  if (!existsSync(metaPath)) return null;
  const meta = JSON.parse(readFileSync(metaPath, "utf8")) as { width: number; height: number; path: string };
  if (!existsSync(meta.path)) return null;
  const buf = readFileSync(meta.path);
  return { width: meta.width, height: meta.height, data: new Uint8ClampedArray(buf.buffer, buf.byteOffset, buf.byteLength) };
}

const report: Record<string, unknown> = {};
const fails: string[] = [];
for (const name of ["ui-kitten-full", "kitten-source-crop"] as const) {
  const img = loadMeta(name);
  if (!img) { report[name] = { skipped: true }; continue; }
  const cls = classifySubject(img);
  const g = generatePhotoDrawGuide(img);
  report[name] = {
    size: `${img.width}x${img.height}`,
    class: cls.class,
    ok: g.ok,
    strokes: g.ok ? g.strokeCount : 0,
    precision: g.validation?.edgePrecision ?? null,
    recall: g.validation?.edgeRecall ?? null,
    conf: g.confidence ?? null,
    subjectClass: g.subjectClass,
    reason: g.ok ? null : g.reason,
    label: g.attemptLabel,
  };
  if (!g.ok) fails.push(`${name} hidden: ${g.reason?.slice(0,100)}`);
  if (g.ok && (g.strokeCount ?? 0) > 1200) fails.push(`${name} scribble ${g.strokeCount}`);
  if (g.ok && (g.strokeCount ?? 0) < 180) fails.push(`${name} too sparse ${g.strokeCount}`);
  if (/Guide stays hidden|unable to draw|Match check failed/i.test(g.reason ?? "")) fails.push("harsh copy");
}
writeFileSync(join(OUT, "full-kitten-report.json"), JSON.stringify(report, null, 2));
console.log(JSON.stringify(report, null, 2));
if (fails.length) { console.error("FAIL", fails); process.exit(1); }
console.log("FULL KITTEN PROOF OK");
