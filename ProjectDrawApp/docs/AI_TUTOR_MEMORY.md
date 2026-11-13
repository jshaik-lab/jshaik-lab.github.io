# AI Tutor / Photo→Ink — project memory

**Read this before changing `shared/photoDrawGuide/**`, the photo-draw worker, or Match Gate.**  
Hard-won lessons from regressions (Ganesha line-art, cat/skyline photos, underlay “missed penciling”). Do not re-learn these the hard way.

Last updated: 2026-11-12 (PT) — portrait densify (cat recognizable without underlay)

---

## What “good” looks like

| Upload type | Must classify as | Must draw via | Must NOT |
|---|---|---|---|
| Graphite / ink / marker **drawing** on paper | `line-art` | ridge + deep-ink line-art path | `animal/face + buildings`, silhouette photo stylize |
| Color comic / marker illustration | `line-art` (posterized / paper-backed) | line-art + Lab color edges when chroma high | treated as photo just because colorful |
| Real pet / face photo | `portrait` | photo stylize + silhouette policy OK | `isLineArt === true` |
| Night / building skyline | `structure` | photo + keep background structure | animal/face false tag from window lights |
| Studio pet on white seamless | `portrait` (photo) | photo path | softPencil / paperBacked line-art (white backdrop ≠ drawing) |

**UI check:** enable “Show faint reference under strokes”, scrub to 100%. Dark strokes must sit on the visible pencil lines — not a fat silhouette, not shade cell-soup with gaps on ear/crown/trunk.

---

## Working architecture (do not casually rewrite)

```
classifySubject / isLineArt / measureArtworkSignals
  → drawPolicyFromClassification
  → generatePhotoDrawGuide attempts (line-art OR photo)
  → stage2Edges → skeleton → vectorize → budget → board map
  → StrokeGuidePlayer (continuous = ALL step draws)
```

Key files:

- `shared/photoDrawGuide/pipeline.ts` — `isLineArt`, `measureArtworkSignals`, board map, stroke width
- `shared/photoDrawGuide/subjectClass.ts` — multi-tag policy, line-art safety net
- `shared/photoDrawGuide/stages.ts` — stage2 line-art ridge path; **no second FG mask on line-art**
- `shared/photoDrawGuide/generate.ts` — attempt lists / match gate
- `web/src/photoDraw/jobStore.ts` — `PIPELINE_VERSION` session key (bump on semantic changes)
- `web/src/photoDraw/runJob.ts` — **fresh Worker every upload** (sticky worker hides fixes)
- `web/src/components/StrokeGuidePlayer.tsx` — continuous mode must flatten all steps’ `draw`

---

## Classification rules that work

### Drawings (B&W and color)
- Detect **paper substrate + stroke/posterized structure**, not “low chroma”.
- **Never chroma-veto** line-art (`meanChroma > 22` alone used to force every color drawing onto the photo path).
- `softPencil` / `hardInk` are **graphite-only** (`graphiteLike`). Color drawings use `colorOnPaper` / `posterized` / paper-heavy minority ink.
- Reject **photoOnPaper**: white-backdrop pets share high `paperFrac` with scans but have vivid midtone fur (`vividFrac`, `texMean`).

### Photos
- Cat / kitten → portrait, `silhouetteFirst` OK.
- Skyline → structure, keep background structure.
- Soft synthetic color blobs (regression `soft-color-photo`) are **photos**, even if posterized/flat — require `nearPaperFrac` (or real paper) before posterized→line-art safety net.

### Mixed `portrait + structure`
- Valid for real cat+skyline photos.
- **Fatal** for drawings (face + brick wall in a graphite scan). If artwork signals fire, force `line-art` tags only — never silhouette stylize.

---

## Line-art tracing rules that work

1. **Ridge + deep ink (+ XDoG∩ink)** — do **not** run shade-fill → `refineInkFillsForLineArt` on graphite (creates topographic cell-soup; underlay looks fine, strokes miss).
2. **No second FG mask** after stage2 for line-art — it clipped ears / crown / trunk that the underlay still showed.
3. **Widen subject mask** after `estimateInkSubjectMask` (~4% min-side) so protruding ears/outer ornament stay eligible.
4. **Budget after hierarchical rank**; reserve ~half for detail/ornament (outer-only slice left jewelry/ears unpenciled).
5. **Lab color edges only when `meanChroma > ~20`** — injecting Lab edges into graphite steals budget and worsens underlay miss.
6. Slightly **thicker** line-art stroke widths so underlay lines read as “penciled”.
7. Match Gate edgeRecall alone is not enough — always eyeball underlay@100%.

---

## Session / deploy mistakes (repeat offenders)

| Mistake | Symptom | Fix |
|---|---|---|
| Forgot to bump `PIPELINE_VERSION` | UI still shows old `76 strokes · photo · animal/face + buildings` | Bump `jobStore` session key; clear `qp-photo-draw-job-v*` |
| Sticky Vite worker | Code fixed, UI behavior unchanged | `resetPhotoDrawWorker()` each upload; hard-refresh |
| Only rebuilt dist, left old session | Same | Hard-refresh + Choose another photo |
| Tuned on one image | Kitten OK, Ganesha dead (or reverse) | Run regression gate below |

Current session key pattern: `qp-photo-draw-job-v7` (bump when pipeline semantics change).

### Portrait densify → graphite coats (2026-11-12 PT)
- Symptom: kitten guide ~103 strokes — silhouette + pupils only; nobody could tell it was a cat without underlay. User reference = dense graphite portrait (fur flow, eyes, M marking).
- Cause: portrait `maxStrokes: 96` + anti-fur feature gate + tonal coats wiped (`tone≈0.01`).
- Fix: portrait structural budget 900; face-ROI NMS; darkest adaptive ridges; **`buildPortraitGraphiteCoats`** (ETF hatch inside FG, budget ~560, no shade wipe). Target ≥~180 strokes and readable cat without underlay. Not photoreal 5k-stroke gallery art — directional graphite coats + features.

---

## Regression gates (run before claiming fixed)

```bash
cd web
npm run test:ai-tutor-regression          # 7 synthetic + optional user crops
npx vite-node ../scripts/prove-artwork-general.mts   # ganesha / marker / comic / cat / skyline
npx vite-node ../scripts/diag-stroke-vs-ink.mts      # underlay ink coverage (Ganesha)
# User-reported photos when present:
npx vite-node ../scripts/prove-user-fail-photos.mts
```

After code change: `npm run build` → restart Vite → **hard-refresh** → re-upload (do not trust hydrated session).

Worker hash lives under `web/dist/assets/photoDraw.worker-*.js`.

---

## Known good numbers (2026-11-12 PT, Ganesha clean)

- Class: `line-art` (not photo / not animal+buildings)
- Guide: ~750 strokes, `full-frame/poly-*`, conf ~0.76
- Stroke-on-ink coverage ~0.83, precision ~0.95+ (`diag-stroke-vs-ink`)
- Cat stays portrait/photo; skyline stays structure/photo

Older bad UI meta to recognize instantly:  
`76 strokes · photo · animal/face + buildings · ink-photo-1024`

---

## Do not

- Hardcode filenames (Ganesha, nginx, etc.) into classifiers.
- Use chroma alone to decide photo vs drawing.
- Re-enable shade-fill refine as the primary graphite path.
- Re-apply photo FG masking on line-art after stage2.
- Ship a “fix” without regression + underlay eyeball.
- Tell the user it’s fixed while session still shows `photo · animal/face + buildings`.

---

## On-device Core ML diffusion (consumer iOS path)

Classical CV photo→ink remains the web/JS tutor. For **native iOS**, prefer local Core ML (`apple/ml-stable-diffusion`) — zero accounts, keys, or billing.

Scaffold (optional SPM; compiles without packages via `#if canImport`):

| File | Role |
|---|---|
| `mobile/modules/quiet-pencil-vision/ios/CoreMLDiffusionPrompts.swift` | Consolidated <77-token positive/negative |
| `CoreMLDiffusionConfig.swift` | Mode + strength 0.65 / 0.35 / txt2img |
| `CoreMLGrayscaleCheck.swift` | `CIAreaAverage` → sketch vs photo |
| `CoreMLModelStore.swift` | URLSession download + ZIPFoundation unpack |
| `CoreMLDiffusionPipeline.swift` | `StableDiffusionPipeline` wrapper (ANE) |
| `QuietPencilVisionModule.swift` + `src/index.ts` | Expo API |

**Prompts (exact):**

- Positive base: `masterpiece traditional graphite pencil sketch on textured off-white cotton sketchbook paper, fine hand-drawn 2H HB and 4B pencil strokes, intricate cross-hatching, soft graphite smudging, natural cold-press paper grain, unshaded paper highlights, faint artist smudges, vignette edges, monochrome grayscale`
- Negative: `color, colored, vibrant, photograph, photorealistic, 3d render, digital art, smooth shading, airbrush, painting, watercolor, ink, messy, blurry, deformed, cropped, watermark, text`
- Prepend `"\(userText), "` **only** in Text-to-Image mode.

**Params:** guidance 7.5, steps 20; photo→sketch strength 0.65; sketch→sketch 0.35; txt2img `startingImage = nil`.

**JS API:** `isDiffusionAvailable`, `getDiffusionStatus`, `previewDiffusionConfig`, `downloadDiffusionModel`, `runDiffusionSketch`.

**Enable (wired):**

1. `mobile/plugins/withCoreMLDiffusion.js` — SPM `StableDiffusion` + `ZIPFoundation` on app target; copies `QPDiffusionProvider.swift`; iOS 17+.
2. QuietPencilVision pod depends on CocoaPods `ZIPFoundation` for unpack.
3. Hostable ZIP = Apple HF 6-bit palettized `split_einsum_v2` compiled (~1.5 GB) — no local torch convert required:
   `https://huggingface.co/apple/coreml-stable-diffusion-v1-5-palettized/resolve/main/coreml-stable-diffusion-v1-5-palettized_split_einsum_v2_compiled.zip`
4. Mirror/serve: `./scripts/coreml-diffusion/prepare-hostable-zip.sh [--serve]` (also `npm run diffusion:zip` in `mobile/`).
5. After `sudo xcodebuild -license accept`: `npm run prebuild:ios` then `downloadDiffusionModel()` (empty URL → HF default).

ControlNet Canny is a later layer. Classical CV still powers web + current skyline/Ganesha tutor until device has SPM build + downloaded ZIP.

---

## Related notes

- Agent-store internals: `ai-tutor-regression-safety.md`, `ai-tutor-artwork-general-fix.md`, `ai-tutor-missed-penciling-fix.md`, `internal/ai-tutor-coreml-diffusion.md`
- Proofs land under `tmp/ai-tutor-regression/`, `tmp/artwork-general/`, `tmp/stroke-vs-ink/`
