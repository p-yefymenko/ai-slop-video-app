# LTX depth gate — start still + Union IC-LoRA

**Verdict (Gate 3 + 3b): do not ship native previs depth as the default IC-LoRA control.** Native inverse fails hard on long push-ins (S2). Remaps (hist-eq / gamma / linear) do not fix it. **Depth-Anything-on-clay (C) is the best control source tested**, but it is **not a clean full-GO**: follow drops below 0.7 on extreme camera-travel exteriors (scene 01) and is borderline on the enclosed dolly (scene 04). Prefer C where it passes; route the failing scene types elsewhere. See **Gate 3b** below.

## Setup

| Item | Value |
| --- | --- |
| Seed (all clips) | **42** |
| i2v / IC-LoRA | **0.7 / 1.0** |
| Size / fps | **448×768 / 24** via `clip_spec` + `fit_to_clip` |
| Prompt | `show_prompt(..., "sceneVideo", {videoPrompt: compile_spatial_video_prompt(scene)})` |
| Start stills | Restored from archived production stills into `output/frames/the-iron-bride/1/` (`output/frames` was empty at gate start; bytes match prior `tmp/1` Qwen stills) |

### Scenes

| ID | Scene | Why |
| --- | --- | --- |
| **S1** | ep1 **scene_05** | 2 s, 49 f, camera move, no dialogue (task-fixed) |
| **S2** | ep1 **scene_19** | Longest `scene_has_spatial_change` scene **without** `speakerId`: 5 s, 121 f (`sky_forge_exterior`, authored camera push). Only ≥5 s silent+spatial candidate in the show. |
| **S3** | ep1 **scene_13** | Dialogue (`speakerId=sela`) **and** spatial change, 5 s / 121 f (depth already on disk from step 2). Alternatives 06/10 also qualify; 13 reused existing depth. |

### Depth sources

| Tag | Source |
| --- | --- |
| A | `guides/depth_video.mp4` inverse |
| A_gamma | same PNGs → `(norm ** 0.6)` → `tmp/.../depth_inverse_gamma06.mp4` |
| B | temporary `prompts.json` `mapping=linear`, render to tmp only, restore byte-identical |
| C | Video Depth Anything on `guides/clay_24fps.mp4` (unload before LTX) |

**Chosen native variant: A (inverse).** On S1, A / A_gamma / B camera-follow means are tied (~0.994). A needs no extra encode and matches production guides. A_gamma is a negligible edge (+0.0002); B is slightly worse and linear far-field is flatter.

## S1 — scene_05 (49 frames)

| Variant | Wall s | Peak VRAM MiB | Cam-follow mean / min | Stutter mod-3 | Appearance Δ (Lab) | First-frame Δ |
| --- | --- | --- | --- | --- | --- | --- |
| A inverse | 133.6 | 15745 | **0.994 / 0.991** | 10.19 / 10.15 / 10.06 | n/a (no static bg under mask) | 6.48 |
| A_gamma | 528.4 | 15762 | **0.995 / 0.992** | 10.19 / 10.15 / 10.06 | 67.6 (25% only) | 6.50 |
| B linear | 397.6 | 15809 | **0.994 / 0.991** | 10.14 / 10.11 / 10.02 | 60.3 | 6.49 |
| C DepthAnything | 368.9 | 15787 | **0.981 / 0.977** | 10.07 / 10.01 / 9.93 | 72.1 | 6.49 |

Artifacts: `tmp/depth_gate/scene_05/` (`A_inverse.mp4`, `A_gamma06.mp4`, `B_linear.mp4`, `C_depthanything.mp4`, `contact_sheet.png`).

Contact sheet shows a clear slow push-in on all variants; start appearance holds. Wall times vary with cold/warm GGUF load — treat peak VRAM (~15.7–15.8 GB) as the hard constraint, not wall seconds.

## S2 — scene_19 (121 frames)

| Variant | Wall s | Peak VRAM MiB | Cam-follow mean / min | Stutter mod-3 | Appearance Δ | First-frame Δ |
| --- | --- | --- | --- | --- | --- | --- |
| A inverse (best native) | 90.4 | 15766 | **0.240 / 0.129** | 4.54 / 4.57 / 4.60 | 55.5 | 4.75 |
| C DepthAnything | 82.2 | 15768 | **0.783 / 0.720** | 3.83 / 3.83 / 3.83 | 35.0 | 4.72 |

**VRAM:** both 121-frame clips completed without OOM. Peak ~15.75 GB / 15.9 GB (~0.2 GB free). **No tiled VAE required.** Max frame count verified fitting: **≥121** (5 s). Separate sample-vs-VAE peaks were not isolable from Comfy history; nvidia-smi peak is overall.

Native inverse depth for this wide exterior (near/far ~60–109 m in meta) does **not** transfer camera motion into LTX. Depth-Anything-on-clay does much better on the same still + IC-LoRA stack.

## S3 — scene_13 dialogue (121 frames, MultimodalGuider)

Workflow: `content-pipeline/workflows/ltx_gemma_api_depth_dialogue.json`  
(IC-LoRA guide on node 32 → MultimodalGuider positive/negative; GuiderParameters on nodes 40/41 so they do not collide with IC-LoRA loader node 30.)

| Variant | Wall s | Peak VRAM | Cam-follow mean / min | Stutter mod-3 | Appearance Δ | First-frame Δ |
| --- | --- | --- | --- | --- | --- | --- |
| A + dialogue | 116.9 | 15792 | **0.935 / 0.909** | 1.10 / 1.10 / 1.08 | 9.5 | 4.42 |

### Dialogue compatibility

| Check | Result |
| --- | --- |
| Graph runs | **Yes** — no node-input mismatch / reject |
| Audio present | **Yes** — AAC stream on `A_inverse_dialogue.mp4` |
| IC-LoRA + MultimodalGuider | **Works** with positive/negative taken from `LTXAddVideoICLoRAGuide` (not raw `LTXVConditioning`) |
| Lip sync | **Weak / inconclusive** — facial expression changes across the clip (e.g. eyes/expression shift by mid clip); frame-accurate lip↔audio sync was **not** verified. Prompt includes the production `LIP SYNC:` prefix via `compile_spatial_video_prompt`. |

No second workaround attempted beyond wiring MultimodalGuider to the IC guide outputs (the one obvious fix vs colliding node ids / conditioning from node 16).

## Recommendation

### NO-GO for production “native previs depth → IC-LoRA” as default

1. **Real-length camera-follow fails for native A on S2** (mean 0.24). That is the gate’s primary NO-GO criterion.
2. S1 looks excellent, so short people-blocking shots can mislead; the long spatial scene is the tie-breaker.
3. VRAM is acceptable (≥121 frames at 448×768 on 16GB) — **not** the blocker.
4. Dialogue path is **technically viable** (runs + audio) but lip quality is unproven — not enough alone for GO.

### If work continues

- Prefer **Depth Anything (or another monocular) control** for camera-heavy long shots, or fix native depth (range/mapping/normalization) so S2-style wides keep a usable disparity signal after `fit_to_clip`.
- Keep seed/strength defaults: i2v **0.7**, IC **1.0**, size **448×768**.
- Dialogue JSON can stay as a side workflow; do not merge into `content:generate` until lip sync is measured.

## Artifacts

| Path | Contents |
| --- | --- |
| `tmp/depth_gate/scene_05/` | S1 controls, 4 LTX mp4s, DA-on-output, contact sheet, metrics.json |
| `tmp/depth_gate/scene_19/` | S2 A+C mp4s, contact sheet, metrics.json |
| `tmp/depth_gate/scene_13/` | S3 dialogue mp4, contact sheet, metrics.json, lip_probe/ |
| `content-pipeline/workflows/ltx_gemma_api_depth.json` | Non-dialogue depth graph |
| `content-pipeline/workflows/ltx_gemma_api_depth_dialogue.json` | MultimodalGuider depth graph |
| `content-pipeline/scripts/experiments/gate_*.py` | Temporary gate runners (not part of the pipeline) |

## Git note (this step)

Step 3 added/edited only:

- `content-pipeline/workflows/ltx_gemma_api_depth.json` (pre-existing from spike)
- `content-pipeline/workflows/ltx_gemma_api_depth_dialogue.json` (**new**)
- `content-pipeline/scripts/experiments/gate_*.py` (**temporary**; not part of the pipeline)
- `docs/ltx-depth-gate.md` (**new**)
- `tmp/depth_gate/**` outputs

Uncommitted diffs in `generate_batch.py` / `spatial_previs.py` / `prompts.json` / `clip_spec.py` are from **STEP 2**, not this gate.

---

## Gate 3b — best available control + S2 diagnosis + C breadth

Fixed rules unchanged: 448×768, 24 fps, `clip_spec` / `fit_to_clip`, seed **42**, i2v **0.7**, IC **1.0**, never Pillow-convert I;16, `LTXV_API_KEY` only via env inject (never in JSON), sequential Comfy runs. No pipeline wiring changes in this step (only `gate_*.py`, workflows already present, `tmp/`, this doc).

### Breadth scene picks

| Pick | Scene | Why |
| --- | --- | --- |
| (i) max travel+rotation | **01** | Authored travel score **39.47** (largest in show); `sky_forge_exterior`; 2 s / 49 f; empty cast |
| (ii) 1–2 chars + move | **11** | 1 character (`sela`), travel **4.04**, medium/close court framing; 2 s / 49 f |
| (iii) most characters | **06** | **6** characters + dialogue (`vardan`); 5 s / 121 f; static camera (spatial change via blocking) |
| (iv) enclosed / other loc | **04** | No true interior and no third location exist; `sun_well_court` enclosed plaza with travel **2.44** (unused by i–iii). Scenes 05/13 share this loc; 19 is `sky_forge_exterior` |

Stills restored from `tmp/1` into `output/frames/...`. Ran `content:previs` for 01/04/06/11 to produce `clay_24fps.mp4`.

### Task 1 — S2 native depth diagnosis (scene 19)

Control videos actually fed to LTX: `depth_inverse.mp4` (A) vs `depth_depthanything.mp4` (C). Sheet: `tmp/depth_gate/scene_19/control_compare.png` (A top, C bottom; 0/25/50/75/100%).

| Quantity | Value |
| --- | --- |
| Scene near / far (meta, p1/p99 of authored inverse) | **59.87 / 108.92** m |
| Valid float z percentiles (all frames) | min 56.3 · p1 59.9 · p5 62.8 · p25 70.0 · p50 76.9 · p75 84.4 · p95 96.8 · p99 108.9 · max 152.2 |

Castle region = brightest 40% of valid control pixels (nearest under inverse). Distinct 8-bit levels and std at frames 0 / 60 / 120:

| Frame | A distinct / std / mean | C distinct / std / mean |
| --- | ---: | ---: |
| 0 | 53 / 12.9 / **105** | 123 / 22.1 / **179** |
| 60 | 65 / 16.2 / **168** | 138 / 27.4 / **170** |
| 120 | 41 / 14.3 / **239** | 160 / 29.1 / **167** |

**Plain words:** native inverse is **not** merely “a bit softer” — as the camera pushes into the castle, A’s castle levels **blow toward white** (mean 105→239) and lose usable levels (~53→41 distinct). C keeps the castle mid-bright and consistent (~179→167) with ~2× the level count. So native control is **low-contrast and washing out on the castle** relative to C; that matches the pale-grey LTX failure on A.

### Task 2 — S2 native remaps (A_eq / A_gamma06 / B_linear)

Encodes under `tmp/depth_gate/scene_19/` only (guides untouched). A_eq = scene-wide CDF of inverse `norm` over all valid pixels/frames (4096 bins).

| Variant | Follow mean/min | Appear Δ | Castle L max\|Δ\| from f0 | Wall s | Peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| A inverse | 0.240 / 0.129 | 55.5 | **91.8** | 90 | 15766 |
| A_eq | 0.195 / 0.059 | 62.7 | **102.0** | 521 | 15977 |
| A_gamma06 | **0.251** / 0.151 | 56.6 | 94.7 | 481 | 15970 |
| B linear | 0.248 / 0.142 | 55.1 | 94.9 | 138 | 15911 |
| C DepthAnything | **0.783** / 0.720 | 35.0 | **5.5** | 82 | 15768 |

**Best native by metrics: A_gamma06** (follow 0.251) — still far below C and still washes the castle (contact_sheet_A_eq / A_gamma / B all go pale by ~75–100%). **None of the remaps match C visually.** Native cannot be the default.

### Task 3 — Brightness / saturation drift (estimator-free)

Lab L and chroma (OpenCV Lab, chroma = √((a−128)²+(b−128)²)), sampled at 0/25/50/75/100%.

#### S2 whole-frame

| Variant | L @ samples | L max\|Δ\| | chroma @ samples | C max\|Δ\| |
| --- | --- | ---: | --- | ---: |
| A | 65.7 / 58.0 / 52.5 / 64.7 / 86.6 | 20.9 | 9.3 / 8.3 / 7.8 / 7.2 / 6.3 | 3.1 |
| A_eq | 65.6 / 57.9 / 52.6 / 67.6 / 95.0 | 29.3 | 9.3 / 8.4 / 7.8 / 7.3 / 6.1 | 3.2 |
| A_gamma | 65.6 / 58.1 / 52.5 / 64.6 / 88.3 | 22.7 | 9.3 / 8.2 / 7.6 / 7.0 / 6.2 | 3.1 |
| B | 65.7 / 57.9 / 52.5 / 65.3 / 88.8 | 23.1 | 9.3 / 8.3 / 7.7 / 7.2 / 6.3 | 3.0 |
| C | 65.6 / 58.1 / 51.4 / 46.2 / 42.6 | 23.1 | 9.3 / 8.2 / 7.7 / 6.9 / 5.9 | 3.4 |

Whole-frame L on C **darkens** as the push-in fills the frame with dark stone (expected). Castle-region L is the washout check:

| Variant | Castle L @ samples | Castle L max\|Δ\| | Castle chroma max\|Δ\| |
| --- | --- | ---: | ---: |
| A | 27.7 / 30.4 / 35.8 / 77.3 / **119.5** | **91.8** | 0.7 |
| A_eq | 30.1 / 31.9 / 38.5 / 84.5 / **132.1** | **102.0** | 1.3 |
| A_gamma | 27.3 / 29.8 / 35.2 / 76.7 / **122.0** | 94.7 | 0.5 |
| B | 27.6 / 30.0 / 35.3 / 77.9 / **122.6** | 94.9 | 0.7 |
| C | 23.7 / 25.3 / 26.8 / 27.7 / **29.2** | **5.5** | 0.7 |

Saturation drift is small for all S2 variants; the failure mode is **brightness washout on the castle**, not desaturation.

#### Other scenes (C unless noted)

| Scene | Variant | L max\|Δ\| | C max\|Δ\| | Notes |
| --- | --- | ---: | ---: | --- |
| 05 (S1) | A / C | 2.5 / 2.1 | 1.6 / 1.7 | Stable |
| 13 (S3) | A dialogue | 1.2 | 0.5 | Stable |
| 01 | C | 19.8 | **5.2** | Noticeable darken + desat on extreme travel |
| 04 | C | 4.8 | 2.2 | Appearance stable; control depth softens late |
| 06 | C | 5.0 | 1.9 | Stable |
| 11 | C | 18.0 | 2.1 | Framing change; chroma holds |

### Task 4 — C breadth results

| Scene | Frames | Follow mean/min | Stutter mod-3 | Appear Δ | First Δ | L/C max\|Δ\| | Wall s | Peak VRAM |
| --- | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 05 S1 | 49 | **0.981 / 0.977** | 10.07 / 10.01 / 9.93 | 72.1 | 6.49 | 2.1 / 1.7 | 369 | 15787 |
| 19 S2 | 121 | **0.783 / 0.720** | 3.83 / 3.83 / 3.83 | 35.0 | 4.72 | 23.1 / 3.4 (castle L **5.5**) | 82 | 15768 |
| 13 S3 | 121 | 0.935 / 0.909 (A+dialogue) | 1.10 / 1.10 / 1.08 | 9.5 | 4.42 | 1.2 / 0.5 | 117 | 15792 |
| **01** travel | 49 | **0.551 / 0.464** | 9.34 / 9.27 / 9.21 | 30.4 | 5.44 | 19.8 / 5.2 | 1214 | 15951 |
| **11** 1-char | 49 | **0.929 / 0.820** | 13.19 / 12.91 / 12.89 | 62.6 | 5.43 | 18.0 / 2.1 | 142 | 15793 |
| **06** crowd | 121 | **0.947 / 0.945** | 0.91 / 0.91 / 0.90 | 18.3 | 5.39 | 5.0 / 1.9 | 84 | 15770 |
| **04** enclosed | 49 | **0.686 / 0.041** | 6.80 / 6.73 / 6.65 | 75.5 | 4.84 | 4.8 / 2.2 | 107 | 15782 |

VRAM fits all tested clips (~15.8 GB peak). Wall times vary with cold GGUF load (scene 01 was an outlier cold run).

### Task 5 — Cost of C (Depth Anything alone)

Measured on scene 19 clay, **121 frames @ 448×768**:

| Metric | Value |
| --- | --- |
| Wall | **9.7 s** |
| Peak VRAM during DA | **12533 MiB** |
| nvidia-smi after `free_comfy_models` (before LTX) | **1480 MiB** |
| Unloads before LTX? | **Yes** (`unloaded_before_ltx: true`) |

#### Proposed install (do not implement in this step)

| Piece | Where |
| --- | --- |
| Custom node | `https://github.com/yuvraj108c/ComfyUI-Video-Depth-Anything.git` @ **`a0db08e63d1ea571601c45cde4aaee0acdd0544d`** → add to `scripts/setup-comfyui.cjs` `REPOS` (same pattern as LTXVideo / VHS) |
| pip deps | node `requirements.txt`: opencv-python, matplotlib, pillow, imageio, imageio-ffmpeg, einops, easydict, tqdm, huggingface_hub, Imath, OpenEXR — install into Comfy venv during `content:setup-comfy` |
| Weights | `video_depth_anything_vits.pth` (**~111 MB**, 116440756 bytes) under `content-pipeline/.comfyui/models/videodepthanything/` — add download in `content:models` / `download_models.py` (or rely on node auto-download from [DepthAnything/Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything) pretrained table) |

### Task 6 — Artifacts

Per-clip contact sheets (`contact_sheet_<variant>.png`: rows = start / control-fed-to-LTX / output; cols = 0/25/50/75/100%) and mp4s under:

- `tmp/depth_gate/scene_19/` — A, A_eq, A_gamma06, B, C + controls + `control_compare.png` + `diagnose_s2.json` + `metrics_3b.json`
- `tmp/depth_gate/scene_05/` — A, C + sheets
- `tmp/depth_gate/scene_13/` — A dialogue + sheet
- `tmp/depth_gate/scene_01|04|06|11/` — C only + sheets
- `tmp/depth_gate/da_cost_121.json` (+ raw DA mp4)

### Gate 3b recommendation

| Decision | Result |
| --- | --- |
| Native as default? | **No.** Remaps fail S2 (best follow 0.25); castle washout remains. Would also need to pass ≥2 breadth scenes — not attempted after S2 remap failure. |
| Full GO with C? | **No.** Follow < 0.7 on **scene 01** (0.55) and **scene 04** (0.686); scene 01 also shows chroma drop. |
| Full NO-GO? | **No.** C passes S1/S2/S3 and breadth **11** + **06** with stable appearance and VRAM headroom. |
| **Practical call** | **Prefer C (Depth Anything on clay) as the control source for IC-LoRA**, with **routing for failing scene types**. |

**Scene types that fail under C (route elsewhere):**

1. **Extreme authored camera travel + rotation on wide exteriors** (scene **01** archetype) — follow 0.55, chroma drift.
2. **Enclosed-court dolly with sparse near geometry** (scene **04** archetype) — follow 0.686 / min 0.041; DA control itself softens late in the clip.

**Scene types that pass under C:** short people-blocking moves (05), long exterior push when DA keeps contrast (19), dialogue+blocking (13/06), single-character medium turn (11).

### Git note (Gate 3b)

Touched only: `content-pipeline/scripts/gate_*.py`, existing depth workflows, `docs/ltx-depth-gate.md`, `tmp/depth_gate/**`. Pipeline modules (`generate_batch.py`, `spatial_previs.py`, etc.) were **not** edited in 3b; any dirty pipeline files are from earlier steps.
