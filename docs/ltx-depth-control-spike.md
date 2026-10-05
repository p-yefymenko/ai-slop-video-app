# LTX Union IC-LoRA + depth control — feasibility spike

**Verdict: yes on our Q4_K_M GGUF LTX-2.3 distilled setup, with a start still + depth video, on 16GB.** All four strength combos completed end-to-end. Camera follow from depth is only partial; start-frame appearance is mostly preserved.

## Hardware / stack

| Item | Value |
| --- | --- |
| GPU | RTX 5070 Ti, 16303 MiB |
| Base weights | `ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf` via `UnetLoaderGGUF` |
| IC-LoRA | `ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors` (~624 MB) via `LTXICLoRALoaderModelOnly` |
| Text | `GemmaAPITextEncode` + `LTXAVUseProcessedAPIEmbeds` (unchanged) |
| Sampler | Euler, 8 steps, LTX shift 2.05 / 0.95 |
| Clip size | **448×768**, 49 frames (`8n+1`), 24 fps |

### Resolution note

Pipeline default 448×800 is **not** compatible with this LoRA’s `reference_downscale_factor=2`. Latent spatial dims must be divisible by that factor; `800/32=25` is odd. Closest near-9:16 size that works: **448×768** (`14×24` latent).

## Scene choice

Task asked for `scene_06`. That scene’s camera has a single keyframe and no character motion in `10–15s`, so it cannot answer “does the camera follow the depth.” Spike used **`the-iron-bride` episode 1 scene 05** instead (authored camera move, start+end stills, clay `blockout.mp4` at `output/previs/the-iron-bride/1/scene_05/`). Start still: `tmp/1/scene_05_start.png`.

## Artifacts

| Path | Role |
| --- | --- |
| `content-pipeline/workflows/ltx_gemma_api_depth.json` | Runnable API workflow |
| `content-pipeline/.comfyui/models/loras/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors` | Downloaded by `pnpm run content:models` |
| `tmp/depth_spike/scene_05_depth_control.mp4` | Depth Anything control video |
| `tmp/depth_spike/ltx_depth_i2v*_ic*.mp4` | Four strength-matrix clips |
| `tmp/depth_spike/summary.json` | Timing / VRAM numbers |

Spike-only (not wired into `content:setup-comfy`): [ComfyUI-Video-Depth-Anything](https://github.com/yuvraj108c/ComfyUI-Video-Depth-Anything) for `LoadVideoDepthAnythingModel` / `VideoDepthAnythingProcess` / `VideoDepthAnythingOutput`.

## Exact node chain

Depth preprocess (separate pass, so Depth Anything is not resident with the GGUF):

1. `VHS_LoadVideo` — clay `blockout.mp4`, `force_rate=24`, `custom_width=448`, `custom_height=768`, `frame_load_cap=49`
2. `LoadVideoDepthAnythingModel` — `video_depth_anything_vits.pth`
3. `VideoDepthAnythingProcess` — `input_size=518`, `max_res=1280`, `precision=fp16`
4. `VideoDepthAnythingOutput` — `colormap=gray`
5. `VHS_VideoCombine` — write depth MP4 (then mux silent AAC; VHS audio extract fails on silent files)

LTX clip (`ltx_gemma_api_depth.json`), wired like the official Union Control example but on GGUF + our Gemma path + `LTXVImgToVideo`:

1. `UnetLoaderGGUF` → `LTXAVUseProcessedAPIEmbeds` → **`LTXICLoRALoaderModelOnly`** (`strength_model`, default 1.0) → `ModelSamplingLTXV`
2. `GemmaAPITextEncode` → `LTXVConditioning`
3. `LoadImage` (start still) → **`LTXVImgToVideo`** (start strength) — *kept; no end-frame `LTXVAddGuide`*
4. **`LoadVideo`** (depth control) → `GetVideoComponents` → **`LTXAddVideoICLoRAGuide`**  
   - `positive`/`negative`/`latent` from `LTXVImgToVideo`  
   - `image` = depth frames  
   - `latent_downscale_factor` from IC-LoRA loader output  
   - `frame_idx=0`, guide `strength=1.0`, `crop=disabled`
5. `LTXVEmptyLatentAudio` + `LTXVConcatAVLatent` (video latent from IC guide)
6. `BasicGuider` ← IC-LoRA positive conditioning; `SamplerCustomAdvanced`
7. `LTXVSeparateAVLatent` → **`LTXVCropGuides`** (positive/negative from IC guide, latent = video half) → `VAEDecode` / `LTXVAudioVAEDecode` → `VHS_VideoCombine`

### Video loader used

- **Control video in the LTX graph:** ComfyUI core **`LoadVideo`** + **`GetVideoComponents`** (matches the official Union Control example).
- **Blockout → depth preprocess only:** `VHS_LoadVideo` (force rate / size / frame cap). Do not feed a silent depth MP4 through `VHS_LoadVideo` in the LTX graph; VHS crashes when extracting missing audio during cache scan.

## Measurements

Depth preprocess: **28 s**, peak **~13077 MiB** (unloaded before LTX).

| i2v strength | IC-LoRA strength | Wall seconds | Peak VRAM (MiB) | OK |
| --- | --- | --- | --- | --- |
| 0.7 | 1.0 | 296.5 | 15899 | yes |
| 0.9 | 1.0 | 350.7 | 15830 | yes |
| 0.7 | 0.8 | 458.3 | 15883 | yes |
| 0.9 | 0.8 | 150.6 | 15790 | yes |

Seconds vary with cold vs warm model load; treat **~2.5–6 min / 49-frame clip** as the band, peak VRAM **~15.8–15.9 GB** (tight on 16GB).

### Qualitative

- **GGUF + LoRA load:** works. Comfy logs `gguf qtypes…`, then IC-LoRA path, then sampling. No fp8 checkpoint required for this spike.
- **Depth → camera/layout:** layout around the fire is held. The authored camera push (wider start → tighter end in clay) is only weakly followed; clips stay closer to the start still’s framing than the clay end. Depth from Depth-Anything-on-shaded-clay is a soft control, and start-frame I2V competes with it.
- **Start-frame appearance:** largely preserved (wardrobe colors, glowing floor, storm look) at both 0.7 and 0.9.
- **i2v 0.7 vs 0.9:** 0.9 locks the still harder (less drift, less useful motion). 0.7 leaves more room for motion/atmosphere while still looking like the still.
- **IC 0.8 vs 1.0:** both ran; layout differences were small. Prefer **1.0** (LoRA training default) unless over-constrained.

## Recommendation

1. **Treat Union IC-LoRA + GGUF as feasible** for a later pipeline step.
2. Ship clip size **448×768** (or another pair divisible by 64) when IC-LoRA ref0.5 is on.
3. Default strengths for a follow-up: **i2v 0.7**, **IC-LoRA 1.0**.
4. Keep depth as a **separate preprocess** from LTX (VRAM).
5. Prefer core **`LoadVideo`** for the control clip in the LTX graph.
6. Next experiment (not this spike): feed **previs native depth** (already rendered) instead of Depth Anything on shaded clay — likely stronger camera/blocking lock.

## If GGUF had failed — minimal fallbacks (not implemented)

| Fallback | What | Approx. VRAM cost on 16GB |
| --- | --- | --- |
| FP8 / distilled safetensors checkpoint | Swap `UnetLoaderGGUF` for `CheckpointLoaderSimple` on official distilled/dev weights | Full 22B distilled is typically **well above 16GB** without aggressive offload; expect OOM or heavy CPU offload (much slower). Official example also stacks `ltx-2.3-22b-distilled-lora-384-1.1` when using the *dev* ckpt. |
| Different LoRA loader | Use core `LoraLoaderModelOnly` instead of `LTXICLoRALoaderModelOnly` | Similar weight memory, but **drops `reference_downscale_factor`**; must hardcode factor `2` into `LTXAddVideoICLoRAGuide` or control tokens are wrong size. Does not fix a GGUF/LoRA apply failure by itself. |
| Smaller / shorter clip | Same GGUF path, fewer frames or lower res | Main practical VRAM lever if peak creeps over 16GB at longer scene lengths. |

## Acceptance

- [x] `content-pipeline/workflows/ltx_gemma_api_depth.json` runs end to end on GGUF + start still + depth control
- [x] IC-LoRA file added to `content:models` (`download_models.py`)
- [x] Findings recorded here; stop after this report (no generate_batch / pipeline wiring)
