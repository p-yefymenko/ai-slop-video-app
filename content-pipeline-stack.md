# Content pipeline stack

A show is one authored JSON file. The pipeline turns that file into a vertical episode on a local GPU. Geometry, blocking, and cameras are deterministic. Image and video models only paint and animate what the blockout already decided.

Commands live in the root `package.json` (`content:*` and `view`). Python runs through those scripts. Qwen, TRELLIS.2, Pixal3D, and LTX run through ComfyUI at `http://127.0.0.1:8188`. Florence-2 runs in that same virtualenv through Transformers, not through ComfyUI.

Depth-control research notes (not the live path): [docs/ltx-depth-control-spike.md](docs/ltx-depth-control-spike.md), [docs/ltx-depth-gate.md](docs/ltx-depth-gate.md). Gate runners live under `content-pipeline/scripts/experiments/` and are not part of the pipeline.

## Machine

Generation is offline, on the GPU machine (an RTX 5070 Ti, 16GB). CUDA 12.8 PyTorch is installed into the ComfyUI virtualenv so Blackwell cards work. Qwen, the mesh models, Depth Anything, and LTX are never resident together: each stage unloads idle weights before the next one starts. LTX peak VRAM on the depth-control path is about **15.8 GB**.

Text for LTX is encoded by the LTX Gemma API (`LTXV_API_KEY` in `content-pipeline/.env`). That call is free. Video denoising stays local. The key is injected into the graph at run time and is not stored in the workflow JSON.

`pnpm run content:comfy` starts ComfyUI with listen `127.0.0.1:8188` only. Set `COMFY_ARGS` to replace those defaults (for example `COMFY_ARGS=--reserve-vram 2`). On this machine `--reserve-vram 2` reduced LTX shared-memory spill but slowed Qwen stills, so it is **not** the default.

## What is authored

| Path | Role |
| --- | --- |
| `content-pipeline/shows/<id>/script.json` | The show. `ShowScript` in `packages/shared/src/script.ts`. Folder name matches `id`. A character has `body` (always sent: build, skin) and `attributes` tagged with body parts. Age is a face-tagged attribute. A still sends an attribute only when any of its parts is at least `partMinPixelHeight` tall and covers at least `partMinScreenFraction` of the frame. |
| `content-pipeline/prompts.json` | Shared Qwen and LTX templates, plus renderer values: `stillOpening`, `rowDepthRatio`, `landmarkMinScreenFraction`, `partMinPixelHeight`, `partMinScreenFraction`, `endStill` (default `false`), `ltxStartStrength`, `ltxIcLoRAStrength`, `controlDepth`, `depthAnything`, `cameraTravelWarnThreshold`, `sceneRenderOptions`, and `depthVideo` (native previs depth export only). Not show content. |
| `content-pipeline/workflows/*.json` | ComfyUI graphs the batch scripts fill in and post to `/prompt`. |

`spatialTimeline` is the physical source of truth: measured locations, character and prop keyframes, and a camera path per shot. Scene length is `timeRangeSeconds`. Prompt-only scenes are invalid.

Schema space is meters, X right, Y forward, Z up. glTF is Y-up, forward −Z. `content-pipeline/scripts/coords.py` is the only converter. The stage viewer reads Y-up and does not convert.

## Stages

`content:render` runs meshes, clay previs, stills, and clips in that order. It does not draw plates and it does not pause for review. Plates are a separate step so those pictures can be kept or redrawn before meshing.

```text
script.json
    │
    ├─ content:plates      Qwen-Image-Edit-2511          plates
    ├─ content:assets      TRELLIS.2 / Pixal3D           meshes
    ├─ content:landmarks   Florence-2                    positions in the script
    ├─ content:previs      moderngl                      clay playblast + guides
    ├─ content:frames      Qwen-Image-Edit-2511          portraits + start stills
    └─ content:generate    Depth Anything → LTX-2.3      control depth + scene clips + episode cut
```

Plates, meshes, stills, and clips skip files already on disk. One selected still or clip is rebuilt with `--force`. `content:previs` always rewrites the playblast and guides. After a structural script rewrite, `content:archive` moves that show’s generated files aside and keeps the reviewed character portraits.

`pnpm run content:validate` checks every show script. Errors fail the command. A high authored camera travel+rotation score (threshold **20**, same formula as `camera_travel_score`) prints a **warning** and does not fail. The warning text notes that scene types like scene 04 (low travel, weak depth follow) are not predicted by this score.

### 1. Plates — Qwen-Image-Edit-2511

`content:plates` draws and stops. ComfyUI must be running.

An empty location is one 1024×1024 picture of the place, prompted as a movie set (`output/plates/<show>/<locationId>/plate.png`). A location with people is one isolated object per landmark. A character is one full-body plate from `body` plus every attribute. Landmarks that share appearance and size are drawn once.

Graph: `workflows/qwen_asset_plate.json`. The 4-step Lightning LoRA, CFG 1, AuraFlow shift 3.1, denoise 1.0. The sampler starts from noise. The attached image is a blank canvas the prompt tells the model to ignore.

### 2. Meshes — TRELLIS.2 and Pixal3D

`content:assets` turns each reviewed plate into one mesh. A missing plate stops the build. It does not draw.

| Subject | Model | Surface |
| --- | --- | --- |
| Empty location, landmark | TRELLIS.2 int8 | Untextured. Fitted uniformly into `sizeMeters` or the landmark `size`. |
| Character | Pixal3D int8 | Front matches the plate. Plate color is stored on each vertex. Fitted to standing height. Front faces schema +Y (body yaw 0). |

Both remove the background with BiRefNet, condition with DINOv3, and run a shape cascade at 1024 through the TRELLIS.2 shape VAE. Pixal3D adds a texture VAE and MoGe so the projected features line up with the photo. ComfyUI’s DecimateMesh caps the mesh. The default limit is 10,000,000 triangles. A character mesh already under that limit is left as generated. Small openings on character meshes are covered with new triangles on the vertices already around the hole.

A missing character mesh falls back to a volume in previs. That volume is left out of the edge guide.

### 3. Landmark positions — Florence-2

`content:landmarks` runs after an empty location’s mesh exists. It is not part of `content:render`. ComfyUI does not need to be running. If ComfyUI is holding the GPU, Florence-2 can run out of memory.

It shades the mesh from known cameras, asks Florence-2-large where each landmark name is, and writes `position` into the script. A location with people already has those positions, so it is skipped. A position already set is kept unless `--force` is passed.

### 4. Clay previs — no diffusion model

`content:previs` renders the timeline with moderngl. One GPU draw produces the shaded clay frame and its depth buffer. People and cameras stay on the script marks.

An empty location draws its one mesh, and the camera stays where it was authored. A location with people draws an open floor plus one mesh per landmark, placed at the script position and fitted to that landmark’s size.

Output is 768×1360. The playblast is 8 fps. Per scene: `output/previs/<show>/<episode>/scene_XX/blockout.mp4`, `start.png`, `end.png`, and `guides/`. Every scene is joined into `output/previs/<show>/<episode>/blockout.mp4`. A scene still missing leaves the episode file untouched.

Guides are written for the **start** and **end** of every scene (end guides are for review; `content:frames` does not generate end stills when `endStill` is false). Additional clip-rate guides:

| Guide | What it is |
| --- | --- |
| `guides/clay_24fps.mp4` | Clay playblast resampled to LTX size/rate (448×768, 24 fps, `8n+1` frames). Input to Depth Anything. |
| `guides/control_depth.mp4` | Written by `content:generate` (not previs): Depth Anything on `clay_24fps.mp4`. |
| `guides/depth_video.mp4` (+ `guides/depth_video/`) | Native inverse (or linear) camera depth at clip rate. Still exported for review and experiments. **Unused by `content:generate`.** |

Per-frame start/end guides:

| Guide | What it is |
| --- | --- |
| `depth` | Camera-forward depth. Near surfaces are bright. Empty space is black. Character meshes stay in this map, so position and occlusion are fixed. |
| `clothes` | Those character meshes from this camera, the place cut away. Cloth, skin, and hair are each one flat color taken from the plate. A small stain takes the color around it. A landmark or prop in front of a person leaves that pixel black, so the cutout matches depth occlusion. |
| `edges` | Outlines where depth jumps or a surface meets empty space. Character meshes stay in this map when they are in frame. The capsule fallback does not. |
| `pose` | OpenPose skeleton, used only when the clothes cutout is missing. |
| `backdrop` | Empty space from this camera in flat sky, ground, and surround color. Landmarks, props, and people are black. The open floor keeps the ground color. |
| `faces` | Head masks for people whose face points at the camera, plus a JSON list of those ids. |
| `parts` | JSON: per `characterId`, visible pixels, width, and height per body part. Face and eyes are the front half of the head in mesh-local space (schema +Y), cut at the head band's bounding-box center on the forward axis. Eyes are the height band 0.86–0.91 of standing height on that front half. Same depth test as clothes. |
| `normal` | Surface normals. Written for review. |

### 5. Scene stills — Qwen-Image-Edit-2511

`content:frames` talks only to the ComfyUI HTTP API. It generates character portraits first, then each scene **start** still. With `endStill: false` in `prompts.json` (the default), it does **not** write `scene_XX_end.png`. Previs end guides remain on disk for review only.

Portraits use `workflows/qwen_image_edit.json`: same Lightning settings, 768×1360, from `body` plus every attribute, on a blank canvas. The template keeps a plain shirt and no costume. They land in `output/frames/<show>/characters/<id>.png`.

Scene stills use `workflows/qwen_image_edit_spatial.json` at 768×1360, 4 steps, CFG 1, AuraFlow shift 3.1, denoise 1.0. The sampler starts from noise. The shaded clay frame is not an input. CFG 1 has no negative channel, so the blockout still does not use “do not” sentences. ControlNet is not used. Surface normals are written for review and are not sent to Qwen. Clothes and the empty-space plate are never reference latents in the same pass.

Picture order:

- **Shot with a clothes cutout.** Depth, then the clothes cutout, then edges. All three are reference latents on `TextEncodeQwenImageEditPlus`, so garment color is copied and the outlines are kept. The empty-space plate is not attached: it is black where the person stands, and a reference latent would paint that black over the cloth.
- **Shot with people and no clothes cutout.** Depth, pose, edges, plus the empty-space plate as picture 4. The encoder switches to the local `TextEncodeQwenBackdrop` node.
- **Empty shot.** Depth and edges, plus the empty-space plate as picture 3. Same `TextEncodeQwenBackdrop` encoder.

Every shot uses one prompt skeleton, filled by `still_prompt` in `generate_batch.py` and the people builder in `still_people.py`. `spatialBlockout` in `prompts.json` is `{stillPrompt}`. Order: `stillOpening` from `prompts.json`, a legend for each attached picture, one keep sentence hardcoded in `still_prompt` (“Keep the shape, position, and occlusion from the pictures.”), the people sentence on a clothes shot, the visible landmark appearances, then “Behind and around:” and the sky, ground, and surround this camera shows. An empty slot is left out, including “No people.” Pose and empty shots omit the people sentence. The still does not read `scene.imagePrompt`. Character names and ids are not sent to Qwen. The log may keep `characterId`.

On a clothes shot, the people sentence is built from `body`, visible `attributes`, and blocking geometry: depth order, screen side, and occlusion. Visible people are counted in a number word. Each visible person is `{position}: {body, attributes…}` joined with commas. A still sends an attribute only when any of its tagged parts is at least `partMinPixelHeight` tall **and** covers at least `partMinScreenFraction` of the frame (`prompts.json`; one pair of numbers for every part). Age is a face-tagged attribute, so a 10 px face or a 21 px grazing sliver does not receive “52 years old”. If `body` is missing for a visible character, that person's text is left out and the log warns with their id. The log records each part's pixel height and screen fraction, each attribute sent or dropped, and which condition failed when it was dropped. Figures farther than `nearest_depth * rowDepthRatio` (`prompts.json`, 1.5) stand in a row behind, left to right, each with their own line, but only when at least two people qualify. Each other person is named by screen side; “in the foreground” is added only when that behind group exists. A figure whose visible fraction is below the cutoff is partly hidden. The log records `rowDepthRatio` and the farthest/nearest depth ratio once per frame. Character names and ids are not sent. Hair and feet on a distant figure still come from the clothes cutout when their text is dropped.

A landmark mesh is named when its visible front surface covers at least `landmarkMinScreenFraction` of the frame (`prompts.json`, 0.008). The log records each landmark, whether it was sent, its screen fraction, and the skip reason if it was left out. Trailing plate instructions such as “a single object” or “no walls” are dropped from landmark appearance and from the backdrop sky, ground, and surround lines. The script text itself is not edited.

Prompt logs and the attached pictures are written to `output/frames/<show>/<episode>/inputs/scene_XX_start/`. The blockout pass in `log.json` records `people` and `landmarks` the same way: each entry says whether it was sent, and the skip reason if it was left out. The episode folder keeps `scene_XX_start.png` only (no end still when `endStill` is false).

Identity face painting exists in the spatial graph and is off (`FACE_PASS = False`). LTX never receives a character portrait.

### 6. Clips — LTX-2.3 distilled-1.1 + Depth Anything control

`content:generate` animates each reviewed start still. Clip geometry is shared (`clip_spec.py`): **448×768**, **24 fps**, length **`8n+1`** for the scene duration. Stills are center-cropped and resized through `fit_to_clip` before LTX.

**Routing**

| Scene | Graph | Control |
| --- | --- | --- |
| Spatial change (camera/blocking moves) | `workflows/ltx_gemma_api_depth.json`, or `ltx_gemma_api_depth_dialogue.json` when `speakerId` is set | Start still + Depth Anything control video through the union IC-LoRA |
| No spatial change | `workflows/ltx_gemma_api.json` | Start still only (plain image-to-video) |
| Override | `sceneRenderOptions["<show>/<episode>/<scene>"] = {"depthControl": false}` | Forces the plain graph even when the scene is spatial; logs `depth control disabled by override` |

**Depth Anything pass (spatial scenes, before LTX)**

1. Load `guides/clay_24fps.mp4`.
2. Run Video Depth Anything (`video_depth_anything_vits.pth`, small) via ComfyUI-Video-Depth-Anything.
3. Write `guides/control_depth.mp4` (H.264 + silent AAC).
4. Unload Comfy models (`/free`) before queuing LTX.

Native `guides/depth_video.mp4` is not read by this stage.

| Setting | Value |
| --- | --- |
| Weights | `ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf` |
| IC-LoRA (spatial) | `ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors` via `LTXICLoRALoaderModelOnly` |
| Video VAE / audio VAE | Matching LTX-2.3 distilled VAEs |
| Text | `GemmaAPITextEncode` against a tiny safetensors stub that carries the API `model_id`. Local node `LTXAVUseProcessedAPIEmbeds` marks those embeddings as already projected to 6144 so they match the Q4 GGUF. |
| Size / rate | 448×768, 24 fps, `8n+1` frames |
| Sampler | Euler, 8 steps, LTX shift (max 2.05, base 0.95) |
| Start frame | `LTXVImgToVideo` at `ltxStartStrength` (default **0.7**) |
| IC-LoRA strength | `ltxIcLoRAStrength` (default **1.0**) on spatial/depth graphs |
| End frame guide | Not used. No `LTXVAddGuide` end still. |
| Dialogue | When `speakerId` is set on the plain graph, or on the depth-dialogue graph, `MultimodalGuider` raises joint audio and video guidance (`modality_scale` 3, cross-attention on). Silent scenes keep `BasicGuider`. |
| Prompt | `sceneVideo`: continue from the still. `videoPrompt` is the line and the performance. A speaking shot also prepends `LIP SYNC: {speakerId}…`, so that id does go to Gemma. Camera and blocking come from the timeline; spatial scenes also follow the depth control video. |
| Mux | Video Helper Suite, H.264, CRF 19, with the decoded LTX audio. |

Clips land in `output/generate/<show>/<episode>/scene_XX.mp4`. ffmpeg concatenates them into `episode.mp4` in that folder. `concat_videos` **aborts** if clip width, height, or fps differ (it does not auto-regenerate). A partial clip render leaves `episode.mp4` alone.

`content:upload` pushes the finished MP4s and thumbnails to Cloudflare R2 and registers episodes on the Worker. That step is distribution, not generation.

## Models on disk

`pnpm run content:models` downloads these into `content-pipeline/.comfyui/models/` and skips files already present.

| Stack | Files | Licence (model card) |
| --- | --- | --- |
| LTX-2.3 | Q4_K_M GGUF (~14GB), video VAE, audio VAE, distilled API stub | LTX-2 Community License |
| LTX union IC-LoRA | `ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors` (~654 MB) under `models/loras/` | LTX-2 Community License ([Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control)) |
| Video Depth Anything (small) | `video_depth_anything_vits.pth` (~111 MB, 116440756 bytes) under `models/videodepthanything/` | Apache-2.0 ([depth-anything/Video-Depth-Anything-Small](https://huggingface.co/depth-anything/Video-Depth-Anything-Small)) |
| Qwen stills | `qwen-image-edit-2511-Q4_K_M.gguf` (~13GB), `qwen_2.5_vl_7b_fp8_scaled` text encoder, `qwen_image_vae`, 4-step Lightning LoRA | Apache-2.0 |
| TRELLIS.2 | int8 UNet (~5GB), DINOv3 ViT-L, shape VAE, BiRefNet | MIT |
| Pixal3D | int8 UNet (~5.2GB), DINOv3 with NAF, texture VAE, MoGe-2. Shares the TRELLIS shape VAE. | MIT |
| Unused | YuNet ONNX face detector. Downloaded by `content:models`; no stage reads it. | — |

`content:assets -- --credits` rewrites `docs/CREDITS.md` from the records under `output/assets/`.

## ComfyUI checkout

`pnpm run content:setup-comfy` clones into `content-pipeline/.comfyui`:

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI), including native TRELLIS.2 and Pixal3D nodes
- [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo) (`GemmaAPITextEncode`, image-to-video, IC-LoRA guides)
- [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)
- [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)
- [ComfyUI-Video-Depth-Anything](https://github.com/yuvraj108c/ComfyUI-Video-Depth-Anything) pinned at `a0db08e63d1ea571601c45cde4aaee0acdd0544d` (Python deps installed into the Comfy venv)

Local nodes:

- `content:setup-comfy` copies `content-pipeline/comfy_nodes/reelshort_ltx` — accept Gemma API embeddings that are already projected
- `content:comfy` copies `content-pipeline/comfy_nodes/qwen_backdrop.py` — `TextEncodeQwenBackdrop`, a four-image Qwen edit encoder used when the empty-space plate is attached

`pnpm run content:asset-deps` installs trimesh, moderngl, transformers, timm, and einops into that same virtualenv. trimesh reads and writes glTF. moderngl draws previs. The rest load Florence-2. Do not install `content-pipeline/requirements.txt` into that virtualenv.

## Stage viewer

`pnpm run view -- --show <id>` serves `content-pipeline/viewer` at `http://127.0.0.1:5174`. It is Vite and Three.js. It loads location meshes, vertex-colored character meshes, and scene cameras as glTF Y-up. Orbit is the god camera. Scene camera locks to the authored lens and plays the timeline.

## Resolutions

| Stage | Size | Why |
| --- | --- | --- |
| Plates and mesh conditioning | 1024 square | TRELLIS.2 / Pixal3D shape cascade |
| Previs and Qwen stills | 768×1360 | Vertical, near 9:16 |
| LTX clips / Depth Anything control | 448×768 | Union IC-LoRA `reference_downscale_factor=2` needs even latent dims; 448×800 is incompatible |
