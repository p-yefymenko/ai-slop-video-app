# Content pipeline stack

A show is one authored JSON file. The pipeline turns that file into a vertical episode on a local GPU. Geometry, blocking, and cameras are deterministic. Image and video models only paint and animate what the blockout already decided.

Commands live in the root `package.json` (`content:*` and `view`). Python runs through those scripts. ComfyUI is the only model runtime, at `http://127.0.0.1:8188`.

## Machine

Generation is offline, on the GPU machine (an RTX 5070 Ti, 16GB). CUDA 12.8 PyTorch is installed into the ComfyUI virtualenv so Blackwell cards work. Qwen, the mesh models, and LTX are never resident together: each stage unloads idle weights before the next one starts.

Text for LTX is encoded by the LTX Gemma API (`LTXV_API_KEY` in `content-pipeline/.env`). That call is free. Video denoising stays local. The key is injected into the graph at run time and is not stored in the workflow JSON.

## What is authored

| Path | Role |
| --- | --- |
| `content-pipeline/shows/<id>/script.json` | The show. `ShowScript` in `packages/shared/src/script.ts`. Folder name matches `id`. A character has `generalDescription` (visible from all sides) and optional `frontalDescription` (front-only details). |
| `content-pipeline/prompts.json` | Shared Qwen and LTX prompt templates. Renderer config, not show content. |
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
    ├─ content:frames      Qwen-Image-Edit-2511          portraits + scene stills
    └─ content:generate    LTX-2.3 distilled-1.1         scene clips + episode cut
```

Existing outputs are skipped. One scene is rebuilt with `--force`. After a structural script rewrite, `content:archive` moves that show’s generated files aside and keeps the reviewed character portraits.

### 1. Plates — Qwen-Image-Edit-2511

`content:plates` draws and stops. ComfyUI must be running.

An empty location is one 1024×1024 picture of the place, prompted as a movie set (`output/plates/<show>/<locationId>/plate.png`). A location with people is one isolated object per landmark. A character is one full-body plate from `generalDescription` plus `frontalDescription` when present. Landmarks that share appearance and size are drawn once.

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

Guides written for the start frame, and for the end frame when blocking or the camera actually changes:

| Guide | What it is |
| --- | --- |
| `depth` | Camera-forward depth. Near surfaces are bright. Empty space is black. Character meshes stay in this map, so position and occlusion are fixed. |
| `clothes` | Those character meshes from this camera, the place cut away. Cloth, skin, and hair are each one flat color taken from the plate. A small stain takes the color around it. |
| `edges` | Outlines where depth jumps or a surface meets empty space. People are included when they are in frame. |
| `pose` | OpenPose skeleton, used only when the clothes cutout is missing. |
| `backdrop` | Empty space from this camera in flat sky, ground, and surround color. Landmarks, props, and people are black. The open floor keeps the ground color. |
| `faces` | Head masks for people whose face points at the camera, plus a JSON list of those ids. |
| `normal` | Surface normals. Written for review. |

### 5. Scene stills — Qwen-Image-Edit-2511

`content:frames` talks only to the ComfyUI HTTP API. It generates character portraits first, then each scene still.

Portraits use `workflows/qwen_image_edit.json`: same Lightning settings, 768×1360, from `generalDescription` plus `frontalDescription` when present, on a blank canvas. The template keeps a plain shirt and no costume. They land in `output/frames/<show>/characters/<id>.png`.

Scene stills use `workflows/qwen_image_edit_spatial.json` at 768×1360, 4 steps, CFG 1, AuraFlow shift 3.1, denoise 1.0. The sampler starts from noise. The shaded clay frame is not an input. CFG 1 has no negative channel, so the still prompt does not use “do not” sentences. ControlNet is not used. Surface normals are written for review and are not sent to Qwen. Clothes and the empty-space plate are never reference latents in the same pass.

Picture order:

- **Shot with a clothes cutout.** Depth, then the clothes cutout, then edges. All three are reference latents on `TextEncodeQwenImageEditPlus`, so garment color is copied and the outlines are kept. The empty-space plate is not attached: it is black where the person stands, and a reference latent would paint that black over the cloth.
- **Shot with people and no clothes cutout.** Depth, pose, edges.
- **Empty shot.** Depth and edges, plus the empty-space plate. The encoder switches to the local `TextEncodeQwenBackdrop` node so the plate is a fourth picture.

Every shot uses one prompt skeleton, filled by `still_prompt` in `generate_batch.py` and the people builder in `still_people.py`. `spatialBlockout` in `prompts.json` is `{stillPrompt}`. Order: the film-frame line, a legend for each attached picture, one keep sentence (“Keep the shape, position, and occlusion from the pictures, and each person's flat colors, lit by the scene's light.”), the people sentence when someone is visible, the visible landmark appearances, then “Behind and around:” and the sky, ground, and surround this camera shows. An empty slot is left out, including “No people.” The still does not read `scene.imagePrompt`. Character names and ids are not sent to Qwen. The log may keep `characterId`.

The people sentence is built from `generalDescription` and blocking geometry: depth order, screen side, and occlusion. Visible people are counted in a number word. Each visible person is `{position}: {generalDescription}`. `frontalDescription` is appended to that phrase only when the faces guide lists them as facing the camera, their head mask has pixels, and their pixel height is at least `frontalMinPixelHeight` in `prompts.json`. Figures behind the nearest person by a depth gap stand in a row, left to right, each with their own line. Each other person is named by screen side; the nearest by camera depth is also in the foreground. A figure whose visible fraction is below the cutoff is partly hidden. Character names and ids are not sent.

A landmark mesh is named when at least 70% of its on-screen surface is the surface this camera sees. Trailing plate instructions such as “a single object” or “no walls” are dropped from landmark appearance and from the backdrop sky, ground, and surround lines. The script text itself is not edited.

Prompt logs and the attached pictures are written to `output/frames/<show>/<episode>/inputs/scene_XX_start/` (and `_end` when that frame exists). The episode folder itself keeps `scene_XX_start.png` and, when the timeline or camera changes, `scene_XX_end.png`. The end still is generated from the end guides, because Qwen-Image-Edit keeps the camera of whatever picture it is given.

Identity face painting exists in the spatial graph and is off (`FACE_PASS = False`). LTX never receives a character portrait.

### 6. Clips — LTX-2.3 distilled-1.1

`content:generate` animates each reviewed still. Graph: `workflows/ltx_gemma_api.json`.

| Setting | Value |
| --- | --- |
| Weights | `ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf` |
| Video VAE / audio VAE | Matching LTX-2.3 distilled VAEs |
| Text | `GemmaAPITextEncode` against a tiny safetensors stub that carries the API `model_id`. Local node `LTXAVUseProcessedAPIEmbeds` marks those embeddings as already projected to 6144 so they match the Q4 GGUF. |
| Size | 448×800, near 9:16, divisible by 32 |
| Rate | 24 fps. Length is `8n+1` frames for the scene duration. |
| Sampler | Euler, 8 steps, LTX shift (max 2.05, base 0.95) |
| Start frame | `LTXVImgToVideo` at strength 0.7 |
| End frame | `LTXVAddGuide` at strength 0.85 on the last frame, only when a generative scene’s blocking or camera changes. `LTXVCropGuides` strips the guide tokens before decode. |
| Dialogue | When `speakerId` is set, `MultimodalGuider` raises joint audio and video guidance (`modality_scale` 3, cross-attention on). Silent scenes keep `BasicGuider`. |
| Prompt | `sceneVideo`: continue from the still. `videoPrompt` is the line and the performance. Camera and blocking come from the timeline and the start and end frames. |
| Mux | Video Helper Suite, H.264, CRF 19, with the decoded LTX audio. |

Clips land in `output/generate/<show>/<episode>/scene_XX.mp4`. ffmpeg concatenates them into `episode.mp4` in that folder. A partial clip render leaves `episode.mp4` alone.

`content:upload` pushes the finished MP4s and thumbnails to Cloudflare R2 and registers episodes on the Worker. That step is distribution, not generation.

## Models on disk

`pnpm run content:models` downloads these into `content-pipeline/.comfyui/models/` and skips files already present.

| Stack | Files |
| --- | --- |
| LTX-2.3 | Q4_K_M GGUF (~14GB), video VAE, audio VAE, distilled API stub |
| Qwen stills | `qwen-image-edit-2511-Q4_K_M.gguf` (~13GB), `qwen_2.5_vl_7b_fp8_scaled` text encoder, `qwen_image_vae`, 4-step Lightning LoRA |
| TRELLIS.2 | int8 UNet (~5GB), DINOv3 ViT-L, shape VAE, BiRefNet |
| Pixal3D | int8 UNet (~5.2GB), DINOv3 with NAF, texture VAE, MoGe-2. Shares the TRELLIS shape VAE. |

Qwen-Image-Edit-2511 is Apache-2.0. TRELLIS.2 and Pixal3D are MIT. `content:assets -- --credits` rewrites `docs/CREDITS.md` from the records under `output/assets/`.

## ComfyUI checkout

`pnpm run content:setup-comfy` clones into `content-pipeline/.comfyui`:

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI), including native TRELLIS.2 and Pixal3D nodes
- [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo) (`GemmaAPITextEncode`, image-to-video, guides)
- [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)
- [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)

Two local nodes are copied in:

- `content-pipeline/comfy_nodes/reelshort_ltx` — accept Gemma API embeddings that are already projected
- `content-pipeline/comfy_nodes/qwen_backdrop.py` — `TextEncodeQwenBackdrop`, a four-image Qwen edit encoder used when the empty-space plate is attached

`pnpm run content:asset-deps` installs trimesh, moderngl, transformers, timm, and einops into that same virtualenv. trimesh reads and writes glTF. moderngl draws previs. The rest load Florence-2. Do not install `content-pipeline/requirements.txt` into that virtualenv.

## Stage viewer

`pnpm run view -- --show <id>` serves `content-pipeline/viewer` at `http://127.0.0.1:5174`. It is Vite and Three.js. It loads location meshes, vertex-colored character meshes, and scene cameras as glTF Y-up. Orbit is the god camera. Scene camera locks to the authored lens and plays the timeline.

## Resolutions

| Stage | Size | Why |
| --- | --- | --- |
| Plates and mesh conditioning | 1024 square | TRELLIS.2 / Pixal3D shape cascade |
| Previs and Qwen stills | 768×1360 | Vertical, near 9:16 |
| LTX clips | 448×800 | Same aspect, divisible by 32, small enough for 16GB |
