# Content pipeline stack

A show is one authored JSON file. The pipeline turns that file into a vertical episode on a local GPU. The script describes a physical world: every character, landmark, and prop is a generated mesh, placed by the timeline. Previs renders that world from each shot's camera, and the script is checked against the render. Image and video models only paint and animate what the render shows.

Commands live in the root `package.json` (`content:*` and `view`). Python runs through those scripts. Qwen, TRELLIS.2, Pixal3D, and LTX run through ComfyUI at `http://127.0.0.1:8188`.

Depth-control research notes (not the live path): [docs/ltx-depth-control-spike.md](docs/ltx-depth-control-spike.md), [docs/ltx-depth-gate.md](docs/ltx-depth-gate.md).

## Machine

Generation is offline, on the GPU machine (an RTX 5070 Ti, 16GB). CUDA 12.8 PyTorch is installed into the ComfyUI virtualenv so Blackwell cards work. Qwen, the mesh models, Depth Anything, and LTX are never resident together: each stage unloads idle weights before the next one starts. LTX peak VRAM on the depth-control path is about **15.8 GB**.

Text for LTX is encoded by the LTX Gemma API (`LTXV_API_KEY` in `content-pipeline/.env`). That call is free. Video denoising stays local. The key is injected into the graph at run time and is not stored in the workflow JSON.

`pnpm run content:comfy` starts ComfyUI with listen `127.0.0.1:8188` only. Set `COMFY_ARGS` to replace those defaults (for example `COMFY_ARGS=--reserve-vram 2`). On this machine `--reserve-vram 2` reduced LTX shared-memory spill but slowed Qwen stills, so it is **not** the default.

## What is authored

| Path | Role |
| --- | --- |
| `content-pipeline/shows/<id>/script.json` | The show. `ShowScript` in `packages/shared/src/script.ts`. Folder name matches `id`. |
| `content-pipeline/prompts.json` | Shared Qwen and LTX templates and renderer numbers: `visibleMinShare` and one minimum size per check: `onScreenMinPixels` (on screen: needs a performance, can move, can make a sound), `speakerFaceMinPixels` (a talking shot), `rowDepthRatio`, `endStill`, `ltxStartStrength`, `ltxIcLoRAStrength`, `controlDepth`, `depthAnything`, `sceneRenderOptions`, the `scene*` LTX templates, `soundLint`, and `episodeLoudnessTargetLufs`. Not show content. |
| `content-pipeline/workflows/*.json` | ComfyUI graphs the batch scripts fill in and post to `/prompt`. |

### The world

Schema space is meters, X right, Y forward, Z up. glTF is Y-up, forward −Z. `content-pipeline/scripts/coords.py` is the only converter.

- **Entities.** A character (`body`, `attributes`, `heightMeters`), a landmark (`position`, `size`, `appearance`, per location), and a prop (`appearance`, `size`) are all the same thing: one generated, plate-colored mesh, fitted into its size, with named regions. A character's regions are body parts; anything else has one region, `whole`. A whole place seen from far away (a hanging citadel) is one landmark the size of the location.
- **Location look.** Each location has `look.materials` and `look.light`, said once. Every landmark's plate adds `materials` (`landmarkMaterials`), so the objects of one place share one material language; every pass of a still there adds `light` (`stillLight`), so things drawn alone are lit alike. A landmark's `appearance` gives shape and parts.
- **Effects.** Fire, smoke, steam, sparks, and glow are never part of a mesh: in a depth guide they would be a solid shape, and the still and the video would keep them solid. Any entity may list `effects`, each a box (`size`, `offset` from the entity's base, turned with it) and its own `appearance`. Validation rejects effect words (flame, fire, smoke, glow, …) in an `appearance`, `body`, `materials`, or attribute. An effect is never rendered into a guide or collided with; for the checks its box hides what is behind it. In a still it is added onto its host's own drawing on black, guided by `look.png` (the plate with the effects added, made once, so the same fire looks the same in every shot); only what that pass adds is light, and it is added to the shot as light.
- **Shots.** Every scene declares `shot`: what it shows, as one of a closed set of types (`single`, `group`, `overShoulder`, `insert`, `establishing`, `action`) with its subjects, `size` (ecu … wide), `angle` (eye, low, high), `side` (front, left, right), and `move` (static, pushIn, pullOut). For every shot of people the camera is computed from the blocking (`shots.py`, when the show is loaded): a band of the subject's height fills the frame, seen along their gaze, so a shot frames its subject by construction and nobody types camera coordinates. Only `establishing` (required) and `action` (optional) set their own `camera`. Validation enforces each type's subjects and sizes.
- **Timeline.** Every character and every prop has a track in every episode, starting at 0. A keyframe puts it in a location at a position, or off stage (`locationId: null`). A prop is held (`heldByCharacterId`, `heldInHand`) or placed. A character's body faces `lookAtId` when set, else `bodyYawDegrees`.
- **Presence.** Who is in a shot is never written down. `world.entities_at` lists everything whose track puts it in the scene's location at that moment. Nobody despawns.
- **Visibility.** `observe.observe` renders one moment and measures, per entity and region, `visible` (pixels that are the front surface) and `extent` (pixels the region would cover with nothing in front, not even the same body). There are two tests for two jobs. **In shot** (`observe.in_shot`): any pixel in the frame. Everything in shot is drawn into the still, with no size cutoff, so no words ever have to find their own place in a picture. **Visible** (`observe.visible`): `visible ≥ minimum` and `visible ≥ visibleMinShare × extent`, where each check has its own minimum: on screen at all (`onScreenMinPixels`), large enough to lip-sync (`speakerFaceMinPixels`). A face seen from behind covers its full extent but shows a sliver, so it is not visible. Only the checks below use it.

`pnpm run content:validate` checks every show script's shape and references (TypeScript, zod). Every `content:*` stage runs it first and stops on an error. It does not decide what is visible; previs does. A high authored camera travel+rotation score (threshold **20**) and sound lint print warnings and do not fail.

## Stages

`content:render` runs meshes, previs, stills, and clips in that order. It does not draw plates and it does not pause for review. Plates are a separate step so those pictures can be kept or redrawn before meshing.

```text
script.json
    │
    ├─ content:plates      Qwen-Image-Edit-2511          one plate per landmark, prop, character
    ├─ content:assets      TRELLIS.2 / Pixal3D           one mesh per plate
    ├─ content:previs      moderngl                      render, guides, observations, script checks
    ├─ content:frames      Qwen-Image-Edit-2511          scene start stills
    └─ content:generate    Depth Anything → LTX-2.3      control depth + scene clips + episode cut
```

Plates, meshes, stills, and clips skip files already on disk. One selected still or clip is rebuilt with `--force`. `content:previs` always rewrites the playblast and guides. After a structural script rewrite, `content:archive` moves that show's generated files aside.

### 1. Plates — Qwen-Image-Edit-2511

`content:plates` draws and stops. ComfyUI must be running. A landmark or prop is one isolated object (`landmarkPlate`). A character is one full-body plate from `body` plus every attribute (`characterPlate`). Landmarks and props that share appearance and size are drawn once. An entity with `effects` also gets `look.png`: its reviewed plate with the effects added (`lookPlate`). The mesh is built from `plate.png` only. Output: `output/plates/<show>/<location>/<landmark>/`, `.../props/<prop>/`, `.../characters/<character>/`.

Graph: `workflows/qwen_asset_plate.json`. The 4-step Lightning LoRA, CFG 1, AuraFlow shift 3.1, denoise 1.0.

### 2. Meshes — TRELLIS.2 and Pixal3D

`content:assets` turns each reviewed plate into one mesh. A missing plate stops the build. Every mesh keeps the plate's color on its vertices (the TRELLIS.2 texture stage, painted onto the mesh).

| Subject | Model | Why |
| --- | --- | --- |
| Landmark, prop | TRELLIS.2 shape + texture | Built in its own upright frame. An object plate is drawn from slightly above; pixel alignment would tilt the object by that angle. Fitted uniformly into `size`, base center at its position. |
| Character | Pixal3D shape + texture | The plate is a frontal, eye-level photo, so pixel alignment makes the front match it. Front turned to schema +Y, stood upright (`stand_upright`: Pixal3D builds the body in its estimated camera's frame, which looks slightly down, so the raw mesh leans by that angle; the body's long axis is turned to vertical, yaw unchanged), fitted to `heightMeters`, then the body cleanup (thin side sheets dropped, small holes capped, a proportion warning). |

Both remove the background with BiRefNet, condition with DINOv3, and run the shape cascade at 1024. ComfyUI's DecimateMesh caps the mesh (default 10,000,000 triangles).

### 3. Previs — no diffusion model

`content:previs` needs every mesh the script names and stops with the list of missing ones. Per shot it samples the timeline at 8 fps. Every sample is measured for the checks; only the start and end samples also read back depth and plate color and label the empty space, because only they become guides. Each sample is one GPU draw of every entity into clay, plate color, entity/region ids, and depth (`render.py`), plus a coverage draw for each region large enough to matter. A floor is drawn as ground when the camera is inside the location's bounds.

Every scene is rendered even when one fails; the stage ends with the full list of mismatches. The checks (`checks.py`), all with the one visibility measurement:

- anyone on screen at any sample has a performance (`parts: []` for stillness), and every performance belongs to someone on screen;
- every part a performance moves is on screen at some moment of the shot;
- an `expression` is only on a face visible in the start frame (the speaker must have one; zod checks that);
- a performance's `target` (what `{target}` in its action points at) is visible in the start frame, so the clip prompt can name it by where the frame shows it;
- the speaker's face shows at least `speakerFaceMinPixels` (22,000 px of the 768×1360 still, about an eighth of the frame height: a medium close-up. A smaller talking face leaves the 448×768 video a few pixels of mouth, which smear), and every `sound.events[].source` is on screen in the start frame;
- the camera is never inside a mesh.

Output is 768×1360. Per scene in `output/previs/<show>/<episode>/scene_XX/`: `blockout.mp4` (8 fps), `start.png`, `end.png`, `scene.json` (every entity's mesh and pose per sample, for the viewer), and `guides/`.

| Guide (`start_*` and `end_*`) | What it is |
| --- | --- |
| `depth` | Camera-forward depth. Near is bright, sky black. The ground runs on past the floor to the horizon at its own depth, so the horizon is where the camera puts it and the ground past the floor is never empty space. The shading spans the meshes' depths; farther ground is the darkest surface. |
| `edges` | Where depth jumps or a surface meets empty space. |
| `color` | Every entity in flat plate colors, empty space in the location's flat sky, ground, and surround colors. |
| `drawn_<id>_depth`, `_edges`, `_color` | One person, prop, or landmark in shot, drawn alone: the shot camera magnified onto a window around the whole entity, so pose and turn are the shot's and it is always drawn whole, even when the frame edge cuts it (the window may run past the frame). Only an entity bigger than the frame gets a frame-sized window over the part in view, so a drawing is only ever scaled down into the shot. |
| `drawn_<id>_shown` | Where the shot shows it: its rendered silhouette, grown 4 px and softened, minus anything closer (the floor covers nothing). A drawing can land only here. |
| `drawn_<id>_effects` | Only for an entity with effects in shot: where the shot shows its effect boxes, grown 8 px and softened, minus anything solid in front (its own near rim too). Their light is added only here. |
| `drawn_<id>_whole_depth`, `_edges`, `_color` | Only when an entity's effects are in shot but the entity is not (flames rising into the frame from a bowl below it): the whole entity and its effects in an unclamped window, the base its effects are added onto (`wholeCrop` in the observation). An entity in shot gets its effects on its own sharp drawing instead. A whole window over `WHOLE_MAX_SCALE` (4) frames tall gives no usable base, and the effects are not shown (nor named to LTX). |
| `start_people`, `clip_people` | Where the 3D scene has people: in the start frame grown 16 px (for the still), and in any sample of the shot grown 96 px (acting room, for the clip). The output checks compare against these. |
| `observation.json` | Per entity, region pixels `[visible, extent]`, screen x, depth, each drawn entity's crop window, and `band`: the slice of its height inside that window (0 at its feet). |
| `clay_24fps.mp4` | Clay at LTX size and rate, only for depth-controlled shots. Input to Depth Anything. |
| `control_depth.mp4` | Written by `content:generate`: Depth Anything on `clay_24fps.mp4`. |

### 4. Scene stills — Qwen-Image-Edit-2511

`content:frames` runs previs for each scene first, unless `guides/previs_record.json` shows the guides were made from the same script, meshes, settings, and previs code, and stops on a mismatch. Each still is drawn in passes with `workflows/qwen_image_edit_spatial.json` (768×1360, 4 steps, CFG 1, AuraFlow shift 3.1, denoise 1.0, from noise, the four-picture `TextEncodeQwenBackdrop` encoder):

1. **One pass per person, prop, and landmark in shot**, however little of it shows. Pictures: its depth, colors, and edges (the shot camera magnified onto its window), plus its plate as the appearance reference, cut to the same slice of its height the window shows (`plate_slice` from `band`: a face close-up gets the plate's head and shoulders, not a small figure in a whole-body picture). Words: only its own, plus the location's `light`. A person gets `body`, every attribute, and their expression; a prop or landmark its appearance. Solid parts only. Nothing borrows another's description, and the drawing has at least the resolution the shot needs. The same pass also writes the drawing's soft outline: BiRefNet on the result, with everything outside the mesh's own outline in its depth guide (grown `REGION_GROW_PIXELS`, 16) painted over in the drawing's own background color first (its median outside that outline, so no new edge appears that could be taken for part of the subject). Background removal keeps the most prominent subject, and the model may draw more than the mesh asks for (a whole person beside the arm that is in shot); only the part where the mesh is can be the subject. An entity with effects in shot gets a second pass: its own drawing cut out onto black (drawn again whole, in its whole window, when only the effect is in shot), `look.png`, and `effectDrawn` with only its effects' words: keep the picture, add only the effects.
2. **Composite.** Each drawing is scaled down into its window and blended into the `color` guide by its own soft outline times `drawn_<id>_shown`. Nothing drawn can land where the 3D scene has no such thing, and the soft cut leaves no stair-step or background fringe. Drawings are laid farthest first (`paint_order`). `coverage.json` records how much of where the shot shows each entity its drawing filled.
3. **The shot, inpainted.** Pictures: depth, the composite, edges. The pasted pixels are kept exactly; only the rest of the frame (and a 3-pixel band at each pasted edge, so it sits in the shot) is drawn. Words: `stillOpening`, the legend, that everything standing in the shot is already painted, "Behind and around:" with the sky, ground, and surround in shot, and the location's `light`. No object is named, so no description can land on the wrong one.
4. **Effects, as light.** After the shot pass, each effect is added to the still as light (screen blend) inside `drawn_<id>_effects` (`add_light`). The light is only what the effect pass added: its picture minus its base, less `LIGHT_NOISE`. Whatever solid it redrew (stones, the bowl) cancels out, and over the entity's own surface (its base's matte) only flame-bright light counts (`FLAME_LOW`–`FLAME_HIGH`, 0.7–0.9), so a surface the pass re-lit never turns to glass. Black adds nothing, and a flame brightens what is behind it, so nothing is cut out and nothing has to be guessed about a background. The LTX prompt adds the effects in the start frame (`sceneEffects`), and the depth video has nothing solid there, so the video animates them freely. Light suits fire, sparks, glow, steam, and mist; a dark smoke would need its own pass.

Trailing plate instructions such as "a single object" are dropped. Names and ids are never sent. `output/frames/<show>/<episode>/inputs/scene_XX_start/` keeps every pass's prompt and pictures, each drawn picture, `composite.png`, and `draw_mask.png`.

### Output checks — `verify.py`

Previs checks the script before anything is drawn; `verify.py` checks every still and clip after it is made, against the same 3D scene. A model's mistake is random, so a failed output is made again with a new seed, up to `verifyAttempts` (3) tries. The last try is kept either way, and `inputs/scene_XX[_start]/verify.json` records the tries and any problems left.

- **Nobody invented.** Mask R-CNN (torchvision, COCO) finds people (a detection mostly inside a bigger one is the same person found again), in a still on the solid shot before any effect light (`inputs/.../solid.png`: through fire a detector sees figures in the flames); a person mostly (over half) outside `start_people` in a still, or `clip_people` in a clip frame (sampled 3 times a second), was invented. A clip frame may not have more people than the most the camera sees at once (`people` in the previs record). In a clip, a detection mostly inside the shot's effect areas (its `drawn_<id>_effects`, grown 48 px) is not judged: a detector sees figures in flames, and fire is only light.
- **Every face its own.** YuNet finds the face inside each character's `drawn_<id>_shown` in a still, and SFace compares it with the face on their plate. It must reach `FACE_SAME_PERSON` (0.30) and match its own plate better than any other character's. Faces under 48 px high are not judged.

- **Every drawing where its mesh is.** Each drawing must fill at least `DRAWN_MIN_COVERAGE` (85%) of where the shot shows its entity (`coverage.json`).

The checks do not judge hair color, wardrobe colors, or lighting.

### 5. Clips — LTX-2.3 distilled-1.1 + Depth Anything control

`content:generate` animates each reviewed start still in two stages. Stage 1 generates the motion at **448×768** (`clip_spec.py`), **24 fps**, length **`8n+1`** for the scene duration. Stage 2 (`ltxSpatialUpscale`, `"x2"`; `null` turns it off) enlarges that latent with LTX's spatial upscaler (`ltx-2.3-spatial-upscaler-x2-1.1`), re-anchors the start still at **896×1536**, and refines it in the distilled model's short schedule (`_REFINE_SIGMAS`) on the plain model, so mouths, eyes, and skin are generated at the larger size while stage 1's motion stays. In a depth-controlled graph the guide frames are cropped off before stage 2. Clips are decoded tiled. It nearly fills a 16 GB card (it spills into shared memory) and takes about 2.5× as long as stage 1 alone.

**Routing**

| Scene | Graph | Control |
| --- | --- | --- |
| The clay shows the shot as it plays (`world.depth_control_fits`, `depthControl` in the previs record): the camera moves, nobody in shot moves or turns, every performance in shot moves only hair, face, eyes, or neck, and nobody speaks | `workflows/ltx_gemma_api_depth.json` | Start still + Depth Anything control video through the union IC-LoRA |
| Any other shot | `workflows/ltx_gemma_api.json` | Start still only (plain image-to-video). Meshes are rigid statues: the clay of a walk, a turn, a shove, or a talking face is a statue sliding, and depth control makes LTX copy exactly that. Such a shot is animated from its still and its words. |
| Override | `sceneRenderOptions["<show>/<episode>/<scene>"] = {"depthControl": false}` | Forces the plain graph; logs `no depth control (disabled by override)` |

**Depth Anything pass (spatial scenes, before LTX)**

1. Load `guides/clay_24fps.mp4`.
2. Run Video Depth Anything (`video_depth_anything_vits.pth`, small) via ComfyUI-Video-Depth-Anything.
3. Write `guides/control_depth.mp4` (H.264 + silent AAC).
4. Unload Comfy models (`/free`) before queuing LTX.

| Setting | Value |
| --- | --- |
| Weights | `ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf` |
| IC-LoRA (spatial) | `ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors` via `LTXICLoRALoaderModelOnly` |
| Video VAE / audio VAE | Matching LTX-2.3 distilled VAEs |
| Text | `GemmaAPITextEncode` against a tiny safetensors stub that carries the API `model_id`. Local node `LTXAVUseProcessedAPIEmbeds` marks those embeddings as already projected to 6144 so they match the Q4 GGUF. |
| Size / rate | 448×768 stage 1, 896×1536 after stage 2, 24 fps, `8n+1` frames |
| Sampler | Euler, 8 steps, LTX shift (max 2.05, base 0.95) |
| Start frame | `LTXVImgToVideo` at `ltxStartStrength` (default **0.7**) |
| IC-LoRA strength | `ltxIcLoRAStrength` (default **1.0**) on spatial/depth graphs |
| End frame guide | Not used. No `LTXVAddGuide` end still. |
| Dialogue | When `speakerId` is set on the plain graph, or on the depth-dialogue graph, `MultimodalGuider` raises joint audio and video guidance (`modality_scale` 3, cross-attention on). Silent scenes keep `BasicGuider`. |
| Prompt | `describe.ltx_prompt`: `sceneVideo` (continue from the still; then `sceneDialogue` for the speaker, `scenePerformance` and `scenePerformanceExpression` per performance, `sceneMotion`, and `sceneCameraLocked` when the camera pose does not change). People are named by `scenePerson`: where the start frame's observation places them plus their `body`, never a character id, because LTX only sees the frame. Then labeled sound sentences from `location.soundscape` + `scene.sound` using `sceneSound*` templates in `prompts.json` (label → bed → space → events → music). Distilled CFG 1: positive descriptions only; no negative-prompt reliance. `GemmaAPITextEncode` runs with `enhance_prompt: false` (encodes the authored text; does not rewrite it). |
| Log | Final LTX prompt, graph name, seed, and `enhance_prompt` land in `output/generate/<show>/<episode>/inputs/scene_XX/log.json`. |
| Mux | Video Helper Suite, H.264, CRF 19, with the decoded LTX audio. |

Clips land in `output/generate/<show>/<episode>/scene_XX.mp4`. ffmpeg loudness-normalizes each clip to `episodeLoudnessTargetLufs` from `prompts.json` (default **-16** LUFS), then concatenates them into `episode.mp4` in that folder. `concat_videos` **aborts** if clip width, height, or fps differ (it does not auto-regenerate). A partial clip render leaves `episode.mp4` alone.

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

`pnpm run content:asset-deps` installs trimesh and moderngl into that same virtualenv. trimesh reads and writes glTF. moderngl draws previs. Do not install `content-pipeline/requirements.txt` into that virtualenv.

## Stage viewer

`pnpm run view -- --show <id>` serves `content-pipeline/viewer` at `http://127.0.0.1:5174`. It is Vite and Three.js. It loads every landmark, prop, and character mesh. A scene replays previs's `scene.json`: each entity's pose per sample and the camera, glTF Y-up. Orbit is the god camera. Scene camera locks to the authored lens and plays the timeline.

## Resolutions

| Stage | Size | Why |
| --- | --- | --- |
| Plates and mesh conditioning | 1024 square | TRELLIS.2 / Pixal3D shape cascade |
| Previs and Qwen stills | 768×1360 | Vertical, near 9:16 |
| LTX clips / Depth Anything control | 448×768 | Union IC-LoRA `reference_downscale_factor=2` needs even latent dims; 448×800 is incompatible |
