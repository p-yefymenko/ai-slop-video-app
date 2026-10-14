# Content pipeline stack

A show is one authored JSON file. The pipeline turns that file into a vertical episode on a local GPU. The script describes a physical world: every character, landmark, and prop is a generated mesh, placed by the timeline. Previs renders that world from each shot's camera, and the script is checked against the render. Image and video models only paint and animate what the render shows.

Commands live in the root `package.json` (`content:*` and `view`). Python runs through those scripts. Qwen-Image, TRELLIS.2, Pixal3D, Wan 2.2, and RIFE run through ComfyUI at `http://127.0.0.1:8188`; Chatterbox and Qwen3-TTS VoiceDesign each run in their own env (`content-pipeline/.voice`, `content-pipeline/.voicedesign`).

The video model is given as little as possible: the more it generates on its own, the more natural the take. The 3D scene supplies only what a shot has proved it needs (today: a depth video for the framing, and a reference photo for each face the shot shows), and each such input says when it applies.

## Machine

Generation is offline, on the GPU machine (an RTX 5070 Ti, 16 GB; 34 GB RAM). CUDA 12.8 PyTorch is installed into the ComfyUI virtualenv so Blackwell cards work. Each stage unloads idle weights (`/free`) before the next one starts; the voice step frees ComfyUI's VRAM before Chatterbox loads. The Wan 14B experts do not fit the card at Q8 and are offloaded to RAM, about 30 s a step at 480p.

`pnpm run content:comfy` starts ComfyUI on `127.0.0.1:8188` with `--reserve-vram 1.5`, which keeps it from spilling into shared memory (many times slower). `COMFY_ARGS` replaces those arguments entirely.

## What is authored

| Path | Role |
| --- | --- |
| `content-pipeline/shows/<id>/script.json` | The show. `ShowScript` in `packages/shared/src/script.ts`. Folder name matches `id`. |
| `content-pipeline/prompts.json` | Every word template the models get (object pictures, reference photos, the Wan prompt pieces, the negatives) and the episode loudness (`episodeLoudnessTargetLufs`). Not show content. |
| `content-pipeline/scripts/wan.py` | The Wan chain's graphs and every model setting (steps, CFG, shift, strength, sizes, frame rates), one named constant each, the same for every scene. |

### The world

Schema space is meters, X right, Y forward, Z up. glTF is Y-up, forward −Z. `content-pipeline/scripts/coords.py` is the only converter.

- **Entities.** A character (`body`, `attributes`, `heightMeters`), a landmark (`position`, `size`, `appearance`, per location), and a prop (`appearance`, `size`) are all the same thing: one generated mesh colored by its picture, fitted into its size. A whole place seen from far away (a hanging citadel) is one landmark the size of the location.
- **Location look.** Each location has `look.materials` and `look.light`, said once. Every landmark's picture adds `materials` (`landmarkMaterials`), so the objects of one place share one material language; every Wan prompt there ends with `light`.
- **Effects.** Fire, smoke, steam, sparks, and glow are never part of a mesh: in a depth video they would be a solid shape, and the video would keep them solid. Any entity may list `effects`, each a box (`size`, `offset` from the entity's base, turned with it) and its own `appearance`. Effect words never go in an `appearance`, `body`, `materials`, or attribute: a mesh is built from those words. An effect is never rendered into the depth video or collided with. The Wan prompt names the effects the start frame shows, so the video makes them.
- **Shots.** Every scene declares `shot`: what it shows, as one of a closed set of types (`single`, `group`, `overShoulder`, `insert`, `establishing`, `action`) with its subjects, `size` (ecu … wide), `angle` (eye, low, high), `side` (front, left, right), and `move` (static, pushIn, pullOut, tiltUp: the same place, turning from the feet up to the framing). An own camera that moves says how in `cameraMotion`, the words the clip prompt uses. For every shot of people the camera is computed from the blocking (`shots.py`, when the show is loaded): the top of the subject's height down by the shot size fills the frame, seen along their gaze, so a shot frames its subject by construction and nobody types camera coordinates. Only `insert` and `establishing` (required) and `action` (optional) set their own `camera`. Validation enforces each type's subjects and sizes.
- **Timeline.** Every character and every prop has a track in every episode, starting at 0. A keyframe puts it in a location at a position, or off stage (`locationId: null`). A prop is placed; holding one is not supported (no robust hand position). A character's body faces `lookAtId` when set, else `bodyYawDegrees`.
- **Presence.** Who is in a shot is never written down. `world.entities_at` lists everything whose track puts it in the scene's location at that moment. Nobody despawns.
- **In shot.** `observe.observe` renders one moment and lists, per entity, its pixels in the frame, its mean screen x, and its median distance. An entity is in shot when any pixel of it is (`observe.in_shot`); there are no size or visibility thresholds anywhere.

`pnpm run content:validate` checks every show script's shape and references (TypeScript, zod). Every `content:*` stage runs it first and stops on an error. It checks structure and references only (ids, times, keyframe order, each shot type's subjects and sizes, `{target}` use); it reads no words.

## Stages

`content:render` runs meshes, previs, stills, and clips in that order. It does not draw plates and it does not pause for review. Plates are a separate step so those pictures can be kept or redrawn before meshing.

```text
script.json
    │
    ├─ content:plates      Qwen-Image-2512, Qwen3-TTS VoiceDesign one picture per landmark, prop, character; a voice per speaker
    ├─ content:assets      TRELLIS.2 / Pixal3D                   one mesh per picture
    ├─ content:previs      moderngl                              render, guides, observations, script checks
    ├─ content:frames      Wan 2.2 VACE                          scene start stills
    └─ content:generate    Chatterbox → Wan S2V or VACE → RIFE   voice, scene clips at 24 fps, episode cut
```

Plates and meshes skip files already on disk. A still, a voice track, or a clip is made again when what it was made from changed (its words, its size, its lines, the still under a talking take: each output's `inputs/<name>/log.json` records them), and is otherwise skipped; `--force` remakes one selected scene anyway. `content:previs` always rewrites the playblast and guides. After a structural script rewrite, `content:archive` moves that show's generated files aside.

### 1. Pictures and voices — Qwen-Image-2512, Qwen3-TTS VoiceDesign

`content:plates` draws and stops. ComfyUI must be running. Every entity has exactly one picture, the one its mesh is built from: Qwen-Image-2512 (Q8 GGUF, 50 steps, CFG 4, euler, AuraFlow shift 3.1; `asset_generate.picture_graph`), at a seed fixed by its words, drawn again only when its words change, and reviewed. A landmark or prop is one isolated object on light gray, 1328×1328 (`plate.png`; `objectPicture`, `objectNegative`, the location's `landmarkMaterials`); landmarks and props with the same appearance and size share one picture. A character's picture is their reference photo (`reference.png`): a full-length studio photo on pure white from `body` and every attribute, item by item, 928×1664 (`referencePhoto`, `referenceNegative`). It is also the picture the video model knows them by, so the mesh's silhouette and the look are one person. `effects` are never drawn into a picture (the video prompt names them). Each picture's words, size and seed are in `inputs/<name>/log.json`. Output: `output/plates/<show>/<location>/<landmark>/`, `.../props/<prop>/`, `.../characters/<character>/`.

**Voices.** Every character with a `voice` description also gets `voice.wav` beside the reference: Qwen3-TTS VoiceDesign (1.7B, bf16, in `content-pipeline/.voicedesign`) speaks `voiceSample` (prompts.json) in a voice made from that description, in English, at a seed fixed by it (`voice.design_voices`). It is made again when the description, the sample or the model changes, and reviewed like every picture. ComfyUI's VRAM is freed first. Every line of that character is spoken in it (stage 5).

### 2. Meshes — TRELLIS.2 and Pixal3D

`content:assets` turns each reviewed picture into one mesh. A missing or stale picture stops the build. Every mesh keeps the picture's color on its vertices (the TRELLIS.2 texture stage, painted onto the mesh).

| Subject | Model | Why |
| --- | --- | --- |
| Landmark, prop | TRELLIS.2 shape + texture | Built in its own upright frame. An object picture may be seen from slightly above; pixel alignment would tilt the object by that angle. Fitted uniformly into `size`, base center at its position. |
| Character | Pixal3D shape + texture | The reference photo is frontal and eye-level, so pixel alignment makes the front match it. Front turned to schema +Y (Pixal3D's axis convention), fitted to `heightMeters`. No cleanup: the mesh is used as generated. |

Both remove the background with BiRefNet, condition with DINOv3, and run the shape cascade at 1024. ComfyUI's DecimateMesh caps the mesh (default 10,000,000 triangles).

### 3. Previs — no diffusion model

`content:previs` needs every mesh the script names and stops with the list of missing ones. Per shot it samples the timeline at 8 fps. Each sample is one GPU draw of every entity into clay, plate color, entity ids, and depth (`render.py`). A floor is drawn as ground when the camera is inside the location's bounds.

Every scene is rendered even when one fails; the stage ends with the full list of mismatches. The checks (`checks.py`) are exact facts only:

- the camera is never inside a mesh;
- everyone with a performance is in the frame at some moment of the shot;
- a performance's `target` (what `{target}` in its action points at) is in the start frame, so the prompt can name it by where the frame shows it.

Output is 768×1360. Per scene in `output/previs/<show>/<episode>/scene_XX/`: `blockout.mp4` (8 fps), `start.png`, `end.png` (clay), `scene.json` (every entity's mesh and pose per sample, for the viewer), and `guides/`: `start_observation.json` and `end_observation.json` (per entity: pixels in the frame, screen x, distance; sky, ground, and surround pixels; the effects in the frame) and `previs_record.json` (the check errors and a digest of everything previs read, so an unchanged scene is not rendered again).

### 4. Scene stills — Wan 2.2 VACE

`content:frames` runs previs for each scene first, unless `guides/previs_record.json` shows the guides were made from the same script, meshes, settings, and previs code, and stops on a mismatch. The still is the scene's first frame: a talking take starts from it, and the stills are reviewed before any clip is made.

**Model.** Wan 2.2 VACE Fun A14B (QuantStack Q8 GGUF), its two experts as ComfyUI's template runs them: 20 steps, CFG 3.5, shift 8, euler, the high-noise expert for the first 10 steps and the low-noise expert for the rest (`wan.vace_graph`). The few-step Lightning LoRAs flattened the light and the acting, so every Wan pass runs at full steps.

**A still is the first frame of a short take** (`STILL_FRAMES`, 9 frames): a one-frame take ignores the control video and copies the reference's layout.

**Inputs** (`shot_inputs.py`), the least that does the job:

- **Depth video** (every shot), rendered from the timeline at the take's own size and 16 fps. People are flat silhouettes at their own median distance: the clay's face relief, copied, froze the face; only where they stand and how big comes from here. One fixed log scale for every shot (0.5 m brightest, 60 m darkest, gray 48–255; black only where nothing is, the sky), so a lone person far away is not stretched to white. Past every mesh the ground runs on to the horizon at its own distance. Control strength `VACE_STRENGTH` 0.6: at 1.0 the flat silhouettes are copied and the face freezes.
- **Reference sheet**: the reference photos of who the script says the shot is about (`describe.reference_people`): the shot's `subjects`, or, without subjects, everyone with a performance; never the shot's `over`. Each photo goes on as drawn (on white, by its prompt), side by side in one picture (VACE takes one). Anyone on the sheet gets a face painted in, so the shoulder in front stays off it. No sheet when nobody is listed.

**Words** (`describe.video_prompt`): only what holds whatever part of anyone or anything the frame shows. No clothes, props, or landmarks, no screen side, no facing: words cannot know which part of a person or thing is in frame, and a described part that is not gets drawn anyway. Every piece is a `prompts.json` template; nothing names a scene:

1. The shot (`shotSentence`): size and angle from the script's `shot` (`shot.sizes`, `shot.angles`; close sizes add a shallow depth of field). Then the camera: `cameraMotion` with an own moving camera's `cameraMotion`, `cameraMoves` for a shot's move, else `cameraHeld`.
2. Everyone with a performance, in the script's order, by their `body` only, with their action (`{target}`, always a person, named by their `body`) and their `expression` (`person`, `expression`); a performer not in the start frame comes into it (`personEnters`). Nobody else is named, the shot's `over` included: the depth video gives their shape.
3. Effects in shot (`effects`), each stretch of the backdrop in shot (`backdrop`: above, underfoot, behind), the scene's `motion`, and the location's `light`.

Lines are never in a picture prompt: told among the actions, a voice from off screen makes the model cut to its speaker. No names, ids, or pronouns are written. Negative: `vaceNegative`.

Output: `output/frames/<show>/<episode>/scene_XX_start.png`; `inputs/scene_XX_start/` keeps the depth video, the reference sheet, and `log.json` (prompt, size, seed, who is on the sheet).

### No output checks

Stills and clips are not checked automatically and not remade on their own: each is made once, at a seed fixed by its scene, and reviewed. `--force` remakes one selected scene; `--seed` rerolls it.

### 5. Clips — Chatterbox, then Wan S2V or VACE, then RIFE

`content:generate` makes each scene's clip in three steps (`generate_batch.generate_clip`).

**Voice** (`voice.py`, Chatterbox in `content-pipeline/.voice`, installed by `content:voice-deps`). One speaker's lines in a row are one speech, spoken in one go so the voice runs on, starting at the first line's `startSeconds`. The speeches are laid on a track exactly the scene's length (`output/generate/<show>/<episode>/voices/scene_XX.wav`); a scene without lines gets a silent track, so every clip has sound. A speech that runs into the next one or past the scene is reported. Each character speaks in their voice, `voice.wav` beside their reference photo (made by `content:plates`); validation requires a `voice` for every speaker. Exaggeration 0.6, CFG weight 0.4. Chatterbox reads no delivery words; the delivery reaches only the picture. Background sound and music are not made here.

**The take**, at 16 fps (Wan 14B's own rate), `4n+1` frames covering the scene (`wan.frames_for`):

| Scene | Model | Inputs |
| --- | --- | --- |
| A speaker with a performance, i.e. on screen (`describe.talking_speaker`: the first such speaker) | Wan 2.2 S2V 14B (fp8): 20 steps, CFG 6, uni_pc, shift 8, wav2vec2 audio encoder, and its first-frame fix (the first latent decoded twice, `S2V_DROP_FRAMES` 3 dropped) | The start still and the voice track. Prompt: `video_prompt` plus how they speak (`speaking`: the lines' deliveries, never the words). Negative: `s2vNegative` (the template's). |
| Anything else: no lines, or only voices from off screen | Wan 2.2 VACE, as for the still, over the whole take | The depth video and the reference sheet. The voice track is laid under it. |

S2V moves one face. In a take with two speakers on screen it is about the first, and mouths every line on the track (a model limit).

**24 fps.** RIFE (ComfyUI-Frame-Interpolation, `rife49`, ensemble) triples the take to 48 fps with its sound, every other frame is dropped (24 fps), and the clip is cut to the scene's exact length.

Output: `output/generate/<show>/<episode>/scene_XX.mp4`; `inputs/scene_XX/` keeps the 16 fps take, the 48 fps take, a VACE take's depth video and sheet, and `log.json` (prompt, size, model, seed, speaker, and digests of the voice and the still).

When every scene has its clip, ffmpeg loudness-normalizes each to `episodeLoudnessTargetLufs` (-16 LUFS) and joins them into `episode.mp4`; `concat_videos` aborts if their width, height, or fps differ. `captions.caption_episode` then burns in subtitles from each scene's `lines`, timed to the words Whisper (large-v3-turbo) hears, and a name card (`characters.<id>.name`) the first time someone speaks on screen; the plain cut is kept as `episode_plain.mp4`. A single-scene run leaves `episode.mp4` alone.

`content:upload` pushes the finished MP4s and thumbnails to Cloudflare R2 and registers episodes on the Worker. That step is distribution, not generation.

## Models on disk

`pnpm run content:models` downloads these into `content-pipeline/.comfyui/models/` (`download_models.py`, one table) and skips files already present.

| Stack | Files | Licence (model card) |
| --- | --- | --- |
| Wan 2.2 VACE Fun A14B | high- and low-noise experts, Q8 GGUF (QuantStack) | Apache-2.0 |
| Wan 2.2 S2V 14B | fp8 weights, `wav2vec2_large_english` audio encoder | Apache-2.0 |
| Wan, shared | `umt5_xxl_fp8_e4m3fn_scaled` text encoder, `wan_2.1_vae` | Apache-2.0 |
| Qwen-Image-2512 | Q8 GGUF (unsloth), `qwen_2.5_vl_7b_fp8_scaled` text encoder, `qwen_image_vae` | Apache-2.0 |
| TRELLIS.2 | int8 UNet (~5GB), DINOv3 ViT-L, shape VAE, BiRefNet | MIT |
| Pixal3D | int8 UNet (~5.2GB), DINOv3 with NAF, texture VAE, MoGe-2. Shares the TRELLIS shape VAE. | MIT |
| RIFE | `rife49.pth`, fetched by ComfyUI-Frame-Interpolation on first use | MIT |
| Chatterbox | `content-pipeline/.voice` (`voice-requirements.txt`, installed by `content:voice-deps`); weights fetched on first use | MIT |
| Qwen3-TTS-12Hz-1.7B-VoiceDesign | `content-pipeline/.voicedesign` (`voice-design-requirements.txt`, installed by `content:voice-deps`); weights fetched on first use | Apache-2.0 |

`content:assets -- --credits` rewrites `docs/CREDITS.md` from the records under `output/assets/`.

## ComfyUI checkout

`pnpm run content:setup-comfy` clones into `content-pipeline/.comfyui`:

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI), including native Wan (VACE, S2V), Qwen-Image, TRELLIS.2, and Pixal3D nodes
- [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)
- [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)
- [ComfyUI-Frame-Interpolation](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation) (RIFE)

`pnpm run content:asset-deps` installs trimesh and moderngl into that same virtualenv. trimesh reads and writes glTF. moderngl draws previs and the depth video. Do not install `content-pipeline/requirements.txt` into that virtualenv.

## Stage viewer

`pnpm run view -- --show <id>` serves `content-pipeline/viewer` at `http://127.0.0.1:5174`. It is Vite and Three.js. It loads every landmark, prop, and character mesh. A scene replays previs's `scene.json`: each entity's pose per sample and the camera, glTF Y-up. Orbit is the god camera. Scene camera locks to the authored lens and plays the timeline.

## Resolutions

| Stage | Size | Why |
| --- | --- | --- |
| Plates and mesh conditioning | 1024 square | TRELLIS.2 / Pixal3D shape cascade |
| Reference photos | 928×1664 | Qwen-Image-2512's 9:16; a whole figure with room above the head and below the feet |
| Previs | 768×1360 | Vertical, near 9:16 |
| Wan stills and takes | 480×832 (`--size 480`, the default) or 720×1280 (`--size 720`) | 480p answers most questions in a fraction of the time; 720p is the final look. The depth video is rendered at the take's own size. |
| Clips | the take's size, 24 fps | RIFE from Wan's 16 fps |
