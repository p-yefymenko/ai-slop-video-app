# ai-slop-video-app

The goal of `content-pipeline/` is a pipeline that turns any `script.json` into an episode. A good clip for one scene is worth nothing unless the pipeline code made it. Every hour spent must leave the pipeline better for every scene, not one scene better looking.

## Working on the content pipeline

### Code first, render second

- Generation code lives in `content-pipeline/scripts/`: graph builders, prompt builders, and stages in `generate_batch.py`. Each stage is wired to a root `package.json` command. Never build a parallel chain of scratchpad scripts.
- Render only through pipeline code, by scene number and show, never from hand-picked files. If the code to run it does not exist yet, write that code first.
- The scratchpad is only for throwaway helpers that render nothing: inspecting a file, extracting frames, measuring something.

### Nothing hardcoded for one scene

Before writing any value, ask: "would this be right for every scene where this condition holds?" If not, it does not go in.

- **No hand-written prompt text.** Every word comes from `script.json`, the 3D scene (`observe`, `world`), or a template in `prompts.json`. When a result is missing something, the fix is one of these:
  - a general rule in the prompt builder, with its condition (e.g. "a person who is only a soft edge of the frame gets words but no plate");
  - a fix in `script.json` (blocking, wording);
  - a template change.

  Never a sentence about one scene's fur or one character's beard.
- **No per-scene code or values.** No `if scene == N`, no per-scene seeds, strengths or step counts, and no paths to a picked plate or take.
- **One fact, one source.** Never restate in words what a model input already shows exactly (the framing in a layout or depth video, where people stand, which way they face). Authored words for the same fact (`shot.size`, `shot.type`) drift from the authored camera, and the model follows whichever it trusts more.
- **Settings are named constants with a reason**, in one place, the same for every scene. Change one only when the change is meant for all scenes.
- **Picking by eye is not a step.** When a choice is needed (a plate seed, the best take), code makes it with a deterministic rule from the script. If it can't, flag it to the user as an open problem.
- **No stopgaps.** Never use or offer a hand-picked file, a leftover from an experiment, or a one-off workaround in place of a general method (e.g. an old recording as a character's voice). Name the general fix and what it takes; that is the only option to present.

### Only robust code: no heuristics

The pipeline uses only authored script data (`shot.subjects`, `shot.partlySeen`, `performances`, `lines`, …) and exact geometry (the timeline, the meshes, the camera, the depth render, "any pixel of it is in the frame").

- **No thresholds on pixels or mesh regions** (minimum sizes, visible shares, height bands standing in for body parts, color tolerances, mesh cleanup by size).
- **The video prompt never describes what may be only partly in frame.** No clothes, props, or landmarks, no screen side, no facing: words cannot know which part of a person or thing the frame shows, and a described part the frame does not show gets drawn anyway. People are named only by `body`; looks come from the reference sheet, shapes from the depth video.
- **No detectors on model output** (person or face detectors, cut or drift checks), and no automatic retries driven by them. Outputs are reviewed by the user.
- **No word-list lints** in validation (pronouns, effect words, music words). Validation checks structure and references only.
- When a rule seems to need a threshold or a detector, it needs an authored field instead: add it to the schema (`packages/shared/src/script.ts`) and the script.
- **The LLM is part of the pipeline: it authors judgment.** Where a decision needs understanding and code can only guess (which part of a partly seen person the frame shows, what a shape in the layout is), Claude writes the answer into an authored field after looking at the inputs (e.g. `shot.partlySeen` from `to_draw/*/layout.png`). This is not hand-tuning: the field is general, validated (structure and references), reviewable in `script.json`, and gives the same result on every run. Prefer it over any threshold. It is not a live model call from code.

### Every GPU run must answer a question

Before starting anything on the GPU, tell the user in one or two lines:

1. What question it answers.
2. What changes in the code for each possible outcome. If no outcome changes the code, don't run it.
3. How long it will take.

Then:

- **Review every prompt before any generation.** Run `pnpm run content:prompts` (same flags as the run) and read each prompt it prints: the still's (for the image model) and the take's. Check them against the rules here (nobody without a performance described, no part that may be off frame, nothing a model could draw that the shot must not show, no word that contradicts the view). Show the user any prompt a code change altered. A run whose prompts were not read is not started.
- **Use the cheapest run that answers the question:** a still before a clip, 480p before 720p, one scene before an episode, a short take before a full one.
- **Don't re-render a finished result at higher quality just to show it.** Final-quality renders come from the integrated pipeline, at the end.
- **Check with the user before any run over ~15 minutes,** unless they asked for exactly that run.
- **Stop runs that no longer serve a decision.** Don't leave a long run going when nothing waits on it. Stopping a client script doesn't stop ComfyUI: also POST `/interrupt` and clear `/queue`.
- **Probes are allowed only when the answer decides a general rule.** A probe tests whether a model can do something (a node, an input, a prompt trick). Say up front that it's a probe, keep it small, never polish it, and turn its outcome into code or a recorded decision right away.

### When a scene fails

1. Name the general cause:
   - a missing input (3D truth such as depth, or a plate);
   - a wrong prompt-builder rule;
   - a script problem;
   - a model limit.
2. Fix that cause in code, with the condition under which it applies.
3. Rerun the earlier scenes that the rule touches, to check for regressions.

Record a model limit as a model limit. Never paper over it with scene-specific words.

### Reporting

- Review every clip fully before reporting: frame strips over the whole clip, with every defect listed.
- Whenever a result is shown, say plainly what is general and what is still scene-specific or picked by hand.
- Keep messages short. Don't ask questions that the code or the script can answer.

## Conventions

- **Authored:** `content-pipeline/shows/<id>/script.json`. The schema is `packages/shared/src/script.ts`; validate with `pnpm run content:validate`. Use the `write-script` skill when writing or editing a script.
- **Generated:** everything goes under `content-pipeline/output/<command>/<show>/`.
- **Running code:** commands are root `package.json` scripts. ComfyUI-side Python runs with `content-pipeline/.comfyui/.venv`.
- **Tests:** `pnpm run content:test-spatial`. Add tests for new pure logic: prompt builders, depth encoding, frame counts.
- **Coordinates:** schema space is Z-up, +Y forward; glTF is Y-up. Convert only in `content-pipeline/scripts/coords.py`.
- **Separate envs (gitignored):** `content-pipeline/.voice` (Chatterbox) and `content-pipeline/.voicedesign` (Qwen3-TTS VoiceDesign). Free ComfyUI's VRAM (POST `/free`) before starting another GPU process.
- **Git:** commit only when the user asks.
