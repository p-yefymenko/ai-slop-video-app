---
name: draw-stills
description: Draw scene start stills in the Gemini web app from content:frames' to_draw requests, through Claude in Chrome, and save them where content:generate reads them. Use when the user asks to draw, make or generate stills, or to do "the Gemini step".
---

# Drawing the start stills in Gemini

Every take starts from a still the Gemini app draws: the API needs a paid key, and
the app's free quota is a few images a day. `content:frames` writes one request per
scene under `content-pipeline/output/frames/<show>/<episode>/to_draw/scene_XX_start/`:

- `layout.png`: the shot's 3D layout in the meshes' own colors. It is missing when
  none of the 3D scene is in the frame; then the words go alone.
- `prompt.txt`: the words. Sent exactly as written, never edited or added to. A
  wrong word is fixed in the code, a template or the script, then the request is
  rewritten (CLAUDE.md: no hand-written prompt text).

The picture is saved beside the folder as `scene_XX_start.<ext>`, keeping the
format Gemini gives (PNG, JPG and WebP are all read).

## Before drawing

1. Rewrite the requests: `pnpm run content:frames --show <show> --episode <n>`
   (add `--scene <n>` for one scene). It runs previs first, with no GPU generation.
2. Read every `prompt.txt` to be drawn, and look at every `layout.png`. Check them
   against CLAUDE.md. For anyone only partly in the frame whose colors leave the
   material a guess (a near shoulder, a cut at the frame's edge), author
   `shot.partlySeen` in the script (write-script skill), then rewrite the requests.
3. Tell the user which scenes will be drawn. Each still costs one image of their
   daily Gemini quota, so draw only scenes without a still, plus the ones they
   name. Never draw a scene again to get a better result unless they ask.

## Drawing one still

Use the browser tools (Claude in Chrome). If they are not connected, ask the user
to run `/chrome` (or type `@browser`) and stop until they do.

1. Open `https://gemini.google.com/app` in a new tab: a new chat for every still,
   so no earlier picture or words carry over.
2. If a sign-in page, consent screen or CAPTCHA shows, stop and ask the user to
   handle it. Never type their email or password.
3. If the request has a `layout.png`, attach it with the upload control (the "+" or
   paperclip by the prompt box).
4. Paste the whole of `prompt.txt` into the prompt box and send it.
5. Wait until the picture is done. If Gemini answers in text, refuses, or says the
   image limit is reached, stop and report its words; do not reword and retry.
6. Download the picture at full size (open it, then use the download button). Find
   the newest file in the user's Downloads folder, check that it is an image made
   after the send, and move it to `scene_XX_start.<its extension>` beside the
   `to_draw` folder. Before moving, delete any other `scene_XX_start.*` there (the
   clip stage refuses two), but only a file whose stem is exactly
   `scene_XX_start`, never one such as `scene_04_start_old.png`.

## Character reference photos

Each character's reference photo is drawn the same way. Pixal3D builds their mesh
from it, the mesh is painted from it, and the video model gets it as the reference.
`pnpm run content:plates --show <show>` writes one request per character whose
photo is not drawn from the current words:
`content-pipeline/output/plates/<show>/characters/<id>/to_draw/prompt.txt`.

1. Draw each one as above, but with the words alone (there is no layout), in a new chat.
2. Save the download beside `to_draw` as `reference.<its extension>`, after the
   request was written (an older file is not taken). Replace only a file named exactly
   `reference.*`.
3. Run `pnpm run content:plates --show <show>` again. It takes each new picture as
   `reference.png` and logs it, and drops the character's old mesh.
4. Review each photo against the words: the whole figure head to toe on plain white,
   standing and facing the camera with the arms at the sides, every attribute item by
   item, nothing extra. Then `pnpm run content:assets` remakes the meshes on the GPU
   (about 10 minutes each). Say so before starting it.

## After drawing

Review every still against its layout, as CLAUDE.md asks for every result. Read
both images and list each defect:

- **Framing:** each person where the layout has them, as large, with the same cut
  at the frame's edges. Nobody pulled back to full length or added.
- **Partly seen people:** only the part the layout shows, in the right material.
- **Faces:** the expression the words give, and no face for someone seen from behind.
- **Artifacts** copied from the mesh colors, such as specks.

For each defect, name the general cause (a word, a missing `partlySeen`, a mesh
color, or a model limit) and the fix. Report the stills drawn, the ones that were
not and why, and the quota used.
