/**
 * Authoring JSON for `content-pipeline/scripts_input/<id>.json`. One file per show.
 *
 * Reliability contract for the script-writing model:
 * - One `ScriptScene` is one independently generated still plus one short I2V clip, not a
 *   screenplay scene. Make every shot atomic. Split a beat into more scenes whenever it
 *   needs a new pose, composition, speaker, prop state, or camera angle.
 * - Prefer one visible character. Use two only when their shared frame is essential.
 * - Never ask either model to invent a crowd, readable text, a reflection, or an unseen
 *   character. Put off-screen voices in audio only.
 * - Continuity is explicit: repeat wardrobe and held props in every `imagePrompt`. Models
 *   do not remember earlier scenes.
 */

export type ShowCharacter = {
  /**
   * Qwen identity source only; scene stills attach its PNG and never receive this text.
   * Describe one adult's stable physical identity: apparent age, build, skin, face shape,
   * eyes, distinctive facial feature, and hair. No wardrobe, pose, expression, action,
   * location, camera direction, alternatives, or other people. Keep it one short sentence.
   */
  promptBlock: string;
};

/**
 * Reusable Qwen environment only. Describe one standable set at human scale: the surfaces
 * immediately behind/beside the characters, practical light, and time of day. Keep it to
 * one or two short sentences. No people, action, wardrobe, camera, shot size, viewpoint,
 * multiple rooms, distant architecture, readable text, mirrors, or crowds.
 */
export type ShowLocation = {
  promptBlock: string;
};

/**
 * Templates sent verbatim to Qwen/LTX. Copy the exact template shown in each field comment
 * and keep every listed `{placeholder}` exactly once. Do not improvise these templates or
 * move per-scene story content into them.
 */
export type ShowPrompts = {
  /**
   * Qwen identity-still template. Must request exactly one neutral, waist-up, front-facing
   * person on a plain backdrop, then include `{characterPromptBlock}`. This is a reference
   * portrait, not a story scene; require plain fitted clothing, hands out of frame, closed
   * mouth, direct gaze, and even light.
   *
   * Exact template:
   * `Photorealistic vertical 9:16 identity reference, waist-up, exactly one person facing
   * camera, neutral closed-mouth expression, hands out of frame, plain fitted crew-neck
   * shirt, plain warm-grey studio backdrop, soft even light, natural skin, sharp eyes. No
   * text, props, jewelry, costume, or other people. Ignore the attached blank image and
   * create a new person from this description: {characterPromptBlock}`
   */
  characterImage: string;
  /**
   * Qwen-Image-Edit-2511 template. State that Picture 1 and optional Picture 2 are identity
   * references, preserve their faces, and create one new photograph. Map references in the
   * same order as `{characterIds}`; then include the environment and shot description.
   *
   * Exact template:
   * `The attached pictures are identity references in this exact order: {characterIds}.
   * Picture 1 is the first named person; Picture 2, if attached, is the second. Create one
   * new photorealistic vertical 9:16 photograph. Preserve each referenced face, hair,
   * apparent age, and skin tone. Show each named person exactly once. Do not copy the
   * reference backdrop, shirt, pose, or gaze. {locationPromptBlock} {imagePrompt}`
   *
   * `{characterIds}` `{locationPromptBlock}` `{imagePrompt}`
   */
  sceneStill: string;
  /**
   * LTX-2.3 I2V template. State that the attached image is the exact first frame, must keep
   * its people/composition/wardrobe/set, and then include only the chronological motion.
   * Do not ask LTX to add a person or repair/change anything visible in the still.
   *
   * Exact template:
   * `Continue directly from this image as the exact first frame. Keep the same people,
   * faces, wardrobe, props, composition, lighting, and set. Do not add people or objects.
   * {videoPrompt}`
   *
   * `{videoPrompt}`
   */
  sceneVideo: string;
};

export type ScriptScene = {
  sceneNumber: number;
  locationId: string;
  /**
   * One ID preferred, two maximum. Order is exact: first ID = Qwen Picture 1, second ID =
   * Picture 2. Every listed character must be clearly visible exactly once in `imagePrompt`;
   * no unlisted visible people.
   */
  characterIds: string[];
  /**
   * Qwen description of the clip's exact first frame, written as one static photograph with
   * no before/after sequence. Repeat each character's complete scene wardrobe and held prop.
   * Specify a simple medium close-up or waist-up composition, left/right placement, gaze,
   * and one stable emotion/pose. With two people, keep both faces unobscured, similar-sized,
   * and on the same focal plane.
   *
   * Use a pose from which `videoPrompt` can begin without changing geometry. Avoid wide/full
   * body shots, extreme close-ups, crossed/hidden limbs, hand-to-hand contact, fights,
   * embraces, walking poses, object transfer, pouring, eating, dressing, readable documents,
   * phones/screens, mirrors, crowds, duplicate people, and important tiny objects. If a beat
   * depends on one of these, show its stable result or split it into simpler shots.
   */
  imagePrompt: string;
  /**
   * LTX chronological motion continuing directly from `imagePrompt`. One clip gets one
   * primary visible actor, one simple continuous action, and one camera behavior. Prefer
   * facial micro-action (blink, breath, gaze shift, slight head turn) over hand/body action.
   * Keep the camera locked; use at most one slow push-in only when it helps the beat.
   *
   * For dialogue, use one visible speaker per clip, one quoted line of at most 12 spoken
   * words, and name the speaker immediately before it with voice quality. The other person
   * may only hold a simple reaction. Do not alternate speakers, make both mouths move, or
   * combine speech with walking, prop manipulation, touch, entrances/exits, large gestures,
   * camera cuts, angle changes, time jumps, transformations, or newly appearing people or
   * objects.
   *
   * End with one short ambience/foley sentence. Do not redescribe the still. If the story
   * beat needs another action, speaker, prop state, or viewpoint, create another scene.
   */
  videoPrompt: string;
  /**
   * Clip length in seconds. Use 4 for a silent micro-action, 6 for one short spoken line,
   * or 8 only for one slow action with one short line and reaction. More time is not a
   * license to add more events.
   */
  durationSeconds: number;
};

export type ShowEpisode = {
  episodeNumber: number;
  title: string;
  isFree: boolean;
  coinCost: number;
  scenes: ScriptScene[];
};

export type ShowScript = {
  id: string;
  title: string;
  characters: Record<string, ShowCharacter>;
  locations: Record<string, ShowLocation>;
  prompts: ShowPrompts;
  episodes: ShowEpisode[];
};
