/**
 * Authoring JSON for `content-pipeline/scripts_input/<id>.json`. One file per show.
 *
 * Story guidance for the script-writing model. These are authoring instructions, not
 * runtime validation rules; the renderer validates only fields needed to execute safely.
 * - Write a vertical micro-drama, not disconnected prompt demonstrations. Each episode is
 *   a causal chain: setup -> pressure -> choice/reveal -> reaction -> cliffhanger.
 * - One `ScriptScene` is one generated shot. Use 10-12 shots totaling at least 60 seconds
 *   per episode. Actions, dialogue, and reactions get separate shots.
 * - Every spoken line has a clear speaker and addressee. A character must not answer a
 *   question the audience never heard or refer to a prop they never saw established.
 * - Ground every conversation with a two-shot before using singles/reactions. In a
 *   three-person exchange, use pairwise anchors when the participant changes: A+B establish
 *   the first axis, then C+B establishes the entrant, then C+A grounds their connection.
 *   Never fake a three-shot: Qwen receives at most two identity references.
 * - Continuity is explicit because the models remember nothing between shots.
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
   * Qwen-Image-Edit-2511 template. Give an exact per-shot picture map and output person
   * count. Never mention a nonexistent Picture 2: the four-step model may duplicate Picture
   * 1 when given conditional multi-picture wording.
   *
   * Exact template:
   * `The attached identity pictures map exactly as follows: {referenceMap} Transform those
   * people into one new photorealistic vertical 9:16 scene. The finished scene contains
   * exactly {characterCount} visible people: {characterIds}. Each appears once only. No
   * duplicates, twins, background people, portraits, paintings, mirrors, or reflections.
   * Preserve each referenced face, hair, apparent age, and skin tone, but do not copy the
   * reference backdrop, shirt, pose, or gaze. This is a dramatic film frame, not a frontal
   * identity portrait; obey the stated eyeline and placement. Do not visualize an off-frame
   * or absent addressee. {locationPromptBlock} {imagePrompt}`
   *
   * `{referenceMap}` `{characterCount}` `{characterIds}` `{locationPromptBlock}`
   * `{imagePrompt}`
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
   * Use one locked continuous take with constant framing, exposure, and color. Keep each
   * person planted at the same distance and preserve the starting body angle and eyeline
   * unless the motion prompt explicitly changes them. The final frame remains a normally
   * lit continuation of the shot, not a fade or transition. {videoPrompt}`
   *
   * `{videoPrompt}`
   */
  sceneVideo: string;
};

export type ScriptScene = {
  sceneNumber: number;
  locationId: string;
  /**
   * Why this shot exists in the story, written as cause and effect rather than visuals.
   * Example: "Julian publicly rejects Mara, causing her humiliation." The next shot must
   * respond to this beat; no isolated exposition.
   */
  storyBeat: string;
  /**
   * Relevant state already established before this shot: wardrobe, relationships, known
   * information, location, and important prop state. Must agree with the prior shot's
   * `continuityOut`.
   */
  continuityIn: string;
  /**
   * What visibly or narratively changes by the end of this shot. The following shot must
   * begin from this state.
   */
  continuityOut: string;
  /**
   * `single`: one visible speaking/acting character. `reaction`: one visible character
   * reacts while another may speak off-screen. `twoShot`: two visible people establish or
   * refresh their shared space. Start each conversation/entrant with a two-shot, then cut to
   * singles; return to a two-shot after several singles or when the active pair changes.
   * In video, only the speaker moves their mouth; explicitly keep the listener silent with
   * a closed mouth and restrained reaction.
   */
  shotType: "single" | "reaction" | "twoShot";
  /**
   * One ID preferred, two maximum. Order is exact: first ID = Qwen Picture 1, second ID =
   * Picture 2. Every listed character must be clearly visible exactly once in `imagePrompt`;
   * no unlisted visible people.
   */
  characterIds: string[];
  /**
   * Character delivering the quoted line, or null for a silent shot. May be absent from
   * `characterIds` only for an off-screen line over a visible reaction.
   */
  speakerId: string | null;
  /**
   * Character the line/action is directed toward, or null when genuinely private. Usually
   * visible or established immediately off-screen. Never use dialogue without an addressee
   * merely to explain the plot to the audience.
   */
  addresseeId: string | null;
  /**
   * Qwen description of the clip's exact first frame. Repeat complete wardrobe and visible
   * props. Specify shot size, left/right placement, gaze target, and emotion. Singles and
   * reactions should be chest-up or medium close-ups with the subject looking toward the
   * established off-screen addressee, never into the camera. Two-shots keep both faces
   * readable but need not be symmetrical or posed.
   *
   * Establish the beginning of one achievable action. Simple turns, one step, raising or
   * lowering an already-held prop, and restrained gestures are allowed. Avoid readable
   * documents/screens, mirrors, crowds, fights, complex hand contact, transfers between
   * people, and tiny plot-critical details.
   *
   * Critical: name only characters listed in `characterIds`. Never name an off-frame or
   * absent addressee—the four-step Qwen model may render that name as an extra, unreferenced
   * person. Express eyelines as "toward empty space beyond frame left/right." Describe an
   * absent character's clothing or prop generically, without possessive names.
   */
  imagePrompt: string;
  /**
   * LTX motion continuing directly from the first frame. A silent shot gets one meaningful,
   * achievable action that starts immediately. A dialogue shot starts speech immediately
   * and gets only natural acting during/after the line—never put a step, turn, or prop action
   * before speech, because LTX often performs it after the words instead.
   *
   * At most one quoted line of 16 words, spoken by `speakerId` to `addresseeId`. Put the
   * quoted line near the beginning and explicitly say speech begins immediately with no
   * silent pause. Specify projected volume, emotional intensity, pace, and vocal texture;
   * vary these by beat instead of defaulting to flat/quiet speech. Outside quoted dialogue,
   * name only visible `characterIds`; direct eyelines toward the frame edge instead of
   * naming an absent addressee, or LTX may invent them. For a reaction shot, keep an
   * off-screen voice brief and animate only the visible listener. Do not alternate speakers,
   * cut angles inside a clip, introduce a new person, or repeat the first frame.
   *
   * Dialogue shots keep the camera, body position, body angle, and off-camera eyeline fixed
   * for the full clip. Do not add an after-line turn, step, approach, exit, zoom, push-in,
   * lighting change, fade, or transition: distilled LTX often converts such end beats into
   * spatial drift or a different face. Use only blinks, breathing, lip movement, and a small
   * expression change during the line. End with one short ambience/foley sentence. If the
   * beat needs a reply or physical action, create the next shot.
   */
  videoPrompt: string;
  /**
   * Clip length in seconds. Dialogue shots are 6 seconds so a substantial line can begin
   * immediately and land emotionally. Silent inserts/reactions are 4 or 6 seconds. Episode
   * scenes together must total at least 60 seconds.
   */
  durationSeconds: number;
};

export type ShowEpisode = {
  episodeNumber: number;
  title: string;
  isFree: boolean;
  coinCost: number;
  /**
   * Persistent 180-degree-axis plan for every recurring location in this episode. Assign
   * each character a fixed screen side before writing shots, then derive every off-camera
   * eyeline from that map. Example: "At the desk, Elena is screen left and always looks
   * frame right toward Marcus; Marcus is screen right and always looks frame left. Theo
   * enters from farther screen left and looks frame right." Never independently choose
   * left/right per prompt. A character may reverse eyeline only after an establishing shot
   * or a visible turn shows that their addressee changed.
   */
  screenDirection: string;
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
