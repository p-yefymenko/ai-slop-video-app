/**
 * Authoring JSON for `content-pipeline/scripts_input/<id>.json`. One file per show.
 *
 * Authoring JSON for one vertical microdrama. Persistent screenwriting guidance lives in
 * `.cursor/rules/episode-scripts.mdc`; these field comments document renderer semantics.
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

export type Vec3 = [number, number, number];

export type StageLandmark = {
  /** Stable object ID used by blocking, e.g. `crown_pedestal` or `eclipse_window`. */
  kind: "box" | "column" | "pedestal" | "window" | "door" | "seat";
  position: Vec3;
  size: Vec3;
};

export type StageGeometry = {
  /** Interior dimensions `[width X, depth Y, height Z]` in meters. */
  sizeMeters: Vec3;
  landmarks: Record<string, StageLandmark>;
};

/**
 * Reusable Qwen environment only. Describe one standable set at human scale: the surfaces
 * immediately behind/beside the characters, practical light, and time of day. Keep it to
 * one or two short sentences. No people, action, wardrobe, camera, shot size, viewpoint,
 * multiple rooms, distant architecture, readable text, mirrors, or crowds.
 */
export type ShowLocation = {
  promptBlock: string;
  /** Deterministic blocking space used by every scene at this location. */
  spatial: StageGeometry;
};

export type CharacterSpatialKeyframe = {
  timeSeconds: number;
  locationId: string;
  /** Feet position in location-local meters. */
  position: Vec3;
  /** Body rotation around Z. `0` faces +Y; positive turns toward +X. */
  bodyYawDegrees: number;
  /** Character ID or landmark ID the head/eyes face. */
  lookAtId?: string;
  stance: "standing" | "sitting" | "kneeling" | "walking";
  leftHandTargetId?: string;
  rightHandTargetId?: string;
};

export type PropSpatialKeyframe = {
  timeSeconds: number;
  locationId: string;
  position?: Vec3;
  heldByCharacterId?: string;
  heldInHand?: "left" | "right";
};

export type SpatialTimeline = {
  durationSeconds: number;
  characterTracks: Record<string, CharacterSpatialKeyframe[]>;
  propTracks: Record<string, PropSpatialKeyframe[]>;
};

export type SpatialCamera = {
  position: Vec3;
  lookAt: Vec3;
  verticalFovDegrees: number;
  endPosition?: Vec3;
  endLookAt?: Vec3;
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
   * Qwen identity-turnaround template. Picture 1 is the approved frontal identity.
   * Uses `{characterId}` and `{direction}` (`left` or `right`) to create exactly one
   * neutral three-quarter/profile reference on the same studio backdrop.
   */
  characterProfile: string;
  /**
   * Qwen template that removes all people from a coverage master while preserving the
   * exact set, camera, permanent landmarks, and light. Uses `{locationPromptBlock}`.
   */
  setPlate: string;
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
   * Qwen template for dialogue coverage derived from an already-rendered master shot.
   * Identity portraits remain the first pictures; the final picture is the coverage master.
   * Use `{referenceMap}`, `{coverageReferencePictureNumber}`, `{characterCount}`,
   * `{characterIds}`, and `{imagePrompt}`.
   */
  coverageStill?: string;
  /**
   * Qwen master-shot template using identity picture(s) followed by the rendered 3D proxy.
   * Uses `{referenceMap}`, `{proxyPictureNumber}`, `{characterCount}`, `{characterIds}`,
   * `{locationPromptBlock}`, `{blockingSummary}`, and `{imagePrompt}`.
   */
  spatialStill: string;
  /**
   * Qwen dialogue-single template using an eyeline-matched identity, person-free set plate,
   * then character-only pose proxy. Adds `{coverageReferencePictureNumber}`.
   */
  spatialCoverageStill: string;
  /**
   * Qwen environment/insert template using the 3D proxy as Picture 1.
   * Uses `{proxyPictureNumber}`, `{locationPromptBlock}`, and `{imagePrompt}`.
   */
  spatialEnvironmentStill: string;
  /**
   * Qwen prop-insert template using a crop from an earlier coverage master plus proxy.
   * Uses `{focusTargetId}`, `{locationPromptBlock}`, and `{imagePrompt}`.
   */
  spatialInsertStill: string;
  /**
   * Qwen end-guide template for a moving two-character shot. Picture 1 is the
   * photorealistic start and Picture 2 is the end-state proxy.
   */
  spatialGroupEndStill: string;
  /**
   * Qwen final-guide template: Picture 1 photorealistic start, Picture 2 identity, Picture 3
   * end-state proxy. Use only for action shots; static dialogue copies its start guide.
   * Uses `{characterId}` and `{imagePrompt}`.
   */
  spatialEndStill: string;
  /**
   * Qwen template for establishing shots and prop inserts with no identity references.
   * Uses `{locationPromptBlock}` and `{imagePrompt}` from a blank canvas.
   */
  environmentStill: string;
  /**
   * LTX-2.3 I2V template. State that the attached image is the exact first frame, must keep
   * its people/composition/wardrobe/set, and then include only the chronological motion.
   * Do not ask LTX to add a person or repair/change anything visible in the still.
   *
   * Exact template:
   * `Continue directly from this image as the exact first frame. Preserve its people,
   * wardrobe, props, set, composition, and lighting. Use one continuous take. Animate only
   * the motion, performance, camera, dialogue, and sound described here: {videoPrompt}`
   *
   * `{videoPrompt}`
   */
  sceneVideo: string;
};

export type ScriptScene = {
  sceneNumber: number;
  /** Dramatic function in the one-minute episode, independent of camera coverage. */
  beatType: "hook" | "pressure" | "reversal" | "cliffhanger";
  /** Editing function: spatial anchor, speaking close-up, silent reaction, or prop detail. */
  coverageRole: "establishing" | "anchor" | "closeup" | "reaction" | "insert";
  locationId: string;
  /**
   * Why this shot exists in the story, written as cause and effect rather than visuals.
   * The next shot must respond to this beat; no isolated exposition.
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
   * `establishing`/`insert`: no visible character. `single`: one speaking/acting character.
   * `reaction`: one silent listener. `twoShot`: two people in a brief silent spatial anchor.
   */
  shotType: "establishing" | "insert" | "single" | "reaction" | "twoShot";
  /**
   * `cameraOnly` preserves the approved still and applies deterministic camera motion.
   * Use it for establishing shots, prop inserts, reactions, and silent anchors unless a
   * visible action truly requires synthesis. `generative` invokes LTX.
   */
  motionMode?: "cameraOnly" | "generative";
  /**
   * Empty for establishing/insert shots; otherwise one ID preferred and two maximum. Order
   * is exact: first ID = Qwen Picture 1, second ID = Picture 2.
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
   * Earlier anchor/master scene whose rendered PNG supplies the set, lighting, axis, and
   * screen geography for this shot. Use it for every single in a dialogue coverage run.
   * The pipeline attaches it after the identity portrait(s); omit it for masters and inserts.
   */
  coverageReferenceSceneNumber?: number;
  /**
   * Prop or landmark ID centered by an insert. Inserts inside an established set should
   * combine this with `coverageReferenceSceneNumber` instead of reinventing the prop.
   */
  focusTargetId?: string;
  /**
   * Interval on the episode spatial timeline projected into this edit shot.
   * Omit only for legacy scenes that have not yet been migrated.
   */
  timeRangeSeconds: [number, number];
  /** Physical camera used to project the timeline into a proxy guide. */
  camera: SpatialCamera;
  /** Additional timeline times to pin as photorealistic LTX guides. */
  guideKeyframesSeconds?: number[];
  /** Generate and pin a photorealistic guide at the shot's final timeline state. */
  endGuideFrame?: boolean;
  /**
   * Qwen description of the clip's exact first frame. Repeat complete wardrobe and visible
   * props. Specify shot size, left/right placement, gaze target, and emotion. Dialogue
   * singles should vary between medium close-up, tight close-up, and extreme close-up while
   * preserving reciprocal frame sides from their coverage master. Two-shots keep both faces
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
   *
   * For a spatially staged scene, geometry comes only from `spatialTimeline` and `camera`.
   * Limit this field to wardrobe, expression, atmosphere, and non-spatial visual styling.
   */
  imagePrompt: string;
  /**
   * Compact LTX instructions using `VISUAL:`, `DIALOGUE:`, `CAMERA:`, and optional `AUDIO:`.
   * Describe one literal action or speaker; omit recaps, metaphors, and decorative foley.
   */
  videoPrompt: string;
  /**
   * Target clip duration. The sum of episode scenes should be about 60 seconds.
   */
  durationSeconds: number;
};

export type ShowEpisode = {
  episodeNumber: number;
  title: string;
  isFree: boolean;
  coinCost: number;
  /** One-sentence conflict promise for this episode. */
  logline: string;
  /** The urgent question that drives every beat until the final reversal. */
  dramaticQuestion: string;
  /** Mute-readable conflict image delivered in the first three seconds. */
  hook: string;
  /** Mid/late episode fact or choice that changes the price of the conflict. */
  reversal: string;
  /** Unresolved final action or revelation that forces the next episode. */
  cliffhanger: string;
  /** Concrete image that episode two would open on. */
  nextEpisodeOpening: string;
  /**
   * Persistent 180-degree-axis map. Change an eyeline only after a new anchor establishes it.
   */
  screenDirection: string;
  /** Ground truth for character and prop state before camera coverage is authored. */
  spatialTimeline: SpatialTimeline;
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
