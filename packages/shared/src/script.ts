/**
 * Authoring JSON for `content-pipeline/shows/<id>/script.json`. One file per show.
 *
 * Screenwriting guidance lives in `.cursor/rules/Short-reel-scripts-writer.mdc`.
 * These comments are renderer semantics only. Shared Qwen/LTX templates live in
 * `content-pipeline/prompts.json`, not in show JSON.
 *
 * The script describes a physical world. Every character, landmark, and prop is
 * a generated mesh placed by the timeline. Who appears in a shot, and what is
 * visible, is never written down: previs renders the world from the scene's
 * camera and checks every on-camera claim (performances, expressions, the
 * speaker, sound sources) against that render.
 */

/** Body parts are the regions of a character mesh. Defined once; previs and stills share it. */
export const BODY_PARTS = [
  "hair",
  "face",
  "eyes",
  "neck",
  "torso",
  "arms",
  "hands",
  "legs",
  "feet",
] as const;

export type BodyPart = (typeof BODY_PARTS)[number];

export type ShowCharacter = {
  /** Build and skin. Also how the video prompt points at this person. */
  body: string;
  /**
   * Every other detail of their look, one clause each: age, hair, clothes,
   * scars. The plate and every still draw the whole person from `body` plus
   * these, from the camera's angle; the shot keeps only what it shows.
   */
  attributes: string[];
  /** Standing height in meters. The generated mesh is fitted to it. */
  heightMeters: number;
};

export type Vec3 = [number, number, number];

/** One generated object of a location, placed at its position and fitted into its size. */
export type LocationLandmark = {
  /** Base center, meters: `[X right, Y forward, Z up]`. */
  position: Vec3;
  /** Box `[width X, depth Y, height Z]` in meters the mesh is fitted into. */
  size: Vec3;
  /**
   * What this one object looks like. Drawn as an isolated plate, meshed, and
   * named in a still whenever it is visible. A whole place seen from far away
   * (a citadel, a ship) is one landmark the size of the location.
   */
  appearance: string;
};

/** A movable object. Generated and placed the same way as a landmark. */
export type ShowProp = {
  appearance: string;
  /** Box `[width X, depth Y, height Z]` in meters. */
  size: Vec3;
};

export type StageGeometry = {
  /** Location bounds `[width X, depth Y, height Z]` in meters. The floor spans width and depth. */
  sizeMeters: Vec3;
  landmarks: Record<string, LocationLandmark>;
};

export type LocationBackdrop = {
  /** Empty space above the set, and the flat color painted there. */
  sky: string;
  skyColor: [number, number, number];
  /** Empty space on the ground inside the location, and the flat color painted there. */
  ground: string;
  groundColor: [number, number, number];
  /** Empty space beyond the location, at the horizon, and the flat color painted there. */
  surround: string;
  surroundColor: [number, number, number];
};

/**
 * Persistent audio of a place. Required on every location. LTX always generates
 * audio with the clip, so omitting this lets the model invent music.
 */
export type LocationSoundscape = {
  /** The bed that is always audible here. Present tense. Not music unless the place truly has source music. */
  ambience: string;
  /** Acoustic character of the place: open air, enclosed hall, wet reverb, dry stone. */
  space: string;
};

/**
 * Per-clip music policy. Authors must choose explicitly; there is no default.
 * Distilled LTX runs at CFG 1, so "none" is steered only by a positive phrase.
 */
export type SceneMusic =
  | { kind: "none" }
  | { kind: "described"; description: string };

/** What makes a sound. Previs fails when it is not visible in the start frame. */
export type SoundSource =
  /** A body part, or wardrobe worn on it. The part must be in that character's performance parts. */
  | { characterId: string; part: BodyPart }
  /** A landmark of this scene's location. */
  | { landmarkId: string }
  /** A prop. */
  | { propId: string };

export type SoundEvent = {
  /** The audible action, present tense. */
  text: string;
  source: SoundSource;
};

export type SceneSound = {
  /** Sounds of things visibly happening in this clip, at least one. Dialogue is not an event. */
  events: SoundEvent[];
  /** `present` = full location bed. `faint` = bed lowered (use on speaking scenes). */
  bed: "present" | "faint";
  music: SceneMusic;
};

export type ShowLocation = {
  /** What to paint where the camera sees no mesh. */
  backdrop: LocationBackdrop;
  soundscape: LocationSoundscape;
  spatial: StageGeometry;
};

/**
 * Where one character is from this time until the next keyframe. Every
 * character has a track in every episode, starting at time 0, so the script
 * says where everyone is at every moment. Between two keyframes in the same
 * location the character walks in a straight line; a change of location
 * happens at the later keyframe.
 */
export type CharacterSpatialKeyframe = {
  timeSeconds: number;
  /** `null` is off stage: in no location of this episode. */
  locationId: string | null;
  /** Feet position in location meters. Required in a location. */
  position?: Vec3;
  /** Body rotation around Z. `0` faces +Y; positive turns toward +X. */
  bodyYawDegrees?: number;
  /** Character, landmark, or prop the whole body turns to face. Overrides `bodyYawDegrees`. */
  lookAtId?: string;
};

/** Either held by a character or placed in a location. Every prop has a track starting at 0. */
export type PropSpatialKeyframe =
  | { timeSeconds: number; heldByCharacterId: string; heldInHand: "left" | "right" }
  | { timeSeconds: number; locationId: string; position: Vec3 }
  | { timeSeconds: number; locationId: null };

export type SpatialTimeline = {
  durationSeconds: number;
  /** One track per show character. */
  characterTracks: Record<string, CharacterSpatialKeyframe[]>;
  /** One track per show prop. */
  propTracks: Record<string, PropSpatialKeyframe[]>;
};

export type SpatialCameraKeyframe = {
  timeSeconds: number;
  position: Vec3;
  lookAt: Vec3;
  verticalFovDegrees: number;
  /** Rotation around the view axis. Omit or `0` for a level horizon. */
  rollDegrees?: number;
};

export type SpatialCamera = {
  /** Timed poses on the episode clock. One keyframe holds the camera still. */
  keyframes: SpatialCameraKeyframe[];
};

export type ScriptScene = {
  sceneNumber: number;
  locationId: string;
  /** Why this scene exists in the story, as cause and effect. Not visuals or blocking. */
  storyBeat: string;
  /** Who speaks. Omit on silent scenes. Needs a performance with an expression. */
  speakerId?: string;
  /** Interval on the episode timeline. Scene duration is this interval. */
  timeRangeSeconds: [number, number];
  /** Never inside a mesh. */
  camera: SpatialCamera;
  /**
   * Exactly one entry per character visible at any moment of the shot, no
   * more: previs renders the shot and fails on a missing or extra entry.
   * `{}` when nobody is visible.
   */
  performances: Record<string, ScenePerformance>;
  /** The spoken line. Required exactly when `speakerId` is set. */
  dialogue?: SceneDialogue;
  /** Motion that belongs to no character: fire, smoke, cloth, debris, light. */
  motion?: string;
  sound: SceneSound;
};

export type ScenePerformance = {
  /** Visible motion for the whole take, positive and present tense. */
  action: string;
  /** Body parts the action moves; empty when the action is stillness. Each must be on screen at some moment of the shot. */
  parts: BodyPart[];
  /**
   * The face in the start frame, as visible cues ("jaw set, lips slightly
   * parted"). It is drawn into the still and continued by the clip, so it must
   * match how the take begins: a speaker or someone about to exhale has the
   * lips slightly parted. Required for the speaker. Only on a face visible in
   * the start frame.
   */
  expression?: string;
};

export type SceneDialogue = {
  /** The words spoken, nothing else. */
  line: string;
  /** How it is said: "low, final", "harsh, rising". */
  delivery: string;
};

export type ShowEpisode = {
  episodeNumber: number;
  title: string;
  isFree: boolean;
  coinCost: number;
  spatialTimeline: SpatialTimeline;
  scenes: ScriptScene[];
};

export type ShowScript = {
  id: string;
  title: string;
  characters: Record<string, ShowCharacter>;
  locations: Record<string, ShowLocation>;
  /** Omit when the show has no props. */
  props?: Record<string, ShowProp>;
  episodes: ShowEpisode[];
};
