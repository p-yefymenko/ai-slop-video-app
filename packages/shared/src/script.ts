/**
 * Authoring JSON for `content-pipeline/shows/<id>/script.json`. One file per show.
 *
 * Screenwriting guidance lives in `.claude/skills/write-script/SKILL.md`.
 * These comments are renderer semantics only. Shared word templates live in
 * `content-pipeline/prompts.json`, not in show JSON.
 *
 * The script describes a physical world. Every character, landmark, and prop is
 * a generated mesh placed by the timeline. Who appears in a shot is never
 * written down: previs renders the world from the scene's camera and checks
 * that everyone with a performance is in the frame.
 */

export type ShowCharacter = {
  /** How the episode's name card titles them the first time they speak on screen ("Master Adam"). Never sent to a model. */
  name?: string;
  /** Build and skin, never clothes. The only words the video prompt says about this person. */
  body: string;
  /**
   * Every other detail of their look, one clause each: age, hair, clothes,
   * scars. Only the plate and the reference photo are drawn from them; the
   * video model gets the look from the reference photo, never in words.
   */
  attributes: string[];
  /** Standing height in meters. The generated mesh is fitted to it. */
  heightMeters: number;
  /**
   * How they sound, as a voice designer would brief it: sex, age, pitch,
   * texture, pace, accent ("an old man's voice, low and gravelly, slow and
   * measured"). Their `voice.wav` is made from it, and every line of theirs is
   * spoken in that voice. Required for anyone with a line.
   */
  voice?: string;
  effects?: Effect[];
};

export type Vec3 = [number, number, number];

/**
 * Fire, smoke, steam, sparks, glow: something that is not a solid surface. It
 * is never part of a plate, a mesh, or a depth guide; the video prompt names it
 * when the shot shows its box, and the video model draws and animates it. An
 * `appearance` describes only the solid object.
 */
export type Effect = {
  /** What it looks like, e.g. "white-gold flames rising from the bowl". */
  appearance: string;
  /** Box `[width X, depth Y, height Z]` in meters it fills. */
  size: Vec3;
  /** Base center of the box, meters, relative to the object's base center and turned with it. */
  offset: Vec3;
};

/** One generated object of a location, placed at its position and fitted into its size. */
export type LocationLandmark = {
  /** Base center, meters: `[X right, Y forward, Z up]`. */
  position: Vec3;
  /** Box `[width X, depth Y, height Z]` in meters the mesh is fitted into. */
  size: Vec3;
  /**
   * What this one solid object looks like, without fire, smoke, or other
   * effects (those are `effects`). Drawn as an isolated plate, meshed, and drawn
   * alone into every still that shows any of it. A whole place seen from far
   * away (a citadel, a ship) is one landmark the size of the location.
   */
  appearance: string;
  effects?: Effect[];
};

/**
 * A movable object. Generated and placed the same way as a landmark. Something a
 * character wears for the whole show is one of their attributes instead, never both.
 */
export type ShowProp = {
  appearance: string;
  /** Box `[width X, depth Y, height Z]` in meters. */
  size: Vec3;
  effects?: Effect[];
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
  /** Acoustic character of the place: open air, enclosed hall, wet reverb, dry stone. Sound only, never what the place looks like: the clip prompt is shared by picture and sound, and a place described here makes the video cut to a wide of it. */
  space: string;
};

/**
 * Per-clip music policy. Authors must choose explicitly; there is no default.
 * Distilled LTX runs at CFG 1, so "none" is steered only by a positive phrase.
 */
export type SceneMusic =
  | { kind: "none" }
  | { kind: "described"; description: string };

/** What makes a sound. */
export type SoundSource =
  /** A character. */
  | { characterId: string }
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

/**
 * One look for everything in a location, said once so its objects cannot drift
 * apart: each object is drawn alone, and only these shared words make them
 * belong to the same place.
 */
export type LocationLook = {
  /** The light on everything in a shot here, e.g. "cold blue storm light from above, wet sheen". */
  light: string;
};

export type ShowLocation = {
  look: LocationLook;
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

/** Placed in a location, or gone. Every prop has a track starting at 0. */
export type PropSpatialKeyframe =
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

export const SHOT_TYPES = ["single", "group", "overShoulder", "insert", "establishing", "action"] as const;
export type ShotType = (typeof SHOT_TYPES)[number];
export const SHOT_SIZES = ["ecu", "cu", "mcu", "medium", "full", "wide"] as const;
export type ShotSize = (typeof SHOT_SIZES)[number];

/**
 * What the shot shows. For every shot of people the camera is computed from the
 * blocking (content-pipeline/scripts/shots.py), so it frames its subjects by
 * construction; only `insert`, `establishing` and `action` may set `camera` themselves.
 *
 * - `single`: one character (`subjects: [id]`), `size` ecu|cu|mcu|medium|full.
 * - `group`: two or more characters, `size` mcu|medium|full|wide.
 * - `overShoulder`: `subjects: [seen]` over the near shoulder of `over`, `size` cu|mcu|medium.
 * - `insert`: one detail (hands, a weapon); needs `camera`.
 * - `establishing`: the place; needs `camera`.
 * - `action`: characters in motion, `size` full|wide, or its own `camera`.
 */
export type SceneShot = {
  type: ShotType;
  subjects?: string[];
  over?: string;
  size?: ShotSize;
  /** eye (default) | low (looking up: power) | high (looking down: defeat). */
  angle?: "eye" | "low" | "high";
  /** front (default, along the gaze) | left | right (a three-quarter view). */
  side?: "front" | "left" | "right";
  /** static (default) | pushIn | pullOut | tiltUp (from the feet up to the framing). */
  move?: "static" | "pushIn" | "pullOut" | "tiltUp";
};

export type ScriptScene = {
  sceneNumber: number;
  locationId: string;
  /** Why this scene exists in the story, as cause and effect. Not visuals or blocking. */
  storyBeat: string;
  /** Interval on the episode timeline. Scene duration is this interval. */
  timeRangeSeconds: [number, number];
  /** What the shot shows; its camera follows from it. */
  shot: SceneShot;
  /** Only for `establishing` (required) and `action` (optional). Never inside a mesh. */
  camera?: SpatialCamera;
  /** How an own camera that moves moves, in words for the video model ("flies low over the treetops toward the cliff edge"). Required exactly when it moves. */
  cameraMotion?: string;
  /**
   * Who acts in the shot. Everyone here is said to the video model with their
   * whole look and their action; anyone else in the frame is not named (the
   * shot's `over` is the shoulder in front). Previs fails when someone here is
   * never in the frame. `{}` when nobody acts.
   */
  performances: Record<string, ScenePerformance>;
  /**
   * What is said, in order. A speaker with a performance speaks on screen (and
   * needs an expression); a speaker without one is a voice from off screen.
   */
  lines?: SceneLine[];
  /** Motion that belongs to no character: fire, smoke, cloth, debris, light. */
  motion?: string;
  sound: SceneSound;
};

export type ScenePerformance = {
  /**
   * Visible motion for the whole take, positive and present tense, of this
   * person's own body only ("the jaw tightens", "shoves {target} with both
   * hands"). No pronouns and no names: the video model sees only the frame and
   * invents whoever a word points at. Another person is only ever `{target}`;
   * gaze comes from lookAtId. Never describe clothes, props, or landmarks: the
   * words cannot know which part of them the frame shows.
   */
  action: string;
  /**
   * Who the action is aimed at, written as `{target}` in the action: a
   * character id. They must be in the start frame; the clip prompt names them
   * by their `body`.
   */
  target?: string;
  /**
   * The face in the start frame, as visible cues ("jaw set, lips slightly
   * parted"). It is drawn into the still and continued by the clip, so it must
   * match how the take begins: a speaker or someone about to exhale has the
   * lips slightly parted. Required for the speaker. Only on a face visible in
   * the start frame.
   */
  expression?: string;
};

export type SceneLine = {
  /** Who says it: on screen when they have a performance in this scene, else off screen. */
  speakerId: string;
  /** The words spoken, nothing else. */
  line: string;
  /** How it is said: "low, final", "harsh, rising". */
  delivery: string;
  /** Seconds into the scene when the line starts; omit to start when the previous line ends. */
  startSeconds?: number;
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
