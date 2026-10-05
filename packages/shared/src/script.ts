/**
 * Authoring JSON for `content-pipeline/shows/<id>/script.json`. One file per show.
 *
 * Screenwriting guidance lives in `.cursor/rules/Short-reel-scripts-writer.mdc`.
 * These comments are renderer semantics only. Shared Qwen/LTX templates live in
 * `content-pipeline/prompts.json`, not in show JSON.
 */

export type CharacterProxy = {
  /**
   * Standing height in meters. The clay mannequin scales to this.
   * Qwen restyles the body; identity faces are painted afterward.
   * Omit to use 1.72m, the blockout's standing head height.
   */
  heightMeters: number;
  /** Limb thickness of the clay mannequin. */
  build: "slim" | "average" | "broad";
};

/** Body parts an attribute may depend on. Defined once; previs and stills share it. */
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

export type CharacterAttribute = {
  /** Words sent to Qwen when any listed part is tall enough in this camera. */
  text: string;
  /** Parts this clause names. Empty is invalid. */
  parts: BodyPart[];
};

export type CharacterPartRequirement = {
  characterId: string;
  part: BodyPart;
};

export type ShowCharacter = {
  /**
   * Always sent: build, skin. Plates, portraits, and every still receive
   * this text. Age and other appearance that lives on a body part goes in
   * `attributes`.
   */
  body: string;
  /**
   * Tagged clauses. A still sends a clause only when any of its parts is at
   * least `partMinPixelHeight` tall in that camera. Plates and portraits send
   * every clause.
   */
  attributes: CharacterAttribute[];
  /** Clay mannequin proportions. Identity is still the portrait PNG. */
  proxy?: CharacterProxy;
};

export type Vec3 = [number, number, number];

export type LocationLandmark = {
  /**
   * Meters: `[X right, Y forward, Z up]`.
   * A location with people authors this. A location with no people may omit it
   * until `pnpm run content:landmarks` fills it from the whole-location mesh.
   */
  position?: Vec3;
  /**
   * Target size `[width X, depth Y, height Z]` in meters.
   * Required on a location that has people. The landmark mesh is fitted into this box.
   */
  size?: Vec3;
  /**
   * What this one object looks like.
   * On a location with people, Qwen draws it and TRELLIS.2 turns that picture
   * into the mesh. No people, camera, or surrounding place.
   * On an empty location it is only words for the still. That landmark stays a
   * point in the one location mesh.
   */
  appearance?: string;
};

/**
 * A prop named by `propTracks`. Held and moving props stay on the scene;
 * they are not part of the location mesh.
 */
export type ShowProp = {
  appearance: string;
  /** Target size `[width X, depth Y, height Z]` in meters. */
  sizeMeters?: Vec3;
};

export type StageGeometry = {
  /** Location size `[width X, depth Y, height Z]` in meters. */
  sizeMeters: Vec3;
  /**
   * Named objects. A location with people generates one mesh per landmark at `size`.
   * A location with no people is one mesh, and these are optional points in it.
   */
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
  /**
   * The bed that is always audible here: wind, fire, crowd, machines.
   * Present tense. Not music unless this place truly has source music.
   */
  ambience: string;
  /**
   * Acoustic character of the place: open air, enclosed hall, wet reverb,
   * dry stone, little echo.
   */
  space: string;
};

/**
 * Per-clip music policy. Authors must choose explicitly; there is no default.
 * Distilled LTX runs at CFG 1, so "none" is steered only by a positive phrase
 * from prompts.json, never by a negative prompt.
 */
export type SceneMusic =
  | { kind: "none" }
  | { kind: "described"; description: string };

/**
 * Per-scene sound direction. Required on every scene. Composed into the LTX
 * prompt after visuals and after any videoPrompt / LIP SYNC text.
 */
export type SceneSound = {
  /**
   * Sounds of things visibly happening in this clip. Each event needs a cause
   * visible in the start still. Add to the location bed; never restate or
   * contradict it. On speaking scenes, describe only non-speech sounds;
   * dialogue stays in videoPrompt.
   */
  events: string;
  /**
   * How loud the location bed sits under this scene.
   * `present` = full bed. `faint` = bed lowered under the scene (use on
   * speaking scenes so dialogue is not buried).
   */
  bed: "present" | "faint";
  music: SceneMusic;
};

export type ShowLocation = {
  /**
   * Materials and light next to the people. Not a viewpoint. Not a space you look down.
   */
  promptBlock: string;
  /** What to paint where the camera sees no asset. */
  backdrop: LocationBackdrop;
  /** Persistent ambience and acoustic space for every scene here. */
  soundscape: LocationSoundscape;
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
  /**
   * Why this scene exists in the story, as cause and effect. Not visuals, blocking,
   * or camera; those come from the timeline.
   */
  storyBeat: string;
  /**
   * Visible people. Empty for environments and prop inserts.
   * The clay blockout is the picture Qwen restyles, so the camera stays.
   * The first two identities are painted onto those faces afterward.
   */
  characterIds: string[];
  /**
   * Who speaks. Omit on silent scenes. Must be in `characterIds`.
   */
  speakerId?: string;
  /**
   * Interval on `spatialTimeline`. Scene duration is this interval; do not also
   * store `durationSeconds`.
   */
  timeRangeSeconds: [number, number];
  /**
   * Physical camera that projects the timeline into the proxy guide.
   * Stay at least 1.5m from every character and landmark volume.
   */
  camera: SpatialCamera;
  /**
   * Optional wardrobe, expression, and atmosphere. Omit when the location
   * text and the clay frame are enough. Never restate position,
   * facing, eyeline, framing, or location geometry.
   */
  imagePrompt?: string;
  /**
   * Spoken line and non-spatial performance for generative scenes. Camera and
   * blocking come from the timeline and start/end frames. Required when
   * characterIds is not empty, silent scenes included: LTX invents motion it
   * is not given.
   * Dialogue stays here; non-speech sounds go in `sound.events`.
   */
  videoPrompt?: string;
  /**
   * Required sound direction for this clip. Location bed + events + music
   * policy are composed into the LTX prompt; LTX always generates audio.
   */
  sound: SceneSound;
  /**
   * Previs fails the scene when any of these parts has no visible pixels, or
   * is under 5px wide at LTX's 448px output, on a playblast frame.
   */
  requiresParts?: CharacterPartRequirement[];
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
  /**
   * Props referenced by `propTracks`. Omit when the show has no props.
   * Every prop-track id must be a key here.
   */
  props?: Record<string, ShowProp>;
  episodes: ShowEpisode[];
};
