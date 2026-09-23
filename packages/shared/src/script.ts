/**
 * Authoring JSON for `content-pipeline/shows/<id>/script.json`. One file per show.
 *
 * Screenwriting guidance lives in `.cursor/rules/Short-reel-scripts-writer.mdc`.
 * These comments are renderer semantics only. Shared Qwen/LTX templates live in
 * `content-pipeline/prompts.json`, not in show JSON.
 */

/**
 * Search sent to open-source asset databases. `prefabId` skips this search.
 */
export type AssetNeed = {
  /** Noun phrase a model catalog can match, e.g. "wooden armchair". */
  query: string;
  tags?: string[];
  /** Look used when ranking candidates. Not a second query. */
  style?: string;
};

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

export type ShowCharacter = {
  /**
   * Qwen identity source only; scene stills attach its PNG and never receive this text.
   * One adult's stable physical identity: age, build, skin, face, eyes, distinctive
   * feature, hair. No wardrobe, pose, expression, action, location, or camera.
   */
  promptBlock: string;
  /** Clay mannequin proportions. Identity is still the portrait PNG. */
  proxy?: CharacterProxy;
};

export type Vec3 = [number, number, number];

export type StageLandmark = {
  /**
   * Primitive used when no prefab is resolved, and the category hint for search.
   * The object ID is the landmarks key, not this field.
   */
  kind: "box" | "column" | "pedestal" | "window" | "door" | "seat";
  position: Vec3;
  size: Vec3;
  /** Open-source search that replaces this primitive when a candidate is accepted. */
  need?: AssetNeed;
  /** Library or show prefab to instance. Skips search. */
  prefabId?: string;
};

/**
 * A prop named by `propTracks`. Held and moving props stay on the shot;
 * they are not set dressing.
 */
export type ShowProp = {
  need?: AssetNeed;
  /** Library or show prefab to instance. Skips search. */
  prefabId?: string;
  /** Target size `[width X, depth Y, height Z]` in meters. */
  sizeMeters?: Vec3;
};

export type StageGeometry = {
  /** Interior dimensions `[width X, depth Y, height Z]` in meters. */
  sizeMeters: Vec3;
  landmarks: Record<string, StageLandmark>;
};

export type ShowLocation = {
  /**
   * Materials and light next to the people. Not a viewpoint. Not a space you look down.
   */
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

export type SpatialCameraKeyframe = {
  timeSeconds: number;
  position: Vec3;
  lookAt: Vec3;
  verticalFovDegrees: number;
  /** Rotation around the view axis. Omit or `0` for a level horizon. */
  rollDegrees?: number;
};

export type SpatialCamera = {
  /** Timed poses on the episode clock. One keyframe is a locked-off shot. */
  keyframes: SpatialCameraKeyframe[];
};

export type ScriptScene = {
  sceneNumber: number;
  locationId: string;
  /**
   * Why this shot exists in the story, as cause and effect. Not visuals, blocking,
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
   * Who speaks. Omit on silent shots. Must be in `characterIds`.
   */
  speakerId?: string;
  /**
   * Interval on `spatialTimeline`. Shot duration is this interval; do not also
   * store `durationSeconds`.
   */
  timeRangeSeconds: [number, number];
  /** Physical camera that projects the timeline into the proxy guide. */
  camera: SpatialCamera;
  /**
   * Optional wardrobe, expression, and atmosphere. Omit when the location
   * block and the clay blockout are enough. Never restate position,
   * facing, eyeline, framing, or set geometry.
   */
  imagePrompt?: string;
  /**
   * Spoken line and non-spatial performance for generative shots. Camera and
   * blocking come from the timeline and start/end frames. Omit on silent shots.
   */
  videoPrompt?: string;
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
