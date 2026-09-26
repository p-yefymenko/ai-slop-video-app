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
   * What this one object looks like. Qwen draws it. TRELLIS.2 turns that picture
   * into the mesh. No people, camera, or surrounding place.
   * Required on a location that has people. Omitted on an empty location.
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
  /** Physical camera that projects the timeline into the proxy guide. */
  camera: SpatialCamera;
  /**
   * Optional wardrobe, expression, and atmosphere. Omit when the location
   * text and the clay frame are enough. Never restate position,
   * facing, eyeline, framing, or location geometry.
   */
  imagePrompt?: string;
  /**
   * Spoken line and non-spatial performance for generative scenes. Camera and
   * blocking come from the timeline and start/end frames. Omit on silent scenes.
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
