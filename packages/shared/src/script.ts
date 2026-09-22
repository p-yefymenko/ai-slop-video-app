/**
 * Authoring JSON for `content-pipeline/scripts_input/<id>.json`. One file per show.
 *
 * Screenwriting guidance lives in `.cursor/rules/Short-reel-scripts-writer.mdc`.
 * These comments are renderer semantics only. Shared Qwen/LTX templates live in
 * `content-pipeline/prompts.json`, not in show JSON.
 */

export type ShowCharacter = {
  /**
   * Qwen identity source only; scene stills attach its PNG and never receive this text.
   * One adult's stable physical identity: age, build, skin, face, eyes, distinctive
   * feature, hair. No wardrobe, pose, expression, action, location, or camera.
   */
  promptBlock: string;
};

export type Vec3 = [number, number, number];

export type StageLandmark = {
  /** Stable object ID used by blocking. */
  kind: "box" | "column" | "pedestal" | "window" | "door" | "seat";
  position: Vec3;
  size: Vec3;
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

export type SpatialCamera = {
  position: Vec3;
  lookAt: Vec3;
  verticalFovDegrees: number;
  endPosition?: Vec3;
  endLookAt?: Vec3;
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
   * Visible people, at most two. Empty for environments and prop inserts.
   * Order is Qwen Picture 1, then Picture 2.
   */
  characterIds: string[];
  /**
   * On-camera speaker. Omit on silent shots. Exactly one visible person when set;
   * dialogue is never authored on a two-shot.
   */
  speakerId?: string;
  /**
   * Override only. Default is `generative` when `speakerId` is set, otherwise
   * `cameraOnly`. Use `generative` for one visible silent action that a camera
   * move cannot do.
   */
  motionMode?: "cameraOnly" | "generative";
  /**
   * Earlier master whose rendered PNG supplies set, light, and axis.
   * Required on dialogue singles and in-set prop inserts.
   */
  coverageReferenceSceneNumber?: number;
  /**
   * Prop or landmark ID centered by an insert. Combine with
   * `coverageReferenceSceneNumber`; do not reinvent the prop from a proxy box.
   */
  focusTargetId?: string;
  /**
   * Interval on `spatialTimeline`. Shot duration is this interval; do not also
   * store `durationSeconds`.
   */
  timeRangeSeconds: [number, number];
  /** Physical camera that projects the timeline into the proxy guide. */
  camera: SpatialCamera;
  /**
   * Wardrobe, expression, and atmosphere only. Never restate position, facing,
   * eyeline, framing, or set geometry.
   */
  imagePrompt: string;
  /**
   * Spoken line and non-spatial performance for generative shots. Camera and
   * blocking are compiled from the timeline. Omit on `cameraOnly` shots.
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
  episodes: ShowEpisode[];
};
