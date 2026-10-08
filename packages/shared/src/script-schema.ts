/**
 * Authoring check for ShowScript JSON: shape and references only. Anything that
 * needs the meshes and the camera (what is visible, camera distance to a mesh)
 * is checked by content:previs against the rendered world.
 */

import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { z } from "zod";

import {
  BODY_PARTS,
  type CharacterSpatialKeyframe,
  type PropSpatialKeyframe,
  type ScriptScene,
  type ShowEpisode,
  type ShowLocation,
  type ShowScript,
  type SpatialCameraKeyframe,
} from "./script";

const BODY_PART_SET = new Set<string>(BODY_PARTS);

export type ScriptIssue = {
  path: string;
  message: string;
  /** Defaults to error. Warnings do not fail `content:validate`. */
  severity?: "error" | "warning";
};

export type ScriptSource = {
  /** Value `id` must equal: filename stem, or the parent folder of script.json. */
  showId: string;
  /** Phrase used in the mismatch error, e.g. "the filename stem". */
  showIdLabel: string;
};

export type ParseResult =
  | { ok: true; script: ShowScript; issues: ScriptIssue[] }
  | { ok: false; issues: ScriptIssue[] };

/**
 * Authored camera travel+rotation warn threshold (Gate 3b / docs/ltx-depth-gate.md).
 * Scene 01 scored 39.47 with follow 0.55 (fail); scenes 11/06 passed with travel ≤4.04
 * and follow ≥0.93. Threshold sits in that gap. Scene 04 scored only 2.44 with weak
 * follow 0.686 — not explained by this score.
 */
export const CAMERA_TRAVEL_WARN_THRESHOLD = 20;

const SNAKE_ID = /^[a-z][a-z0-9]*(_[a-z0-9]+)*$/;
const SHOW_ID = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

const BED_LEVELS = ["present", "faint"] as const;

const PROMPTS_PATH = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../../content-pipeline/prompts.json",
);

type SoundLintConfig = {
  emptyEventsMaxWords: number;
  musicConflictWords: string[];
};

function loadSoundLint(): SoundLintConfig {
  if (!existsSync(PROMPTS_PATH)) {
    return { emptyEventsMaxWords: 2, musicConflictWords: [] };
  }
  try {
    const prompts = JSON.parse(readFileSync(PROMPTS_PATH, "utf8")) as {
      soundLint?: { emptyEventsMaxWords?: unknown; musicConflictWords?: unknown };
    };
    const lint = prompts.soundLint;
    const maxWords =
      typeof lint?.emptyEventsMaxWords === "number" && Number.isFinite(lint.emptyEventsMaxWords)
        ? Math.max(0, Math.floor(lint.emptyEventsMaxWords))
        : 2;
    const words = Array.isArray(lint?.musicConflictWords)
      ? lint.musicConflictWords
          .filter((word): word is string => typeof word === "string" && word.trim().length > 0)
          .map((word) => word.trim().toLowerCase())
      : [];
    return { emptyEventsMaxWords: maxWords, musicConflictWords: words };
  } catch {
    return { emptyEventsMaxWords: 2, musicConflictWords: [] };
  }
}

function wordCount(value: string): number {
  return value
    .trim()
    .split(/\s+/)
    .filter(Boolean).length;
}

function containsMusicWord(text: string, words: string[]): string | undefined {
  const lower = text.toLowerCase();
  return words.find((word) => {
    const pattern = new RegExp(`\\b${word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`, "i");
    return pattern.test(lower);
  });
}

function text(label: string) {
  return z
    .string({
      required_error: `${label} is required`,
      invalid_type_error: `${label} must be a string`,
    })
    .trim()
    .min(1, `${label} is required`);
}

function finiteNumber(label: string) {
  return z
    .number({
      required_error: `${label} is required`,
      invalid_type_error: `${label} must be a number`,
    })
    .refine((value) => Number.isFinite(value), `${label} must be a finite number`);
}

function vec3(label: string, positive = false) {
  const axis = (name: "x" | "y" | "z") => {
    const schema = finiteNumber(`${label} ${name}`);
    return positive
      ? schema.refine((value) => value > 0, `${label} ${name} must be greater than 0`)
      : schema;
  };
  return z.tuple([axis("x"), axis("y"), axis("z")], {
    required_error: `${label} is required`,
    invalid_type_error: `${label} must be [x, y, z] in meters`,
  });
}

const effectSchema = z
  .object({
    appearance: text("appearance"),
    size: vec3("size", true),
    offset: vec3("offset"),
  })
  .strict();

const effectsSchema = z.array(effectSchema, { invalid_type_error: "effects must be an array" }).optional();

const locationLandmarkSchema = z
  .object({
    position: vec3("position"),
    size: vec3("size", true),
    appearance: text("appearance"),
    effects: effectsSchema,
  })
  .strict();

const stageGeometrySchema = z
  .object({
    sizeMeters: vec3("sizeMeters", true),
    landmarks: z.record(z.string(), locationLandmarkSchema, {
      required_error: "landmarks is required",
      invalid_type_error: "landmarks must be an object",
    }),
  })
  .strict();

const showCharacterSchema = z
  .object({
    body: text("body"),
    attributes: z.array(text("attribute"), {
      required_error: "attributes is required",
      invalid_type_error: "attributes must be an array",
    }),
    heightMeters: finiteNumber("heightMeters").refine(
      (value) => value >= 1.2 && value <= 2.4,
      "heightMeters must be a standing height from 1.2 to 2.4 meters",
    ),
    effects: effectsSchema,
  })
  .strict();

const showPropSchema = z
  .object({
    appearance: text("appearance"),
    size: vec3("size", true),
    effects: effectsSchema,
  })
  .strict();

const characterKeyframeSchema = z
  .object({
    timeSeconds: finiteNumber("timeSeconds"),
    locationId: text("locationId").nullable(),
    position: vec3("position").optional(),
    bodyYawDegrees: finiteNumber("bodyYawDegrees").optional(),
    lookAtId: text("lookAtId").optional(),
  })
  .strict();

const propKeyframeSchema = z.union(
  [
    z
      .object({
        timeSeconds: finiteNumber("timeSeconds"),
        heldByCharacterId: text("heldByCharacterId"),
        heldInHand: z.enum(["left", "right"]),
      })
      .strict(),
    z
      .object({ timeSeconds: finiteNumber("timeSeconds"), locationId: text("locationId"), position: vec3("position") })
      .strict(),
    z.object({ timeSeconds: finiteNumber("timeSeconds"), locationId: z.null() }).strict(),
  ],
  {
    errorMap: () => ({
      message:
        "prop keyframe must be { timeSeconds, heldByCharacterId, heldInHand }, { timeSeconds, locationId, position }, or { timeSeconds, locationId: null }",
    }),
  },
);

const cameraKeyframeSchema = z
  .object({
    timeSeconds: finiteNumber("timeSeconds"),
    position: vec3("position"),
    lookAt: vec3("lookAt"),
    verticalFovDegrees: finiteNumber("verticalFovDegrees").refine(
      (value) => value > 0 && value < 180,
      "verticalFovDegrees must be between 0 and 180",
    ),
    rollDegrees: finiteNumber("rollDegrees").optional(),
  })
  .strict();

const locationSoundscapeSchema = z
  .object({
    ambience: text("soundscape.ambience").describe(
      "Persistent bed always audible at this location. Present tense. Not music unless the place truly has source music.",
    ),
    space: text("soundscape.space").describe(
      "Acoustic character: open air, enclosed hall, reverb, dry stone, little echo.",
    ),
  })
  .strict()
  .describe(
    "Required location soundscape. ambience is the bed; space is the acoustic character. Both non-empty.",
  );

const sceneMusicSchema = z
  .discriminatedUnion("kind", [
    z
      .object({
        kind: z.literal("none"),
      })
      .strict()
      .describe('No music in this clip. Authors must choose kind "none" explicitly; there is no default.'),
    z
      .object({
        kind: z.literal("described"),
        description: text("sound.music.description").describe(
          "Positive description of diegetic or scored music that should play in this clip.",
        ),
      })
      .strict()
      .describe("Music is intentionally present; describe it in present tense."),
  ])
  .describe('Music policy: { kind: "none" } or { kind: "described", description }. No default.');

const soundSourceSchema = z
  .union(
    [
      z
        .object({ characterId: text("source.characterId"), part: text("source.part") })
        .strict(),
      z.object({ landmarkId: text("source.landmarkId") }).strict(),
      z.object({ propId: text("source.propId") }).strict(),
    ],
    {
      errorMap: () => ({
        message: "source must be { characterId, part }, { landmarkId }, or { propId }",
      }),
    },
  )
  .describe("What makes the sound. Previs fails when it is not on screen in the start frame.");

const soundEventSchema = z
  .object({
    text: text("sound event text"),
    source: soundSourceSchema,
  })
  .strict();

const performanceSchema = z
  .object({
    action: text("performance action").describe(
      "Visible motion of this person for the whole take, positive and present tense.",
    ),
    parts: z
      .array(text("performance part"), {
        required_error: "parts is required",
        invalid_type_error: "parts must be an array",
      }),
    expression: text("performance expression").optional(),
  })
  .strict();

const dialogueSchema = z
  .object({
    line: text("dialogue.line"),
    delivery: text("dialogue.delivery"),
  })
  .strict();

const sceneSoundSchema = z
  .object({
    events: z
      .array(soundEventSchema, {
        required_error: "sound.events is required",
        invalid_type_error: "sound.events must be an array",
      })
      .min(1, "sound.events needs at least one event")
      .describe(
        "Sounds of things visibly happening in this clip, each with its on-screen source. Add to the location bed; never restate or contradict it. Dialogue is not an event.",
      ),
    bed: z
      .enum(BED_LEVELS, {
        required_error: "sound.bed is required",
        invalid_type_error: `sound.bed must be ${BED_LEVELS.join(" or ")}`,
      })
      .describe(
        'How loud the location bed sits. "present" = full bed. "faint" = bed lowered under the scene (preferred when speakerId is set).',
      ),
    music: sceneMusicSchema,
  })
  .strict()
  .describe(
    "Required per-scene sound. Composed into the LTX prompt after visuals. Distilled CFG 1: steer only with positive text.",
  );

const sceneSchema = z
  .object({
    sceneNumber: z
      .number({
        required_error: "sceneNumber is required",
        invalid_type_error: "sceneNumber must be a number",
      })
      .int("sceneNumber must be an integer")
      .positive("sceneNumber must be greater than 0"),
    locationId: text("locationId"),
    storyBeat: text("storyBeat"),
    speakerId: text("speakerId").optional(),
    timeRangeSeconds: z.tuple([finiteNumber("timeRangeSeconds start"), finiteNumber("timeRangeSeconds end")], {
      required_error: "timeRangeSeconds is required",
      invalid_type_error: "timeRangeSeconds must be [start, end]",
    }),
    camera: z
      .object({
        keyframes: z
          .array(cameraKeyframeSchema, {
            required_error: "camera.keyframes is required",
            invalid_type_error: "camera.keyframes must be an array",
          })
          .min(1, "camera.keyframes needs at least one pose"),
      })
      .strict(),
    performances: z.record(performanceSchema, {
      required_error: "performances is required; use {} when no one is on camera",
      invalid_type_error: "performances must be an object keyed by characterId",
    }),
    dialogue: dialogueSchema.optional(),
    motion: text("motion").optional(),
    sound: sceneSoundSchema,
  })
  .strict();

const episodeSchema = z
  .object({
    episodeNumber: z
      .number({
        required_error: "episodeNumber is required",
        invalid_type_error: "episodeNumber must be a number",
      })
      .int("episodeNumber must be an integer")
      .positive("episodeNumber must be greater than 0"),
    title: text("title"),
    isFree: z.boolean({
      required_error: "isFree is required",
      invalid_type_error: "isFree must be true or false",
    }),
    coinCost: z
      .number({
        required_error: "coinCost is required",
        invalid_type_error: "coinCost must be a number",
      })
      .int("coinCost must be an integer")
      .nonnegative("coinCost must be 0 or greater"),
    spatialTimeline: z
      .object({
        durationSeconds: finiteNumber("durationSeconds").refine(
          (value) => value > 0,
          "durationSeconds must be greater than 0",
        ),
        characterTracks: z.record(z.string(), z.array(characterKeyframeSchema).min(1, "track cannot be empty"), {
          required_error: "characterTracks is required",
          invalid_type_error: "characterTracks must be an object",
        }),
        propTracks: z.record(z.string(), z.array(propKeyframeSchema).min(1, "track cannot be empty"), {
          required_error: "propTracks is required",
          invalid_type_error: "propTracks must be an object",
        }),
      })
      .strict(),
    scenes: z
      .array(sceneSchema, {
        required_error: "scenes is required",
        invalid_type_error: "scenes must be an array",
      })
      .min(1, "scenes needs at least one shot"),
  })
  .strict();

const showScriptSchema = z
  .object({
    id: text("id").regex(SHOW_ID, "id must be kebab-case, for example the-iron-bride"),
    title: text("title"),
    characters: z
      .record(z.string(), showCharacterSchema, {
        required_error: "characters is required",
        invalid_type_error: "characters must be an object",
      })
      .refine((value) => Object.keys(value).length > 0, "characters needs at least one character"),
    locations: z
      .record(
        z.string(),
        z
          .object({
            look: z
              .object({ materials: text("look.materials"), light: text("look.light") })
              .strict(),
            backdrop: z
              .object({
                sky: text("backdrop.sky"),
                skyColor: z.tuple([z.number(), z.number(), z.number()]),
                ground: text("backdrop.ground"),
                groundColor: z.tuple([z.number(), z.number(), z.number()]),
                surround: text("backdrop.surround"),
                surroundColor: z.tuple([z.number(), z.number(), z.number()]),
              })
              .strict(),
            soundscape: locationSoundscapeSchema,
            spatial: stageGeometrySchema,
          })
          .strict(),
        {
          required_error: "locations is required",
          invalid_type_error: "locations must be an object",
        },
      )
      .refine((value) => Object.keys(value).length > 0, "locations needs at least one location"),
    props: z
      .record(z.string(), showPropSchema, {
        invalid_type_error: "props must be an object",
      })
      .optional(),
    episodes: z
      .array(episodeSchema, {
        required_error: "episodes is required",
        invalid_type_error: "episodes must be an array",
      })
      .min(1, "episodes needs at least one episode"),
  })
  .strict();

function formatPath(segments: Array<string | number>): string {
  let out = "";
  for (const part of segments) {
    if (typeof part === "number") {
      out += `[${part}]`;
    } else if (!out) {
      out = part;
    } else {
      out += `.${part}`;
    }
  }
  return out || "(root)";
}

function zodIssues(error: z.ZodError): ScriptIssue[] {
  return error.issues.map((issue) => ({
    path: formatPath(issue.path),
    message: issue.message,
  }));
}

function samePoint(a: [number, number, number], b: [number, number, number]): boolean {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  const dz = a[2] - b[2];
  return dx * dx + dy * dy + dz * dz < 1e-12;
}

function outsideStage(position: [number, number, number], location: ShowLocation): boolean {
  const [width, depth] = location.spatial.sizeMeters;
  return Math.abs(position[0]) > width / 2 || Math.abs(position[1]) > depth / 2 || position[2] < 0;
}

function checkPartId(issues: ScriptIssue[], path: string, part: string) {
  if (!BODY_PART_SET.has(part)) {
    issues.push({ path, message: `unknown part id ${JSON.stringify(part)}` });
  }
}

function checkSnakeId(issues: ScriptIssue[], path: string, id: string, label: string) {
  if (!SNAKE_ID.test(id)) {
    issues.push({ path, message: `${label} ${JSON.stringify(id)} must be lowercase snake_case` });
  }
}

/** Keyframes increase, the first is at 0, and none is past the episode end. */
function checkTrackTimes(
  issues: ScriptIssue[],
  path: string,
  frames: Array<{ timeSeconds: number }>,
  durationSeconds: number,
) {
  if (frames.length && frames[0].timeSeconds !== 0) {
    issues.push({
      path: `${path}[0].timeSeconds`,
      message: "a track starts at 0, so it says where this is for the whole episode",
    });
  }
  frames.forEach((frame, index) => {
    if (index > 0 && frame.timeSeconds <= frames[index - 1].timeSeconds) {
      issues.push({ path: `${path}[${index}].timeSeconds`, message: "keyframe times must increase" });
    }
    if (frame.timeSeconds < 0 || frame.timeSeconds > durationSeconds + 1e-6) {
      issues.push({
        path: `${path}[${index}].timeSeconds`,
        message: `time ${frame.timeSeconds} is outside 0..${durationSeconds}`,
      });
    }
  });
}

function checkCharacterFrame(issues: ScriptIssue[], path: string, frame: CharacterSpatialKeyframe, show: ShowScript) {
  if (frame.locationId === null) {
    for (const key of ["position", "bodyYawDegrees", "lookAtId"] as const) {
      if (frame[key] !== undefined) {
        issues.push({ path: `${path}.${key}`, message: "an off-stage keyframe (locationId null) has no pose" });
      }
    }
    return;
  }
  const location = show.locations[frame.locationId];
  if (!location) {
    issues.push({ path: `${path}.locationId`, message: `unknown location ${JSON.stringify(frame.locationId)}` });
    return;
  }
  if (!frame.position) {
    issues.push({ path: `${path}.position`, message: "position is required in a location" });
  } else if (outsideStage(frame.position, location)) {
    issues.push({
      path: `${path}.position`,
      message: `position ${JSON.stringify(frame.position)} is outside ${frame.locationId}`,
    });
  }
  if (frame.bodyYawDegrees === undefined) {
    issues.push({ path: `${path}.bodyYawDegrees`, message: "bodyYawDegrees is required in a location" });
  }
  const target = frame.lookAtId;
  if (
    target &&
    !(target in show.characters) &&
    !(target in (show.props ?? {})) &&
    !(target in location.spatial.landmarks)
  ) {
    issues.push({
      path: `${path}.lookAtId`,
      message: `${JSON.stringify(target)} must be a character, a prop, or a landmark in ${JSON.stringify(frame.locationId)}`,
    });
  }
}

function checkPropFrame(issues: ScriptIssue[], path: string, frame: PropSpatialKeyframe, show: ShowScript) {
  if ("heldByCharacterId" in frame) {
    if (!(frame.heldByCharacterId in show.characters)) {
      issues.push({
        path: `${path}.heldByCharacterId`,
        message: `unknown character ${JSON.stringify(frame.heldByCharacterId)}`,
      });
    }
    return;
  }
  if (frame.locationId === null) {
    return;
  }
  const location = show.locations[frame.locationId];
  if (!location) {
    issues.push({ path: `${path}.locationId`, message: `unknown location ${JSON.stringify(frame.locationId)}` });
  } else if ("position" in frame && outsideStage(frame.position, location)) {
    issues.push({
      path: `${path}.position`,
      message: `position ${JSON.stringify(frame.position)} is outside ${frame.locationId}`,
    });
  }
}

function checkCameraFrame(
  issues: ScriptIssue[],
  path: string,
  frame: SpatialCameraKeyframe,
  start: number,
  finish: number,
) {
  if (frame.timeSeconds < start - 1e-6 || frame.timeSeconds > finish + 1e-6) {
    issues.push({
      path: `${path}.timeSeconds`,
      message: `time ${frame.timeSeconds} is outside the shot range ${start}..${finish}`,
    });
  }
  if (samePoint(frame.position, frame.lookAt)) {
    issues.push({ path: `${path}.lookAt`, message: "lookAt must differ from position" });
  }
}

function vecLength(v: readonly [number, number, number]): number {
  return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

function vecSub(a: readonly [number, number, number], b: readonly [number, number, number]): [number, number, number] {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

/** Authored camera travel and rotation across consecutive keyframes. */
export function cameraTravelScore(scene: ScriptScene): number {
  const keyframes = scene.camera.keyframes;
  let travel = 0;
  for (let i = 0; i < keyframes.length - 1; i++) {
    const a = keyframes[i]!;
    const b = keyframes[i + 1]!;
    travel += vecLength(vecSub(b.position, a.position));
    travel += 0.5 * vecLength(vecSub(b.lookAt, a.lookAt));
    travel += 0.02 * Math.abs(b.verticalFovDegrees - a.verticalFovDegrees);
    let roll = ((b.rollDegrees ?? 0) - (a.rollDegrees ?? 0) + 180) % 360;
    if (roll < 0) {
      roll += 360;
    }
    travel += 0.02 * Math.abs(roll - 180);
  }
  return travel;
}

function cameraTravelWarnMessage(score: number, threshold: number): string {
  return (
    `high camera travel+rotation score ${score.toFixed(2)} exceeds warn threshold ${threshold} ` +
    `(Gate 3b: scene 01 scored 39.47 with follow 0.55; scenes 11/06 passed with travel ≤4.04 and ` +
    `follow ≥0.93; scene 04 scored only 2.44 with weak follow 0.686 — not explained by this score)`
  );
}

// Words for something that is not a solid surface. A mesh built from them would
// turn fire into a solid shape in every depth guide; they belong in `effects`.
const EFFECT_WORDS = /\b(fire\w*|flames?|flaming|smoke|smoking|smoulder\w*|smolder\w*|steam\w*|sparks?|embers?|mist|fog|glow\w*|burning|ablaze|lightning)\b/i;

function checkSolid(issues: ScriptIssue[], path: string, value: string) {
  const found = EFFECT_WORDS.exec(value);
  if (found) {
    issues.push({
      path,
      message: `${JSON.stringify(found[0])} is not a solid surface and would be built into the mesh; describe it in effects`,
    });
  }
}

function crossCheck(show: ShowScript, source?: ScriptSource): ScriptIssue[] {
  const issues: ScriptIssue[] = [];
  const soundLint = loadSoundLint();
  if (source && show.id !== source.showId) {
    issues.push({
      path: "id",
      message: `id ${JSON.stringify(show.id)} must match ${source.showIdLabel} ${JSON.stringify(source.showId)}`,
    });
  }
  for (const [characterId, character] of Object.entries(show.characters)) {
    checkSnakeId(issues, `characters.${characterId}`, characterId, "character id");
    checkSolid(issues, `characters.${characterId}.body`, character.body);
    character.attributes.forEach((attribute, index) =>
      checkSolid(issues, `characters.${characterId}.attributes[${index}]`, attribute),
    );
  }
  for (const [locationId, location] of Object.entries(show.locations)) {
    checkSnakeId(issues, `locations.${locationId}`, locationId, "location id");
    checkSolid(issues, `locations.${locationId}.look.materials`, location.look.materials);
    for (const [landmarkId, landmark] of Object.entries(location.spatial.landmarks)) {
      const path = `locations.${locationId}.spatial.landmarks.${landmarkId}`;
      checkSnakeId(issues, path, landmarkId, "landmark id");
      checkSolid(issues, `${path}.appearance`, landmark.appearance);
    }
  }
  for (const [propId, prop] of Object.entries(show.props ?? {})) {
    checkSnakeId(issues, `props.${propId}`, propId, "prop id");
    checkSolid(issues, `props.${propId}.appearance`, prop.appearance);
    // One object is written once: worn for the whole show it is part of the
    // character's look; a prop is a separate mesh the timeline places.
    const words = propId.split("_");
    for (const [characterId, character] of Object.entries(show.characters)) {
      character.attributes.forEach((attribute, attributeIndex) => {
        if (words.every((word) => new RegExp(`\\b${word}\\b`, "i").test(attribute))) {
          issues.push({
            path: `characters.${characterId}.attributes[${attributeIndex}]`,
            message: `describes prop ${JSON.stringify(propId)}; an object is either part of the character's look (an attribute) or a prop the timeline places, never both`,
          });
        }
      });
    }
  }

  const seenEpisodes = new Set<number>();
  show.episodes.forEach((episode, episodeIndex) => {
    const episodePath = `episodes[${episodeIndex}]`;
    if (seenEpisodes.has(episode.episodeNumber)) {
      issues.push({ path: `${episodePath}.episodeNumber`, message: `duplicate episode number ${episode.episodeNumber}` });
    }
    seenEpisodes.add(episode.episodeNumber);

    const { durationSeconds, characterTracks, propTracks } = episode.spatialTimeline;
    const tracksPath = `${episodePath}.spatialTimeline`;
    for (const characterId of Object.keys(show.characters)) {
      if (!(characterId in characterTracks)) {
        issues.push({
          path: `${tracksPath}.characterTracks.${characterId}`,
          message: "every character has a track; use locationId null while they are off stage",
        });
      }
    }
    for (const propId of Object.keys(show.props ?? {})) {
      if (!(propId in propTracks)) {
        issues.push({
          path: `${tracksPath}.propTracks.${propId}`,
          message: "every prop has a track; use locationId null while it is off stage",
        });
      }
    }
    for (const [characterId, track] of Object.entries(characterTracks)) {
      const trackPath = `${tracksPath}.characterTracks.${characterId}`;
      if (!(characterId in show.characters)) {
        issues.push({ path: trackPath, message: `unknown character ${JSON.stringify(characterId)}` });
      }
      checkTrackTimes(issues, trackPath, track, durationSeconds);
      track.forEach((frame, frameIndex) => checkCharacterFrame(issues, `${trackPath}[${frameIndex}]`, frame, show));
    }
    for (const [propId, track] of Object.entries(propTracks)) {
      const trackPath = `${tracksPath}.propTracks.${propId}`;
      if (!(propId in (show.props ?? {}))) {
        issues.push({ path: trackPath, message: `prop ${JSON.stringify(propId)} is not declared. Add props.${propId}.` });
      }
      checkTrackTimes(issues, trackPath, track, durationSeconds);
      track.forEach((frame, frameIndex) => checkPropFrame(issues, `${trackPath}[${frameIndex}]`, frame, show));
    }

    episode.scenes.forEach((scene, sceneIndex) => {
      const scenePath = `${episodePath}.scenes[${sceneIndex}]`;
      if (scene.sceneNumber !== sceneIndex + 1) {
        issues.push({ path: `${scenePath}.sceneNumber`, message: `expected sceneNumber ${sceneIndex + 1}` });
      }
      if (!(scene.locationId in show.locations)) {
        issues.push({ path: `${scenePath}.locationId`, message: `unknown location ${JSON.stringify(scene.locationId)}` });
      }
      const [start, finish] = scene.timeRangeSeconds;
      if (!(finish > start)) {
        issues.push({ path: `${scenePath}.timeRangeSeconds`, message: "time range must increase" });
      } else if (start < -1e-6 || finish > durationSeconds + 1e-6) {
        issues.push({
          path: `${scenePath}.timeRangeSeconds`,
          message: `range ${start}..${finish} is outside the episode duration 0..${durationSeconds}`,
        });
      }
      checkPerformances(issues, show, scene, scenePath);
      checkSceneSoundWarnings(issues, show, scene, scenePath, soundLint);
      scene.camera.keyframes.forEach((frame, frameIndex) => {
        const path = `${scenePath}.camera.keyframes[${frameIndex}]`;
        if (frameIndex > 0 && frame.timeSeconds <= scene.camera.keyframes[frameIndex - 1].timeSeconds) {
          issues.push({ path: `${path}.timeSeconds`, message: "keyframe times must increase" });
        }
        checkCameraFrame(issues, path, frame, start, finish);
      });
      const travel = cameraTravelScore(scene);
      if (travel > CAMERA_TRAVEL_WARN_THRESHOLD) {
        issues.push({
          path: `${scenePath}.camera`,
          message: cameraTravelWarnMessage(travel, CAMERA_TRAVEL_WARN_THRESHOLD),
          severity: "warning",
        });
      }
    });
  });
  return issues;
}

function checkSceneSoundWarnings(
  issues: ScriptIssue[],
  show: ShowScript,
  scene: ScriptScene,
  scenePath: string,
  soundLint: SoundLintConfig,
) {
  const sound = scene.sound;
  if (scene.speakerId && sound.bed !== "faint") {
    issues.push({
      path: `${scenePath}.sound.bed`,
      message: `speaking scene should use bed "faint" so dialogue is not buried under the location bed`,
      severity: "warning",
    });
  }
  sound.events.forEach((event, eventIndex) => {
    if (wordCount(event.text) <= soundLint.emptyEventsMaxWords) {
      issues.push({
        path: `${scenePath}.sound.events[${eventIndex}].text`,
        message: `event looks empty-ish (${wordCount(event.text)} word(s); describe the audible action)`,
        severity: "warning",
      });
    }
  });
  if (sound.music.kind === "none") {
    const ambience = show.locations[scene.locationId]?.soundscape.ambience ?? "";
    const hitAmbience = containsMusicWord(ambience, soundLint.musicConflictWords);
    if (hitAmbience) {
      issues.push({
        path: `${scenePath}.sound.music`,
        message: `music is "none" but location ambience contains ${JSON.stringify(hitAmbience)}`,
        severity: "warning",
      });
    }
    const hitEvents = containsMusicWord(sound.events.map((event) => event.text).join(" "), soundLint.musicConflictWords);
    if (hitEvents) {
      issues.push({
        path: `${scenePath}.sound.music`,
        message: `music is "none" but sound.events contains ${JSON.stringify(hitEvents)}`,
        severity: "warning",
      });
    }
  }
}

/**
 * References and structure. Who is visible, and whether each part, face, and
 * source is on screen, is answered by content:previs from the render.
 */
function checkPerformances(issues: ScriptIssue[], show: ShowScript, scene: ScriptScene, scenePath: string) {
  const performances = scene.performances ?? {};
  // The video model sees pixels, not ids. People are placed for it from the
  // start frame, and gaze comes from lookAtId, so visual text must not name anyone.
  const namesSomeone = (value: string | undefined) =>
    Object.keys(show.characters).find((id) => new RegExp(`\\b${id.split("_").join("[ _]")}\\b`, "i").test(value ?? ""));
  const visualText: [string, string | undefined][] = [[`${scenePath}.motion`, scene.motion]];
  for (const [characterId, performance] of Object.entries(performances)) {
    const path = `${scenePath}.performances.${characterId}`;
    if (!(characterId in show.characters)) {
      issues.push({ path, message: `unknown character ${JSON.stringify(characterId)}` });
    }
    visualText.push([`${path}.action`, performance.action], [`${path}.expression`, performance.expression]);
    const seen = new Set<string>();
    performance.parts.forEach((part, partIndex) => {
      checkPartId(issues, `${path}.parts[${partIndex}]`, part);
      if (seen.has(part)) {
        issues.push({ path: `${path}.parts[${partIndex}]`, message: `duplicate part ${JSON.stringify(part)}` });
      }
      seen.add(part);
    });
  }
  for (const [path, value] of visualText) {
    const named = namesSomeone(value);
    if (named) {
      issues.push({
        path,
        message: `names character ${JSON.stringify(named)}; the video model only sees the frame, so say "him", "her", or "them" and let lookAtId set the gaze`,
      });
    }
  }
  if (scene.speakerId) {
    const speaker = performances[scene.speakerId];
    if (!speaker) {
      issues.push({ path: `${scenePath}.speakerId`, message: "the speaker needs a performance" });
    } else if (!speaker.expression) {
      issues.push({
        path: `${scenePath}.performances.${scene.speakerId}.expression`,
        message: "the speaker needs an expression; the still must show the mouth ready to speak (lips slightly parted)",
      });
    }
    if (!scene.dialogue) {
      issues.push({ path: `${scenePath}.dialogue`, message: "required when speakerId is set" });
    }
  } else if (scene.dialogue) {
    issues.push({ path: `${scenePath}.dialogue`, message: "dialogue needs a speakerId" });
  }
  const landmarks = show.locations[scene.locationId]?.spatial.landmarks ?? {};
  scene.sound.events.forEach((event, eventIndex) => {
    const path = `${scenePath}.sound.events[${eventIndex}].source`;
    const source = event.source;
    if ("characterId" in source) {
      checkPartId(issues, `${path}.part`, source.part);
      const moves = performances[source.characterId]?.parts ?? [];
      const speaks = source.characterId === scene.speakerId && source.part === "face";
      if (!moves.includes(source.part) && !speaks) {
        issues.push({
          path: `${path}.part`,
          message: `${source.characterId}'s ${source.part} makes this sound, but performances.${source.characterId}.parts does not move it; the video model voices the sound and leaves the part still`,
        });
      }
    } else if ("landmarkId" in source) {
      if (!(source.landmarkId in landmarks)) {
        issues.push({
          path: `${path}.landmarkId`,
          message: `unknown landmark ${JSON.stringify(source.landmarkId)} in location ${JSON.stringify(scene.locationId)}`,
        });
      }
    } else if (!(source.propId in (show.props ?? {}))) {
      issues.push({ path: `${path}.propId`, message: `unknown prop ${JSON.stringify(source.propId)}` });
    }
  });
}

function isErrorIssue(issue: ScriptIssue): boolean {
  return issue.severity !== "warning";
}

export function parseShowScript(data: unknown, source?: ScriptSource): ParseResult {
  const parsed = showScriptSchema.safeParse(data);
  if (!parsed.success) {
    return { ok: false, issues: zodIssues(parsed.error) };
  }
  const script = parsed.data as ShowScript;
  const issues = crossCheck(script, source);
  if (issues.some(isErrorIssue)) {
    return { ok: false, issues };
  }
  return { ok: true, script, issues };
}

export function formatScriptReport(label: string, issues: ScriptIssue[]): string {
  if (issues.length === 0) {
    return `${label}\n  ok`;
  }
  const lines = issues.map((issue) => {
    const tag = issue.severity === "warning" ? "warning" : "error";
    return `  ${tag} ${issue.path}: ${issue.message}`;
  });
  return [`${label}`, ...lines].join("\n");
}
