/**
 * Authoring check for ShowScript JSON. Renderer geometry that needs the
 * camera basis (intersection, projection) stays in spatial_previs.py.
 */

import { z } from "zod";

import type {
  CharacterSpatialKeyframe,
  PropSpatialKeyframe,
  ShowLocation,
  ShowScript,
  SpatialCameraKeyframe,
} from "./script";

export type ScriptIssue = {
  path: string;
  message: string;
};

export type ScriptSource = {
  /** Value `id` must equal: filename stem, or the parent folder of script.json. */
  showId: string;
  /** Phrase used in the mismatch error, e.g. "the filename stem". */
  showIdLabel: string;
};

export type ParseResult =
  | { ok: true; script: ShowScript; issues: [] }
  | { ok: false; issues: ScriptIssue[] };

const SNAKE_ID = /^[a-z][a-z0-9]*(_[a-z0-9]+)*$/;
const SHOW_ID = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

const STANCES = ["standing", "sitting", "kneeling", "walking"] as const;
const BUILDS = ["slim", "average", "broad"] as const;

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

const locationLandmarkSchema = z
  .object({
    position: vec3("position").optional(),
    size: vec3("size", true).optional(),
    appearance: text("appearance").optional(),
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

const characterProxySchema = z
  .object({
    heightMeters: finiteNumber("proxy.heightMeters").refine(
      (value) => value >= 1.2 && value <= 2.4,
      "proxy.heightMeters must be a standing height from 1.2 to 2.4 meters",
    ),
    build: z.enum(BUILDS, {
      required_error: "proxy.build is required",
      invalid_type_error: `proxy.build must be ${BUILDS.join(", ")}`,
    }),
  })
  .strict();

const showCharacterSchema = z
  .object({
    promptBlock: text("promptBlock"),
    proxy: characterProxySchema.optional(),
  })
  .strict();

const showPropSchema = z
  .object({
    appearance: text("appearance"),
    sizeMeters: vec3("sizeMeters", true).optional(),
  })
  .strict();

const characterKeyframeSchema = z
  .object({
    timeSeconds: finiteNumber("timeSeconds"),
    locationId: text("locationId"),
    position: vec3("position"),
    bodyYawDegrees: finiteNumber("bodyYawDegrees"),
    lookAtId: text("lookAtId").optional(),
    stance: z.enum(STANCES, {
      required_error: "stance is required",
      invalid_type_error: `stance must be ${STANCES.join(", ")}`,
    }),
    leftHandTargetId: text("leftHandTargetId").optional(),
    rightHandTargetId: text("rightHandTargetId").optional(),
  })
  .strict();

const propKeyframeSchema = z
  .object({
    timeSeconds: finiteNumber("timeSeconds"),
    locationId: text("locationId"),
    position: vec3("position").optional(),
    heldByCharacterId: text("heldByCharacterId").optional(),
    heldInHand: z.enum(["left", "right"], {
      invalid_type_error: "heldInHand must be left or right",
    }).optional(),
  })
  .strict();

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
    characterIds: z.array(text("characterIds"), {
      required_error: "characterIds is required",
      invalid_type_error: "characterIds must be an array",
    }),
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
    imagePrompt: z.string().optional(),
    videoPrompt: z.string().optional(),
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
            promptBlock: text("promptBlock"),
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

function outsideStage(
  position: [number, number, number],
  location: ShowLocation,
): boolean {
  const [width, depth] = location.spatial.sizeMeters;
  return (
    Math.abs(position[0]) > width / 2 ||
    Math.abs(position[1]) > depth / 2 ||
    position[2] < 0
  );
}

function locationHasPeople(show: ShowScript, locationId: string): boolean {
  return show.episodes.some((episode) => {
    if (episode.scenes.some((scene) => scene.locationId === locationId && scene.characterIds.length > 0)) {
      return true;
    }
    return Object.values(episode.spatialTimeline.characterTracks).some((track) =>
      track.some((frame) => frame.locationId === locationId),
    );
  });
}

function checkSnakeId(issues: ScriptIssue[], path: string, id: string, label: string) {
  if (!SNAKE_ID.test(id)) {
    issues.push({
      path,
      message: `${label} ${JSON.stringify(id)} must be lowercase snake_case`,
    });
  }
}

function checkIncreasingTimes(
  issues: ScriptIssue[],
  path: string,
  frames: Array<{ timeSeconds: number }>,
) {
  for (let index = 1; index < frames.length; index += 1) {
    if (frames[index].timeSeconds <= frames[index - 1].timeSeconds) {
      issues.push({
        path: `${path}[${index}].timeSeconds`,
        message: "keyframe times must increase",
      });
    }
  }
}

function targetExists(
  show: ShowScript,
  locationId: string,
  targetId: string,
): boolean {
  if (targetId in show.characters || targetId in (show.props ?? {})) {
    return true;
  }
  const location = show.locations[locationId];
  return Boolean(location && targetId in location.spatial.landmarks);
}

function checkTarget(
  issues: ScriptIssue[],
  path: string,
  targetId: string | undefined,
  locationId: string,
  show: ShowScript,
) {
  if (!targetId || !(locationId in show.locations)) {
    return;
  }
  if (!targetExists(show, locationId, targetId)) {
    issues.push({
      path,
      message: `${JSON.stringify(targetId)} must be a character, a prop, or a landmark in ${JSON.stringify(locationId)}`,
    });
  }
}

function checkCharacterFrame(
  issues: ScriptIssue[],
  path: string,
  frame: CharacterSpatialKeyframe,
  show: ShowScript,
  durationSeconds: number,
) {
  if (frame.timeSeconds < 0 || frame.timeSeconds > durationSeconds + 1e-6) {
    issues.push({
      path: `${path}.timeSeconds`,
      message: `time ${frame.timeSeconds} is outside 0..${durationSeconds}`,
    });
  }
  const location = show.locations[frame.locationId];
  if (!location) {
    issues.push({
      path: `${path}.locationId`,
      message: `unknown location ${JSON.stringify(frame.locationId)}`,
    });
    return;
  }
  if (outsideStage(frame.position, location)) {
    issues.push({
      path: `${path}.position`,
      message: `position ${JSON.stringify(frame.position)} is outside ${frame.locationId}`,
    });
  }
  checkTarget(issues, `${path}.lookAtId`, frame.lookAtId, frame.locationId, show);
  checkTarget(issues, `${path}.leftHandTargetId`, frame.leftHandTargetId, frame.locationId, show);
  checkTarget(issues, `${path}.rightHandTargetId`, frame.rightHandTargetId, frame.locationId, show);
}

function checkPropFrame(
  issues: ScriptIssue[],
  path: string,
  frame: PropSpatialKeyframe,
  show: ShowScript,
  durationSeconds: number,
) {
  if (frame.timeSeconds < 0 || frame.timeSeconds > durationSeconds + 1e-6) {
    issues.push({
      path: `${path}.timeSeconds`,
      message: `time ${frame.timeSeconds} is outside 0..${durationSeconds}`,
    });
  }
  const location = show.locations[frame.locationId];
  if (!location) {
    issues.push({
      path: `${path}.locationId`,
      message: `unknown location ${JSON.stringify(frame.locationId)}`,
    });
  } else if (frame.position && outsideStage(frame.position, location)) {
    issues.push({
      path: `${path}.position`,
      message: `position ${JSON.stringify(frame.position)} is outside ${frame.locationId}`,
    });
  }
  if (!frame.position && !frame.heldByCharacterId) {
    issues.push({
      path,
      message: "prop keyframe needs position or heldByCharacterId",
    });
  }
  if (frame.heldInHand && !frame.heldByCharacterId) {
    issues.push({
      path: `${path}.heldInHand`,
      message: "heldInHand requires heldByCharacterId",
    });
  }
  if (frame.heldByCharacterId && !(frame.heldByCharacterId in show.characters)) {
    issues.push({
      path: `${path}.heldByCharacterId`,
      message: `unknown character ${JSON.stringify(frame.heldByCharacterId)}`,
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
    issues.push({
      path: `${path}.lookAt`,
      message: "lookAt must differ from position",
    });
  }
}

function crossCheck(show: ShowScript, source?: ScriptSource): ScriptIssue[] {
  const issues: ScriptIssue[] = [];
  if (source && show.id !== source.showId) {
    issues.push({
      path: "id",
      message: `id ${JSON.stringify(show.id)} must match ${source.showIdLabel} ${JSON.stringify(source.showId)}`,
    });
  }

  for (const characterId of Object.keys(show.characters)) {
    checkSnakeId(issues, `characters.${characterId}`, characterId, "character id");
  }
  for (const [locationId, location] of Object.entries(show.locations)) {
    checkSnakeId(issues, `locations.${locationId}`, locationId, "location id");
    const occupied = locationHasPeople(show, locationId);
    for (const [landmarkId, landmark] of Object.entries(location.spatial.landmarks)) {
      const landmarkPath = `locations.${locationId}.spatial.landmarks.${landmarkId}`;
      checkSnakeId(issues, landmarkPath, landmarkId, "landmark id");
      if (occupied) {
        if (!landmark.position) {
          issues.push({ path: `${landmarkPath}.position`, message: "position is required" });
        }
        if (!landmark.size) {
          issues.push({ path: `${landmarkPath}.size`, message: "size is required" });
        }
        if (!landmark.appearance) {
          issues.push({ path: `${landmarkPath}.appearance`, message: "appearance is required" });
        }
      } else if (landmark.size || landmark.appearance) {
        issues.push({
          path: landmarkPath,
          message: "size and appearance belong on a location that has people",
        });
      }
    }
  }
  for (const propId of Object.keys(show.props ?? {})) {
    checkSnakeId(issues, `props.${propId}`, propId, "prop id");
  }

  const seenEpisodes = new Set<number>();
  show.episodes.forEach((episode, episodeIndex) => {
    const episodePath = `episodes[${episodeIndex}]`;
    if (seenEpisodes.has(episode.episodeNumber)) {
      issues.push({
        path: `${episodePath}.episodeNumber`,
        message: `duplicate episode number ${episode.episodeNumber}`,
      });
    }
    seenEpisodes.add(episode.episodeNumber);

    const { durationSeconds, characterTracks, propTracks } = episode.spatialTimeline;
    for (const [characterId, track] of Object.entries(characterTracks)) {
      const trackPath = `${episodePath}.spatialTimeline.characterTracks.${characterId}`;
      if (!(characterId in show.characters)) {
        issues.push({
          path: trackPath,
          message: `unknown character ${JSON.stringify(characterId)}`,
        });
      }
      checkIncreasingTimes(issues, trackPath, track);
      track.forEach((frame, frameIndex) => {
        checkCharacterFrame(issues, `${trackPath}[${frameIndex}]`, frame, show, durationSeconds);
      });
    }
    for (const [propId, track] of Object.entries(propTracks)) {
      const trackPath = `${episodePath}.spatialTimeline.propTracks.${propId}`;
      if (!(propId in (show.props ?? {}))) {
        issues.push({
          path: trackPath,
          message: `prop ${JSON.stringify(propId)} is not declared on the show. Add props.${propId}.`,
        });
      }
      checkIncreasingTimes(issues, trackPath, track);
      track.forEach((frame, frameIndex) => {
        checkPropFrame(issues, `${trackPath}[${frameIndex}]`, frame, show, durationSeconds);
      });
    }

    episode.scenes.forEach((scene, sceneIndex) => {
      const scenePath = `${episodePath}.scenes[${sceneIndex}]`;
      if (scene.sceneNumber !== sceneIndex + 1) {
        issues.push({
          path: `${scenePath}.sceneNumber`,
          message: `expected sceneNumber ${sceneIndex + 1}`,
        });
      }
      if (!(scene.locationId in show.locations)) {
        issues.push({
          path: `${scenePath}.locationId`,
          message: `unknown location ${JSON.stringify(scene.locationId)}`,
        });
      }
      const [start, finish] = scene.timeRangeSeconds;
      if (!(finish > start)) {
        issues.push({
          path: `${scenePath}.timeRangeSeconds`,
          message: "time range must increase",
        });
      } else if (start < -1e-6 || finish > durationSeconds + 1e-6) {
        issues.push({
          path: `${scenePath}.timeRangeSeconds`,
          message: `range ${start}..${finish} is outside the episode duration 0..${durationSeconds}`,
        });
      }
      for (const characterId of scene.characterIds) {
        if (!(characterId in show.characters)) {
          issues.push({
            path: `${scenePath}.characterIds`,
            message: `unknown character ${JSON.stringify(characterId)}`,
          });
        }
      }
      if (scene.speakerId && !scene.characterIds.includes(scene.speakerId)) {
        issues.push({
          path: `${scenePath}.speakerId`,
          message: `${JSON.stringify(scene.speakerId)} is not in characterIds`,
        });
      }
      checkIncreasingTimes(issues, `${scenePath}.camera.keyframes`, scene.camera.keyframes);
      if (finish > start) {
        scene.camera.keyframes.forEach((frame, frameIndex) => {
          checkCameraFrame(
            issues,
            `${scenePath}.camera.keyframes[${frameIndex}]`,
            frame,
            start,
            finish,
          );
        });
      }
    });
  });
  return issues;
}

export function parseShowScript(data: unknown, source?: ScriptSource): ParseResult {
  const parsed = showScriptSchema.safeParse(data);
  if (!parsed.success) {
    return { ok: false, issues: zodIssues(parsed.error) };
  }
  const script = parsed.data as ShowScript;
  const issues = crossCheck(script, source);
  if (issues.length > 0) {
    return { ok: false, issues };
  }
  return { ok: true, script, issues: [] };
}

export function formatScriptReport(label: string, issues: ScriptIssue[]): string {
  if (issues.length === 0) {
    return `${label}\n  ok`;
  }
  const lines = issues.map((issue) => `  ${issue.path}: ${issue.message}`);
  return [`${label}`, ...lines].join("\n");
}
