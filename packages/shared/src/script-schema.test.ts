import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import type { ShowScript } from "./script";
import { parseShowScript } from "./script-schema";

const ironBridePath = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../../content-pipeline/scripts_input/the-iron-bride.json",
);

function minimalShow(overrides: Partial<ShowScript> = {}): ShowScript {
  return {
    id: "demo-show",
    title: "Demo",
    characters: {
      ada: { promptBlock: "Adult woman, 30, average build, brown eyes, black hair." },
    },
    locations: {
      room: {
        promptBlock: "A small stone room.",
        spatial: {
          sizeMeters: [8, 10, 4],
          landmarks: {
            bench: {
              kind: "seat",
              position: [0, 1, 0],
              size: [1.2, 0.5, 0.45],
            },
          },
        },
      },
    },
    episodes: [
      {
        episodeNumber: 1,
        title: "One",
        isFree: true,
        coinCost: 0,
        spatialTimeline: {
          durationSeconds: 4,
          characterTracks: {
            ada: [
              {
                timeSeconds: 0,
                locationId: "room",
                position: [0, 0, 0],
                bodyYawDegrees: 0,
                stance: "standing",
              },
            ],
          },
          propTracks: {},
        },
        scenes: [
          {
            sceneNumber: 1,
            locationId: "room",
            storyBeat: "Ada waits.",
            characterIds: ["ada"],
            timeRangeSeconds: [0, 4],
            camera: {
              keyframes: [
                {
                  timeSeconds: 0,
                  position: [0, -3, 1.5],
                  lookAt: [0, 0, 1.5],
                  verticalFovDegrees: 40,
                },
              ],
            },
          },
        ],
      },
    ],
    ...overrides,
  };
}

function messages(data: unknown) {
  const result = parseShowScript(data);
  assert.equal(result.ok, false);
  return result.issues.map((issue) => `${issue.path}: ${issue.message}`).join("\n");
}

test("accepts a show that omits the new optional fields", () => {
  const result = parseShowScript(minimalShow(), {
    showId: "demo-show",
    showIdLabel: "the filename stem",
  });
  assert.equal(result.ok, true);
});

test("the iron bride script matches its filename", () => {
  const data = JSON.parse(readFileSync(ironBridePath, "utf8"));
  const result = parseShowScript(data, {
    showId: "the-iron-bride",
    showIdLabel: "the filename stem",
  });
  assert.equal(result.ok, true, result.ok ? "" : result.issues.map((issue) => `${issue.path}: ${issue.message}`).join("\n"));
});

test("rejects a prop track whose id was never declared", () => {
  const show = minimalShow();
  show.episodes[0].spatialTimeline.propTracks = {
    iron_collar: [
      {
        timeSeconds: 0,
        locationId: "room",
        heldByCharacterId: "ada",
        heldInHand: "left",
      },
    ],
  };
  const report = messages(show);
  assert.match(report, /propTracks\.iron_collar/);
  assert.match(report, /not declared/);
});

test("rejects a prop with nothing to resolve", () => {
  const show = minimalShow({ props: { cup: {} } });
  const report = messages(show);
  assert.match(report, /props\.cup/);
  assert.match(report, /need\.query or prefabId/);
});

test("rejects an unknown landmark kind", () => {
  const show = minimalShow();
  (show.locations.room.spatial.landmarks.bench as { kind: string }).kind = "throne";
  const report = messages(show);
  assert.match(report, /landmarks\.bench\.kind/);
});

test("rejects a speaker who is not on camera", () => {
  const show = minimalShow();
  show.episodes[0].scenes[0].speakerId = "ada";
  show.episodes[0].scenes[0].characterIds = [];
  const report = messages(show);
  assert.match(report, /speakerId/);
  assert.match(report, /not in characterIds/);
});

test("rejects a prefab id that is not kebab-case", () => {
  const show = minimalShow();
  show.locations.room.spatial.landmarks.bench.prefabId = "Wooden Chair";
  const report = messages(show);
  assert.match(report, /prefabId/);
});

test("rejects a show id that does not match the file", () => {
  const result = parseShowScript(minimalShow(), {
    showId: "other-show",
    showIdLabel: "the filename stem",
  });
  assert.equal(result.ok, false);
  assert.match(result.issues[0].message, /other-show/);
});
