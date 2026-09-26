import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import type { ShowScript } from "./script";
import { parseShowScript } from "./script-schema";

const ironBridePath = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../../content-pipeline/shows/the-iron-bride/script.json",
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
              position: [0, 1, 0],
              size: [1.6, 0.6, 0.5],
              appearance: "One stone bench, a single object, no room.",
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
  if (!result.ok) {
    const report = result.issues.map((issue) => `${issue.path}: ${issue.message}`).join("\n");
    assert.fail(report);
    return;
  }
  assert.equal(result.script.id, "the-iron-bride");
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

test("accepts an empty landmark on a location with no people", () => {
  const show = minimalShow();
  show.episodes[0].scenes[0].characterIds = [];
  show.episodes[0].spatialTimeline.characterTracks = {};
  show.locations.room.spatial.landmarks.bench = {};
  const result = parseShowScript(show, {
    showId: "demo-show",
    showIdLabel: "the filename stem",
  });
  assert.equal(result.ok, true);
});

test("requires size and appearance where people stand", () => {
  const show = minimalShow();
  show.locations.room.spatial.landmarks.bench = { position: [0, 1, 0] };
  const report = messages(show);
  assert.match(report, /size is required/);
  assert.match(report, /appearance is required/);
});

test("rejects a catalog search on a landmark", () => {
  const show = minimalShow();
  (show.locations.room.spatial.landmarks.bench as { need?: { query: string } }).need = {
    query: "stone well",
  };
  const report = messages(show);
  assert.match(report, /need/);
});

test("rejects a landmark appearance on a location with no people", () => {
  const show = minimalShow();
  show.episodes[0].scenes[0].characterIds = [];
  show.episodes[0].spatialTimeline.characterTracks = {};
  show.locations.room.spatial.landmarks.bench = {
    appearance: "One low stone fire ring, a single object.",
  };
  const report = messages(show);
  assert.match(report, /size and appearance belong on a location that has people/);
});

test("rejects a catalog id on a landmark", () => {
  const show = minimalShow();
  (show.locations.room.spatial.landmarks.bench as { assetId?: string }).assetId =
    "sketchfab:8ca31b1d1635406ba2db30e48ecddbdd";
  const report = messages(show);
  assert.match(report, /assetId/);
});

test("rejects a prop without an appearance", () => {
  const show = minimalShow({ props: { cup: { sizeMeters: [0.1, 0.1, 0.1] } } });
  const report = messages(show);
  assert.match(report, /props\.cup\.appearance/);
  assert.match(report, /appearance is required/);
});

test("rejects a speaker who is not on camera", () => {
  const show = minimalShow();
  show.episodes[0].scenes[0].speakerId = "ada";
  show.episodes[0].scenes[0].characterIds = [];
  const report = messages(show);
  assert.match(report, /speakerId/);
  assert.match(report, /not in characterIds/);
});

test("rejects a prefab id on a landmark", () => {
  const show = minimalShow();
  (show.locations.room.spatial.landmarks.bench as { prefabId?: string }).prefabId = "blocks/chair";
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
