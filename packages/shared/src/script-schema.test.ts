import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import type { ShowScript } from "./script";
import {
  CAMERA_TRAVEL_WARN_THRESHOLD,
  cameraTravelScore,
  parseShowScript,
} from "./script-schema";

const ironBridePath = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../../content-pipeline/shows/the-iron-bride/script.json",
);

function minimalShow(overrides: Partial<ShowScript> = {}): ShowScript {
  return {
    id: "demo-show",
    title: "Demo",
    characters: {
      ada: {
        body: "brown skin",
        attributes: [
          { text: "in a plain dress", parts: ["torso", "legs"] },
          { text: "black hair", parts: ["hair"] },
          { text: "barefoot", parts: ["feet"] },
        ],
      },
    },
    locations: {
      room: {
        promptBlock: "A small stone room.",
        backdrop: {
          sky: "a gray sky",
          skyColor: [40, 48, 64],
          ground: "stone paving",
          groundColor: [48, 44, 40],
          surround: "open fields",
          surroundColor: [24, 56, 40],
        },
        soundscape: {
          ambience: "soft wind through stone arches, distant drip of water",
          space: "enclosed stone chamber with short dry echo",
        },
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
            performances: {
              ada: { action: "stands still, then shifts her weight once", parts: ["legs"] },
            },
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
            sound: {
              events: [
                {
                  text: "dress fabric rustles as she shifts her weight",
                  source: { characterId: "ada", part: "legs" },
                },
              ],
              bed: "present",
              music: { kind: "none" },
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

test("accepts a complete minimal show", () => {
  const result = parseShowScript(minimalShow(), {
    showId: "demo-show",
    showIdLabel: "the filename stem",
  });
  assert.equal(result.ok, true);
});

test("rejects a location missing soundscape", () => {
  const show = minimalShow();
  delete (show.locations.room as { soundscape?: unknown }).soundscape;
  const report = messages(show);
  assert.match(report, /locations\.room\.soundscape/);
});

test("rejects a scene missing sound", () => {
  const show = minimalShow();
  delete (show.episodes[0].scenes[0] as { sound?: unknown }).sound;
  const report = messages(show);
  assert.match(report, /episodes\[0\]\.scenes\[0\]\.sound/);
});

test("rejects sound.music without an explicit kind", () => {
  const show = minimalShow();
  (show.episodes[0].scenes[0].sound as { music: unknown }).music = {};
  const report = messages(show);
  assert.match(report, /sound\.music/);
});

test("warns when a speaking scene does not use bed faint", () => {
  const show = minimalShow();
  show.episodes[0].scenes[0].speakerId = "ada";
  show.episodes[0].scenes[0].sound.bed = "present";
  show.episodes[0].scenes[0].dialogue = { line: "Hello.", delivery: "soft" };
  const result = parseShowScript(show, {
    showId: "demo-show",
    showIdLabel: "the filename stem",
  });
  assert.equal(result.ok, true);
  const warning = result.issues.find(
    (issue) => issue.severity === "warning" && issue.path.endsWith("sound.bed"),
  );
  assert.ok(warning, "expected bed warning");
  assert.match(warning.message, /faint/);
});

test("warns when music none conflicts with ambience music words", () => {
  const show = minimalShow();
  show.locations.room.soundscape.ambience = "soft choir music under the arches";
  const result = parseShowScript(show, {
    showId: "demo-show",
    showIdLabel: "the filename stem",
  });
  assert.equal(result.ok, true);
  const warning = result.issues.find(
    (issue) => issue.severity === "warning" && issue.path.endsWith("sound.music"),
  );
  assert.ok(warning, "expected music conflict warning");
  assert.match(warning.message, /choir|music/);
});

test("the iron bride script matches its filename", () => {
  const data = JSON.parse(readFileSync(ironBridePath, "utf8"));
  const result = parseShowScript(data, {
    showId: "the-iron-bride",
    showIdLabel: "the filename stem",
  });
  if (!result.ok) {
    const leftover = result.issues.filter(
      (issue) =>
        issue.severity !== "warning" &&
        !issue.message.includes("keep at least 1.5m from every mesh"),
    );
    if (leftover.length > 0) {
      const report = leftover.map((issue) => `${issue.path}: ${issue.message}`).join("\n");
      assert.fail(report);
    }
    return;
  }
  assert.equal(result.script.id, "the-iron-bride");
  const travelWarnings = result.issues.filter((issue) =>
    issue.message.includes("high camera travel+rotation score"),
  );
  assert.ok(travelWarnings.length >= 1, "expected at least one high-travel warning");
  for (const location of Object.values(result.script.locations)) {
    assert.ok(location.soundscape?.ambience);
    assert.ok(location.soundscape?.space);
  }
  for (const scene of result.script.episodes[0]!.scenes) {
    assert.ok(scene.sound?.events);
    assert.ok(scene.sound?.bed === "present" || scene.sound?.bed === "faint");
    assert.ok(scene.sound?.music?.kind === "none" || scene.sound?.music?.kind === "described");
  }
});

test("cameraTravelScore matches Gate 3b scene 01 figure (~39.5)", () => {
  const data = JSON.parse(readFileSync(ironBridePath, "utf8")) as ShowScript;
  const scene01 = data.episodes[0]!.scenes.find((scene) => scene.sceneNumber === 1);
  assert.ok(scene01);
  const score = cameraTravelScore(scene01);
  assert.ok(score > CAMERA_TRAVEL_WARN_THRESHOLD);
  assert.ok(Math.abs(score - 39.47) < 0.05, `expected ~39.47, got ${score}`);
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
  show.episodes[0].scenes[0].performances = {};
  show.episodes[0].scenes[0].sound.events = [
    { text: "the bench creaks as it settles", source: { landmarkId: "bench" } },
  ];
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

test("rejects a character on camera without a performance", () => {
  const show = minimalShow();
  show.episodes[0].scenes[0].performances = {};
  const report = messages(show);
  assert.match(report, /scenes\[0\]\.performances\.ada: every character on camera needs a performance/);
});

test("rejects a performance for someone not in the shot", () => {
  const show = minimalShow();
  show.episodes[0].scenes[0].performances.bram = { action: "waits", parts: ["torso"] };
  const report = messages(show);
  assert.match(report, /scenes\[0\]\.performances\.bram: "bram" is not in characterIds/);
});

test("rejects a performance without parts or with an unknown part", () => {
  const show = minimalShow();
  show.episodes[0].scenes[0].performances.ada.parts = [];
  assert.match(messages(show), /parts needs at least one body part/);
  show.episodes[0].scenes[0].performances.ada.parts = ["toes" as "feet"];
  assert.match(messages(show), /performances\.ada\.parts\[0\]: unknown part id "toes"/);
});

test("rejects the removed free-text prompt fields", () => {
  const show = minimalShow();
  (show.episodes[0].scenes[0] as { videoPrompt?: string }).videoPrompt = "PERFORMANCE: waits.";
  assert.match(messages(show), /videoPrompt/);
});

test("requires dialogue exactly when someone speaks", () => {
  const speaking = minimalShow();
  speaking.episodes[0].scenes[0].speakerId = "ada";
  assert.match(messages(speaking), /scenes\[0\]\.dialogue: required when speakerId is set/);
  const silent = minimalShow();
  silent.episodes[0].scenes[0].dialogue = { line: "Hello.", delivery: "soft" };
  assert.match(messages(silent), /scenes\[0\]\.dialogue: dialogue needs a speakerId/);
});

test("rejects a body sound from a part the performance does not move", () => {
  const show = minimalShow();
  show.episodes[0].scenes[0].sound.events[0].source = { characterId: "ada", part: "face" };
  const report = messages(show);
  assert.match(report, /source\.part: ada's face makes this sound, but performances\.ada\.parts does not move it/);
});

test("lets a speaker's face make a sound", () => {
  const show = minimalShow();
  const scene = show.episodes[0].scenes[0];
  scene.speakerId = "ada";
  scene.sound.bed = "faint";
  scene.dialogue = { line: "Hello.", delivery: "soft" };
  scene.sound.events[0] = { text: "a sharp breath before the word", source: { characterId: "ada", part: "face" } };
  assert.equal(parseShowScript(show).ok, true);
});

test("rejects sound sources that are not in the scene", () => {
  const show = minimalShow();
  const events = show.episodes[0].scenes[0].sound.events;
  events.push({ text: "wood creaks under a weight", source: { landmarkId: "altar" } });
  events.push({ text: "a lantern rattles on its hook", source: { propId: "lantern" } });
  events.push({ text: "a heavy step on stone", source: { characterId: "bram", part: "feet" } });
  const report = messages(show);
  assert.match(report, /events\[1\]\.source\.landmarkId: unknown landmark "altar"/);
  assert.match(report, /events\[2\]\.source\.propId: prop "lantern" has no propTracks entry/);
  assert.match(report, /events\[3\]\.source\.characterId: "bram" is not in characterIds/);
});

test("accepts a world shot with no people and a landmark sound", () => {
  const show = minimalShow();
  const scene = show.episodes[0].scenes[0];
  scene.characterIds = [];
  scene.performances = {};
  scene.motion = "dust drifts through a shaft of light";
  scene.sound.events = [{ text: "the bench creaks as it settles", source: { landmarkId: "bench" } }];
  assert.equal(parseShowScript(show).ok, true);
});

test("rejects a prefab id on a landmark", () => {
  const show = minimalShow();
  (show.locations.room.spatial.landmarks.bench as { prefabId?: string }).prefabId = "blocks/chair";
  const report = messages(show);
  assert.match(report, /prefabId/);
});

test("rejects a character missing body", () => {
  const show = minimalShow();
  delete (show.characters.ada as { body?: string }).body;
  const missing = messages(show);
  assert.match(missing, /characters\.ada\.body/);
  assert.match(missing, /body is required/);
});

test("rejects leftover description or stillDescription on a character", () => {
  const show = minimalShow();
  (show.characters.ada as { description?: string }).description = "leftover";
  const leftoverDescription = messages(show);
  assert.match(leftoverDescription, /description/);
  const again = minimalShow();
  (again.characters.ada as { stillDescription?: string }).stillDescription = "leftover";
  const leftoverStill = messages(again);
  assert.match(leftoverStill, /stillDescription/);
});

test("rejects leftover generalDescription on a character", () => {
  const show = minimalShow();
  (show.characters.ada as { generalDescription?: string }).generalDescription = "leftover";
  const leftover = messages(show);
  assert.match(leftover, /generalDescription/);
});

test("rejects an attribute with empty parts", () => {
  const show = minimalShow();
  show.characters.ada.attributes[0].parts = [];
  const report = messages(show);
  assert.match(report, /characters\.ada\.attributes\[0\]\.parts/);
  assert.match(report, /empty parts/);
});

test("rejects an unknown part id on a character attribute", () => {
  const show = minimalShow();
  (show.characters.ada.attributes[0].parts as string[]) = ["toes"];
  const report = messages(show);
  assert.match(report, /characters\.ada\.attributes\[0\]\.parts/);
  assert.match(report, /unknown part id "toes"/);
});

test("rejects an unknown part id on requiresParts", () => {
  const show = minimalShow();
  show.episodes[0].scenes[0].requiresParts = [{ characterId: "ada", part: "toes" as "feet" }];
  const report = messages(show);
  assert.match(report, /requiresParts\[0\]\.part/);
  assert.match(report, /unknown part id "toes"/);
});

test("rejects a camera closer than 1.5m to a character mesh", () => {
  const show = minimalShow();
  show.episodes[0].scenes[0].camera.keyframes[0].position = [0, 0.05, 1.2];
  show.episodes[0].scenes[0].camera.keyframes[0].lookAt = [0, 0, 1.2];
  const report = messages(show);
  assert.match(report, /scene 1: camera is 0\.00m from ada/);
  assert.match(report, /keep at least 1.5m from every mesh/);
});

test("rejects a camera closer than 1.5m to a landmark mesh", () => {
  const show = minimalShow();
  show.episodes[0].scenes[0].camera.keyframes[0].position = [0, 1, 0.2];
  show.episodes[0].scenes[0].camera.keyframes[0].lookAt = [0, 1, 0.4];
  const report = messages(show);
  assert.match(report, /from bench/);
  assert.match(report, /keep at least 1.5m from every mesh/);
});

test("rejects a show id that does not match the file", () => {
  const result = parseShowScript(minimalShow(), {
    showId: "other-show",
    showIdLabel: "the filename stem",
  });
  assert.equal(result.ok, false);
  assert.match(result.issues[0].message, /other-show/);
});

test("rejects visual text that names a character", () => {
  const show = minimalShow();
  show.episodes[0].scenes[0].performances.ada.action = "stands still, watching Ada's reflection";
  assert.match(messages(show), /performances\.ada\.action: names character "ada"/);
});
