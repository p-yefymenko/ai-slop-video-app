import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import type { ShowScript, Vec3 } from "./script";
import { CAMERA_TRAVEL_WARN_THRESHOLD, cameraTravelScore, parseShowScript } from "./script-schema";

const showPath = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../../content-pipeline/shows/return-of-the-wolf/script.json",
);

function minimalShow(overrides: Partial<ShowScript> = {}): ShowScript {
  return {
    id: "demo-show",
    title: "Demo",
    characters: {
      ada: {
        body: "brown skin",
        attributes: ["in a plain dress", "black hair", "barefoot"],
        heightMeters: 1.7,
      },
    },
    locations: {
      room: {
        look: { materials: "worn gray stone", light: "soft overcast daylight" },
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
            ada: [{ timeSeconds: 0, locationId: "room", position: [0, 0, 0], bodyYawDegrees: 0 }],
          },
          propTracks: {},
        },
        scenes: [
          {
            sceneNumber: 1,
            locationId: "room",
            storyBeat: "Ada waits.",
            timeRangeSeconds: [0, 4],
            performances: {
              ada: { action: "stands still, then shifts the weight once", parts: ["legs"] },
            },
            shot: { type: "single", subjects: ["ada"], size: "full" },
            sound: {
              events: [
                { text: "dress fabric rustles as she shifts her weight", source: { characterId: "ada", part: "legs" } },
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

function speaking(show: ShowScript) {
  const scene = show.episodes[0].scenes[0];
  scene.speakerId = "ada";
  scene.sound.bed = "faint";
  scene.dialogue = { line: "Hello.", delivery: "soft" };
  scene.performances.ada.expression = "calm, lips slightly parted";
  return show;
}

test("accepts a complete minimal show", () => {
  const result = parseShowScript(minimalShow(), { showId: "demo-show", showIdLabel: "the filename stem" });
  assert.equal(result.ok, true);
});

test("the development show script is valid", () => {
  const data = JSON.parse(readFileSync(showPath, "utf8"));
  const result = parseShowScript(data, { showId: "return-of-the-wolf", showIdLabel: "the show folder" });
  if (!result.ok) {
    assert.fail(result.issues.map((issue) => `${issue.path}: ${issue.message}`).join("\n"));
  }
});

test("cameraTravelScore adds travel, half the aim travel, and lens and roll change", () => {
  const scene = minimalShow().episodes[0].scenes[0];
  assert.equal(cameraTravelScore(scene), 0, "a framed shot has no authored travel");
  const first = { timeSeconds: 0, position: [0, -3, 1.5] as Vec3, lookAt: [0, 0, 1.5] as Vec3, verticalFovDegrees: 40 };
  scene.camera = { keyframes: [first, { ...first, timeSeconds: 4, position: [30, -3, 1.5], lookAt: [0, 10, 1.5] }] };
  assert.ok(Math.abs(cameraTravelScore(scene) - 35) < 1e-9);
  assert.ok(cameraTravelScore(scene) > CAMERA_TRAVEL_WARN_THRESHOLD);
});

test("rejects a location missing soundscape and a scene missing sound", () => {
  const show = minimalShow();
  delete (show.locations.room as { soundscape?: unknown }).soundscape;
  delete (show.episodes[0].scenes[0] as { sound?: unknown }).sound;
  const report = messages(show);
  assert.match(report, /locations\.room\.soundscape/);
  assert.match(report, /episodes\[0\]\.scenes\[0\]\.sound/);
});

test("rejects sound.music without an explicit kind", () => {
  const show = minimalShow();
  (show.episodes[0].scenes[0].sound as { music: unknown }).music = {};
  assert.match(messages(show), /sound\.music/);
});

test("warns when a speaking scene does not use bed faint", () => {
  const show = speaking(minimalShow());
  show.episodes[0].scenes[0].sound.bed = "present";
  const result = parseShowScript(show);
  assert.equal(result.ok, true);
  assert.ok(result.issues.some((issue) => issue.severity === "warning" && issue.path.endsWith("sound.bed")));
});

test("warns when music none conflicts with ambience music words", () => {
  const show = minimalShow();
  show.locations.room.soundscape.ambience = "soft choir music under the arches";
  const result = parseShowScript(show);
  assert.equal(result.ok, true);
  assert.ok(result.issues.some((issue) => issue.severity === "warning" && issue.path.endsWith("sound.music")));
});

test("every landmark has position, size, and appearance", () => {
  const show = minimalShow();
  (show.locations.room.spatial.landmarks as Record<string, unknown>).bench = { position: [0, 1, 0] };
  const report = messages(show);
  assert.match(report, /bench\.size: size is required/);
  assert.match(report, /bench\.appearance: appearance is required/);
});

test("rejects removed fields", () => {
  const show = minimalShow();
  (show.locations.room as { promptBlock?: string }).promptBlock = "A room.";
  (show.episodes[0].scenes[0] as { characterIds?: string[] }).characterIds = ["ada"];
  (show.characters.ada as { proxy?: unknown }).proxy = { heightMeters: 1.7, build: "slim" };
  (show.episodes[0].spatialTimeline.characterTracks.ada[0] as { stance?: string }).stance = "standing";
  const report = messages(show);
  for (const field of ["promptBlock", "characterIds", "proxy", "stance"]) {
    assert.match(report, new RegExp(field));
  }
});

test("every character has a track that starts at 0", () => {
  const missing = minimalShow();
  missing.episodes[0].spatialTimeline.characterTracks = {};
  missing.episodes[0].scenes[0].performances = {};
  missing.episodes[0].scenes[0].sound.events = [{ text: "the bench creaks as it settles", source: { landmarkId: "bench" } }];
  assert.match(messages(missing), /characterTracks\.ada: every character has a track/);
  const late = minimalShow();
  late.episodes[0].spatialTimeline.characterTracks.ada[0].timeSeconds = 1;
  assert.match(messages(late), /ada\[0\]\.timeSeconds: a track starts at 0/);
});

test("an off-stage keyframe has no pose, and an on-stage one needs one", () => {
  const show = minimalShow();
  show.episodes[0].spatialTimeline.characterTracks.ada = [
    { timeSeconds: 0, locationId: null, position: [0, 0, 0] },
    { timeSeconds: 2, locationId: "room" },
  ];
  const report = messages(show);
  assert.match(report, /ada\[0\]\.position: an off-stage keyframe/);
  assert.match(report, /ada\[1\]\.position: position is required in a location/);
  assert.match(report, /ada\[1\]\.bodyYawDegrees: bodyYawDegrees is required/);
});

test("accepts a character who enters from off stage", () => {
  const show = minimalShow();
  show.episodes[0].spatialTimeline.characterTracks.ada = [
    { timeSeconds: 0, locationId: null },
    { timeSeconds: 2, locationId: "room", position: [0, 0, 0], bodyYawDegrees: 0 },
  ];
  assert.equal(parseShowScript(show).ok, true);
});

test("a prop is held, placed, or off stage, and has a track", () => {
  const show = minimalShow({ props: { cup: { appearance: "One clay cup", size: [0.1, 0.1, 0.1] } } });
  assert.match(messages(show), /propTracks\.cup: every prop has a track/);
  show.episodes[0].spatialTimeline.propTracks = {
    cup: [
      { timeSeconds: 0, heldByCharacterId: "ada", heldInHand: "left" },
      { timeSeconds: 2, locationId: "room", position: [0, 1, 0.5] },
      { timeSeconds: 3, locationId: null },
    ],
  };
  assert.equal(parseShowScript(show).ok, true);
  (show.episodes[0].spatialTimeline.propTracks.cup[0] as { locationId?: string }).locationId = "room";
  assert.match(messages(show), /propTracks\.cup\[0\]: Unrecognized key/);
});

test("rejects a prop track whose id was never declared", () => {
  const show = minimalShow();
  show.episodes[0].spatialTimeline.propTracks = {
    iron_collar: [{ timeSeconds: 0, heldByCharacterId: "ada", heldInHand: "left" }],
  };
  assert.match(messages(show), /propTracks\.iron_collar: prop "iron_collar" is not declared/);
});

test("a prop needs appearance and size", () => {
  const show = minimalShow({ props: { cup: { size: [0.1, 0.1, 0.1] } as never } });
  assert.match(messages(show), /props\.cup\.appearance: appearance is required/);
});

test("the speaker needs a performance with an expression and a line", () => {
  const show = speaking(minimalShow());
  assert.equal(parseShowScript(show).ok, true);
  delete show.episodes[0].scenes[0].performances.ada.expression;
  assert.match(messages(show), /performances\.ada\.expression: the speaker needs an expression/);
  show.episodes[0].scenes[0].performances = {};
  assert.match(messages(show), /speakerId: the speaker needs a performance/);
});

test("requires dialogue exactly when someone speaks", () => {
  const show = speaking(minimalShow());
  delete show.episodes[0].scenes[0].dialogue;
  assert.match(messages(show), /scenes\[0\]\.dialogue: required when speakerId is set/);
  const silent = minimalShow();
  silent.episodes[0].scenes[0].dialogue = { line: "Hello.", delivery: "soft" };
  assert.match(messages(silent), /scenes\[0\]\.dialogue: dialogue needs a speakerId/);
});

test("rejects a performance for an unknown character or with an unknown part", () => {
  const show = minimalShow();
  show.episodes[0].scenes[0].performances.bram = { action: "waits", parts: ["torso"] };
  assert.match(messages(show), /performances\.bram: unknown character "bram"/);
  const odd = minimalShow();
  odd.episodes[0].scenes[0].performances.ada.parts = ["toes" as "feet"];
  assert.match(messages(odd), /performances\.ada\.parts\[0\]: unknown part id "toes"/);
});

test("accepts stillness: a performance that moves no part", () => {
  const show = minimalShow();
  const scene = show.episodes[0].scenes[0];
  scene.performances.ada = { action: "holds still, watching", parts: [] };
  scene.sound.events = [{ text: "the bench creaks as it settles", source: { landmarkId: "bench" } }];
  assert.equal(parseShowScript(show).ok, true);
});

test("rejects a body sound from a part the performance does not move", () => {
  const show = minimalShow();
  show.episodes[0].scenes[0].sound.events[0].source = { characterId: "ada", part: "face" };
  assert.match(messages(show), /ada's face makes this sound, but performances\.ada\.parts does not move it/);
});

test("lets a speaker's face make a sound", () => {
  const show = speaking(minimalShow());
  show.episodes[0].scenes[0].sound.events[0] = {
    text: "a sharp breath before the word",
    source: { characterId: "ada", part: "face" },
  };
  assert.equal(parseShowScript(show).ok, true);
});

test("rejects unknown landmark and prop sound sources", () => {
  const show = minimalShow();
  const events = show.episodes[0].scenes[0].sound.events;
  events.push({ text: "wood creaks under a weight", source: { landmarkId: "altar" } });
  events.push({ text: "a lantern rattles on its hook", source: { propId: "lantern" } });
  const report = messages(show);
  assert.match(report, /events\[1\]\.source\.landmarkId: unknown landmark "altar"/);
  assert.match(report, /events\[2\]\.source\.propId: unknown prop "lantern"/);
});

test("rejects an object written both as an attribute and as a prop", () => {
  const show = minimalShow({ props: { iron_collar: { appearance: "One iron torc", size: [0.28, 0.22, 0.08] } } });
  show.characters.ada.attributes.push("a plain iron bridal collar");
  show.episodes[0].spatialTimeline.propTracks = {
    iron_collar: [{ timeSeconds: 0, heldByCharacterId: "ada", heldInHand: "left" }],
  };
  assert.match(messages(show), /characters\.ada\.attributes\[3\]: describes prop "iron_collar"/);
});

test("an effect is never part of a solid appearance", () => {
  const show = minimalShow();
  show.locations.room.spatial.landmarks.bench.appearance = "One stone fire ring with a bowl of white-gold flame";
  show.characters.ada.attributes.push("glowing eyes");
  const report = messages(show);
  assert.match(report, /landmarks\.bench\.appearance: "fire" is not a solid surface/);
  assert.match(report, /characters\.ada\.attributes\[3\]: "glowing" is not a solid surface/);
});

test("accepts effects on a landmark, a prop, and a character", () => {
  const flame = { appearance: "white-gold flames", size: [0.8, 0.8, 1] as [number, number, number], offset: [0, 0, 0.4] as [number, number, number] };
  const show = minimalShow({ props: { lamp: { appearance: "One iron lamp", size: [0.2, 0.2, 0.3], effects: [flame] } } });
  show.locations.room.spatial.landmarks.bench.appearance = "One low stone ring around an empty iron bowl";
  show.locations.room.spatial.landmarks.bench.effects = [flame];
  show.characters.ada.effects = [{ ...flame, appearance: "breath fogging in the cold" }];
  show.episodes[0].spatialTimeline.propTracks = { lamp: [{ timeSeconds: 0, locationId: null }] };
  const result = parseShowScript(show);
  assert.ok(result.ok, result.ok ? "" : result.issues.map((issue) => `${issue.path}: ${issue.message}`).join("\n"));
});

test("rejects tagged attributes; a detail is plain text", () => {
  const show = minimalShow();
  (show.characters.ada.attributes as unknown[]) = [{ text: "black hair", parts: ["hair"] }];
  assert.match(messages(show), /characters\.ada\.attributes\[0\]: attribute must be a string/);
});

test("rejects a show id that does not match the file", () => {
  const result = parseShowScript(minimalShow(), { showId: "other-show", showIdLabel: "the filename stem" });
  assert.equal(result.ok, false);
  assert.match(result.issues[0].message, /other-show/);
});

test("every shot of people is framed from its subjects; only establishing and action set a camera", () => {
  const show = minimalShow();
  const scene = show.episodes[0].scenes[0];
  scene.camera = { keyframes: [{ timeSeconds: 0, position: [0, -3, 1.5], lookAt: [0, 0, 1.5], verticalFovDegrees: 40 }] };
  assert.match(messages(show), /camera: a single shot gets its camera from its subjects/);
  delete scene.camera;
  scene.shot = { type: "single", subjects: [] };
  assert.match(messages(show), /shot.subjects: a single shot frames 1 subject/);
  scene.shot = { type: "single", subjects: ["ada"], size: "wide" };
  assert.match(messages(show), /shot.size: a single shot is ecu, cu, mcu, medium, full/);
  scene.shot = { type: "overShoulder", subjects: ["ada"], over: "ada" };
  assert.match(messages(show), /shot.over: the near shoulder belongs to someone other than the subject/);
  scene.shot = { type: "insert", subjects: ["ada"] };
  assert.match(messages(show), /shot.part: an insert of a character shows one part/);
  scene.shot = { type: "establishing" };
  assert.match(messages(show), /camera: an establishing shot needs its own camera/);
});

test("rejects visual text that names a character", () => {
  const show = minimalShow();
  show.episodes[0].scenes[0].performances.ada.action = "stands still, watching Ada's reflection";
  assert.match(messages(show), /performances\.ada\.action: names character "ada"/);
});

test("rejects visual text that points at someone, and aims an action only through {target}", () => {
  const show = minimalShow();
  const ada = show.episodes[0].scenes[0].performances.ada;
  ada.action = "leans forward over her, glaring down";
  assert.match(messages(show), /performances\.ada\.action: "her" points at someone or something/);
  ada.action = "glares down at {target}";
  assert.match(messages(show), /uses \{target\}, so the performance needs a target/);
  ada.target = "ada";
  assert.match(messages(show), /target: an action is aimed at someone or something else/);
  ada.target = "nobody";
  assert.match(messages(show), /target: unknown target "nobody"/);
  ada.action = "glares down";
  ada.target = "bench";
  assert.match(messages(show), /target: the action must say where it is aimed, as \{target\}/);
});
