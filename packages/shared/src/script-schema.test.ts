import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import type { ShowScript } from "./script";
import { parseShowScript } from "./script-schema";

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
        voice: "a young woman's voice, warm and clear",
      },
    },
    locations: {
      room: {
        look: { light: "soft overcast daylight" },
        backdrop: {
          sky: "a gray sky",
          skyColor: [40, 48, 64],
          ground: "stone paving",
          groundColor: [48, 44, 40],
          surround: "open fields",
          surroundColor: [24, 56, 40],
          surroundHeightMeters: 0,
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
              ada: { action: "stands still, then shifts the weight once" },
            },
            shot: { type: "single", subjects: ["ada"], size: "full" },
            camera: { keyframes: [{ timeSeconds: 0, position: [0, 4, 1.5], lookAt: [0, 0, 1], verticalFovDegrees: 40 }] },
            sound: {
              events: [
                { text: "dress fabric rustles as she shifts her weight", source: { characterId: "ada" } },
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
  scene.sound.bed = "faint";
  scene.lines = [{ speakerId: "ada", line: "Hello.", delivery: "soft" }];
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

test("a prop is placed or off stage, and has a track", () => {
  const show = minimalShow({ props: { cup: { appearance: "One clay cup", size: [0.1, 0.1, 0.1] } } });
  assert.match(messages(show), /propTracks\.cup: every prop has a track/);
  show.episodes[0].spatialTimeline.propTracks = {
    cup: [
      { timeSeconds: 0, locationId: "room", position: [0, 1, 0.5] },
      { timeSeconds: 3, locationId: null },
    ],
  };
  assert.equal(parseShowScript(show).ok, true);
  (show.episodes[0].spatialTimeline.propTracks.cup[0] as { heldByCharacterId?: string }).heldByCharacterId = "ada";
  assert.match(messages(show), /propTracks\.cup\[0\]: Unrecognized key/);
});

test("rejects a prop track whose id was never declared", () => {
  const show = minimalShow();
  show.episodes[0].spatialTimeline.propTracks = {
    iron_collar: [{ timeSeconds: 0, locationId: null }],
  };
  assert.match(messages(show), /propTracks\.iron_collar: prop "iron_collar" is not declared/);
});

test("a prop needs appearance and size", () => {
  const show = minimalShow({ props: { cup: { size: [0.1, 0.1, 0.1] } as never } });
  assert.match(messages(show), /props\.cup\.appearance: appearance is required/);
});

test("a line's speaker is a known character", () => {
  const show = speaking(minimalShow());
  assert.equal(parseShowScript(show).ok, true);
  const unknown = speaking(minimalShow());
  unknown.episodes[0].scenes[0].lines = [{ speakerId: "bram", line: "Hello.", delivery: "soft" }];
  assert.match(messages(unknown), /lines\[0\]\.speakerId: unknown character "bram"/);
});

test("a speaker has a voice", () => {
  const show = speaking(minimalShow());
  delete show.characters.ada.voice;
  assert.match(messages(show), /lines\[0\]\.speakerId: ada speaks but has no voice/);
});

test("a line starts within its scene", () => {
  const show = speaking(minimalShow());
  show.episodes[0].scenes[0].lines![0].startSeconds = 99;
  assert.match(messages(show), /lines\[0\]\.startSeconds: must be within the scene/);
});

test("a camera that moves says how in words", () => {
  const show = minimalShow();
  const scene = show.episodes[0].scenes[0];
  scene.shot = { type: "establishing" };
  scene.performances = {};
  const pose = (t: number, y: number) => ({ timeSeconds: t, position: [0, y, 1.6] as [number, number, number], lookAt: [0, 5, 1.2] as [number, number, number], verticalFovDegrees: 40 });
  scene.camera = { keyframes: [pose(0, -4), pose(1, -3)] };
  assert.match(messages(show), /cameraMotion: a camera with more than one keyframe needs cameraMotion/);
  scene.cameraMotion = "glides forward";
  const result = parseShowScript(show);
  assert.ok(result.ok || !result.issues.some((issue) => issue.path.endsWith("cameraMotion")));
});

test("rejects a performance for an unknown character", () => {
  const show = minimalShow();
  show.episodes[0].scenes[0].performances.bram = { action: "waits" };
  assert.match(messages(show), /performances\.bram: unknown character "bram"/);
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

test("an attribute is plain text, or marked as a detail of the face; nothing else", () => {
  const show = minimalShow();
  show.characters.ada.attributes = [{ face: "brown eyes" }, "black hair"];
  assert.equal(parseShowScript(show, { showId: "demo-show", showIdLabel: "the filename stem" }).ok, true);
  (show.characters.ada.attributes as unknown[]) = [{ text: "black hair", parts: ["hair"] }];
  assert.match(messages(show), /characters\.ada\.attributes\[0\]/);
});

test("rejects a show id that does not match the file", () => {
  const result = parseShowScript(minimalShow(), { showId: "other-show", showIdLabel: "the filename stem" });
  assert.equal(result.ok, false);
  assert.match(result.issues[0].message, /other-show/);
});

test("every scene authors its camera; a shot names its subjects and size", () => {
  const show = minimalShow();
  const scene = show.episodes[0].scenes[0];
  delete (scene as { camera?: unknown }).camera;
  assert.match(messages(show), /camera: camera is required/);
  scene.camera = { keyframes: [{ timeSeconds: 0, position: [0, 4, 1.5], lookAt: [0, 0, 1], verticalFovDegrees: 40 }] };
  scene.shot = { type: "single", subjects: [] };
  assert.match(messages(show), /shot.subjects: a single shot has 1 subject/);
  scene.shot = { type: "single", subjects: ["ada"], size: "wide" };
  assert.match(messages(show), /shot.size: a single shot is ecu, cu, mcu, medium, full/);
  (scene.shot as { over?: string }).over = "ada";
  assert.match(messages(show), /shot: Unrecognized key/);
});

test("what a shot shows of someone partly in it is their own attributes, word for word", () => {
  const show = minimalShow();
  const scene = show.episodes[0].scenes[0];
  scene.shot = { type: "single", subjects: ["ada"], size: "full", partlySeen: { ada: ["a red cloak"] } };
  assert.match(messages(show), /shot.partlySeen.ada\[0\]: "a red cloak" is not one of ada's attributes/);
  scene.shot.partlySeen = { nobody: [] };
  assert.match(messages(show), /shot.partlySeen.nobody: unknown character/);
});

test("an action is aimed only through {target}", () => {
  const show = minimalShow();
  const ada = show.episodes[0].scenes[0].performances.ada;
  ada.action = "glares down at {target}";
  assert.match(messages(show), /uses \{target\}, so the performance needs a target/);
  ada.target = "ada";
  assert.match(messages(show), /target: an action is aimed at someone else/);
  ada.target = "nobody";
  assert.match(messages(show), /target: unknown target "nobody"/);
  ada.action = "glares down";
  ada.target = "bench";
  assert.match(messages(show), /target: the action must say where it is aimed, as \{target\}/);
});

test("an action is aimed only at someone who performs in the scene", () => {
  const show = minimalShow();
  show.characters.bo = { ...show.characters.ada };
  const ada = show.episodes[0].scenes[0].performances.ada;
  ada.action = "glares down at {target}";
  ada.target = "bo";
  assert.match(messages(show), /target: "bo" has no performance in this scene/);
});
