import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { buildCatalog, resolvePipelineFile } from "./pipeline-api.js";

test("pipeline files stay inside generated output", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "stage-"));
  assert.equal(resolvePipelineFile(root, "library/raw/secret.glb"), null);
  assert.equal(resolvePipelineFile(root, "output/../.env"), null);
  assert.equal(
    resolvePipelineFile(root, "output/assets/the-iron-bride/room/model.glb"),
    path.resolve(root, "output/assets/the-iron-bride/room/model.glb"),
  );
});

test("catalog lists one show's locations and scenes", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "stage-"));
  const room = path.join(root, "output", "assets", "demo", "room");
  fs.mkdirSync(room, { recursive: true });
  fs.writeFileSync(path.join(room, "model.glb"), "glb");
  fs.writeFileSync(
    path.join(room, "location.json"),
    JSON.stringify({ locationId: "room", title: "Room", sizeMeters: [8, 10, 4] }),
  );
  const plate = path.join(root, "output", "plates", "demo", "court");
  fs.mkdirSync(plate, { recursive: true });
  fs.writeFileSync(path.join(plate, "plate.png"), "png");
  const scene = path.join(root, "output", "previs", "demo", "1", "scene_02");
  fs.mkdirSync(scene, { recursive: true });
  fs.writeFileSync(path.join(scene, "scene.json"), "{}");
  const other = path.join(root, "output", "assets", "other", "hall");
  fs.mkdirSync(other, { recursive: true });
  fs.writeFileSync(path.join(other, "model.glb"), "glb");

  const catalog = buildCatalog(root, "demo");
  assert.equal(catalog.showId, "demo");
  assert.deepEqual(
    catalog.locations.map((item) => item.locationId),
    ["court", "room"],
  );
  assert.equal(catalog.locations[1].modelUrl, "/pipeline/output/assets/demo/room/model.glb");
  assert.equal(catalog.locations[0].plateUrl, "/pipeline/output/plates/demo/court/plate.png");
  assert.equal(catalog.scenes[0].episodeNumber, 1);
  assert.equal(catalog.scenes[0].sceneNumber, 2);
  assert.equal(buildCatalog(root, "").locations.length, 0);
});
