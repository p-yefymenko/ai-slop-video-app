import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { buildCatalog, pickNeed, resolvePipelineFile } from "./pipeline-api.js";

test("pipeline files stay inside prefabs, show assets, and output", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "stage-"));
  assert.equal(resolvePipelineFile(root, "library/raw/secret.glb"), null);
  assert.equal(resolvePipelineFile(root, "library/prefabs/../../.env"), null);
  assert.equal(resolvePipelineFile(root, "output/the-iron-bride/sets/room/set.glb"), path.resolve(root, "output/the-iron-bride/sets/room/set.glb"));
});

test("catalog lists a prefab and a set", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "stage-"));
  const prefabDir = path.join(root, "library", "prefabs", "blocks", "chair");
  fs.mkdirSync(prefabDir, { recursive: true });
  fs.writeFileSync(path.join(prefabDir, "model.glb"), "glb");
  fs.writeFileSync(
    path.join(prefabDir, "prefab.json"),
    JSON.stringify({ id: "blocks/chair", title: "Chair", license: "CC0", origin: "library" }),
  );
  const setDir = path.join(root, "output", "demo", "sets", "room");
  fs.mkdirSync(setDir, { recursive: true });
  fs.writeFileSync(path.join(setDir, "set.json"), JSON.stringify({ locationId: "room", instances: [] }));
  const catalog = buildCatalog(root);
  assert.equal(catalog.prefabs[0].id, "blocks/chair");
  assert.equal(catalog.prefabs[0].modelUrl, "/pipeline/library/prefabs/blocks/chair/model.glb");
  assert.equal(catalog.sets[0].showId, "demo");
  assert.equal(catalog.sets[0].locationId, "room");
});

test("a review pick overrides the lock", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "stage-"));
  fs.mkdirSync(path.join(root, "library"), { recursive: true });
  const digest = "a".repeat(64);
  fs.writeFileSync(path.join(root, "library", "lock.json"), JSON.stringify({ version: 1, needs: {} }));
  const entry = pickNeed(root, { needHash: digest, prefabId: "blocks/picked", source: "fake" });
  assert.equal(entry.origin, "override");
  assert.equal(entry.provisional, false);
  const stored = JSON.parse(fs.readFileSync(path.join(root, "library", "lock.json"), "utf8"));
  assert.equal(stored.needs[digest].prefabId, "blocks/picked");
});
