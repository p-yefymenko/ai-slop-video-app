/**
 * `pnpm content:validate` — check show JSON against the ShowScript schema.
 *
 * Discovers content-pipeline/scripts_input/<id>.json and
 * content-pipeline/shows/<id>/script.json. Pass file paths to check those instead.
 */

import { existsSync, readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { formatScriptReport, parseShowScript } from "./script-schema";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");

type ShowFile = {
  file: string;
  showId: string;
  showIdLabel: string;
};

function discoverShows(root: string): ShowFile[] {
  const found: ShowFile[] = [];
  const inputDir = path.join(root, "content-pipeline", "scripts_input");
  if (existsSync(inputDir)) {
    for (const name of readdirSync(inputDir)) {
      if (!name.endsWith(".json")) {
        continue;
      }
      found.push({
        file: path.join(inputDir, name),
        showId: name.slice(0, -".json".length),
        showIdLabel: "the filename stem",
      });
    }
  }
  const showsDir = path.join(root, "content-pipeline", "shows");
  if (existsSync(showsDir)) {
    for (const entry of readdirSync(showsDir, { withFileTypes: true })) {
      if (!entry.isDirectory()) {
        continue;
      }
      const file = path.join(showsDir, entry.name, "script.json");
      if (!existsSync(file)) {
        continue;
      }
      found.push({
        file,
        showId: entry.name,
        showIdLabel: "the show folder",
      });
    }
  }
  return found.sort((a, b) => a.file.localeCompare(b.file));
}

function showFileFromArg(arg: string): ShowFile {
  const file = path.resolve(arg);
  const base = path.basename(file);
  if (base === "script.json") {
    const showId = path.basename(path.dirname(file));
    return { file, showId, showIdLabel: "the show folder" };
  }
  const showId = base.endsWith(".json") ? base.slice(0, -".json".length) : base;
  return { file, showId, showIdLabel: "the filename stem" };
}

function validateFile(showFile: ShowFile): boolean {
  const label = path.relative(repoRoot, showFile.file) || showFile.file;
  let data: unknown;
  try {
    data = JSON.parse(readFileSync(showFile.file, "utf8"));
  } catch (error) {
    const message = error instanceof SyntaxError ? error.message : "could not read file";
    console.error(formatScriptReport(label, [{ path: "(file)", message: `invalid JSON (${message})` }]));
    return false;
  }
  const result = parseShowScript(data, {
    showId: showFile.showId,
    showIdLabel: showFile.showIdLabel,
  });
  const report = formatScriptReport(label, result.issues);
  if (result.ok) {
    console.log(report);
    return true;
  }
  console.error(report);
  return false;
}

function main() {
  const args = process.argv.slice(2);
  const files = args.length > 0 ? args.map(showFileFromArg) : discoverShows(repoRoot);
  if (files.length === 0) {
    console.error("No show scripts found in content-pipeline/scripts_input or content-pipeline/shows.");
    process.exit(1);
  }

  const byId = new Map<string, string[]>();
  for (const showFile of files) {
    const list = byId.get(showFile.showId) ?? [];
    list.push(showFile.file);
    byId.set(showFile.showId, list);
  }
  let ok = true;
  for (const [showId, paths] of byId) {
    if (paths.length < 2) {
      continue;
    }
    ok = false;
    console.error(
      `Show ${JSON.stringify(showId)} has more than one script:\n${paths.map((item) => `  ${item}`).join("\n")}`,
    );
  }
  if (!ok) {
    process.exit(1);
  }

  for (const showFile of files) {
    if (!existsSync(showFile.file)) {
      console.error(formatScriptReport(showFile.file, [{ path: "(file)", message: "file not found" }]));
      ok = false;
      continue;
    }
    if (!validateFile(showFile)) {
      ok = false;
    }
  }
  if (!ok) {
    process.exit(1);
  }
  console.log(`Validated ${files.length} show script${files.length === 1 ? "" : "s"}.`);
}

main();
