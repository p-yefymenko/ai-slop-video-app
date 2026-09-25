const fs = require("node:fs");
const path = require("node:path");

const showId = process.argv[2];
if (!showId || !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(showId)) {
  console.error("Usage: pnpm run content:archive -- <show-id>");
  process.exit(1);
}

const root = path.resolve(__dirname, "..");
const outputRoot = path.join(root, "content-pipeline", "output");
const stages = ["plates", "assets", "previs", "frames", "generate"];
const stamp = new Date().toISOString().replace(/[:.]/g, "-");
let found = false;

for (const stage of stages) {
  const source = path.join(outputRoot, stage, showId);
  if (!fs.existsSync(source)) continue;
  found = true;
  const destination = path.join(outputRoot, stage, `${showId}.archive-${stamp}`);
  archiveDirectory(source, destination, stage === "frames");
  console.log(`Archived ${source}`);
  console.log(`      to ${destination}`);
}

const legacy = path.join(outputRoot, showId);
if (fs.existsSync(legacy) && !stages.includes(showId)) {
  found = true;
  const destination = path.join(outputRoot, `${showId}.archive-${stamp}`);
  archiveDirectory(legacy, destination, true);
  console.log(`Archived ${legacy}`);
  console.log(`      to ${destination}`);
}

if (!found) {
  console.log(`No generated output exists for ${showId}.`);
  process.exit(0);
}

function archiveDirectory(source, destination, keepCharacters) {
  try {
    fs.renameSync(source, destination);
    if (!keepCharacters) return;
    const archivedCharacters = path.join(destination, "characters");
    if (!fs.existsSync(archivedCharacters)) return;
    const activeCharacters = path.join(source, "characters");
    fs.mkdirSync(source, { recursive: true });
    fs.cpSync(archivedCharacters, activeCharacters, { recursive: true });
    console.log("Preserved reviewed character identity PNGs in the active output folder.");
  } catch (error) {
    if (error.code !== "EPERM") throw error;
    fs.cpSync(source, destination, { recursive: true });
    for (const entry of fs.readdirSync(source, { withFileTypes: true })) {
      if (keepCharacters && entry.name === "characters") continue;
      fs.rmSync(path.join(source, entry.name), { recursive: true, force: true });
    }
  }
}
