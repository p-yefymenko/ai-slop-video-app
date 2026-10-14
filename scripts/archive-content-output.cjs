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
  archiveDirectory(source, destination);
  console.log(`Archived ${source}`);
  console.log(`      to ${destination}`);
}

if (!found) {
  console.log(`No generated output exists for ${showId}.`);
  process.exit(0);
}

function archiveDirectory(source, destination) {
  try {
    fs.renameSync(source, destination);
  } catch (error) {
    if (error.code !== "EPERM") throw error;
    fs.cpSync(source, destination, { recursive: true });
    fs.rmSync(source, { recursive: true, force: true });
  }
}
