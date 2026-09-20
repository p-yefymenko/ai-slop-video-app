const fs = require("node:fs");
const path = require("node:path");

const showId = process.argv[2];
if (!showId || !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(showId)) {
  console.error("Usage: pnpm run content:archive -- <show-id>");
  process.exit(1);
}

const root = path.resolve(__dirname, "..");
const outputRoot = path.join(root, "content-pipeline", "output");
const source = path.join(outputRoot, showId);
if (!fs.existsSync(source)) {
  console.log(`No generated output exists for ${showId}.`);
  process.exit(0);
}

const stamp = new Date().toISOString().replace(/[:.]/g, "-");
const destination = path.join(outputRoot, `${showId}.archive-${stamp}`);
fs.renameSync(source, destination);
const archivedCharacters = path.join(destination, "characters");
if (fs.existsSync(archivedCharacters)) {
  const activeCharacters = path.join(source, "characters");
  fs.mkdirSync(source, { recursive: true });
  fs.cpSync(archivedCharacters, activeCharacters, { recursive: true });
}
console.log(`Archived ${source}`);
console.log(`      to ${destination}`);
if (fs.existsSync(archivedCharacters)) {
  console.log("Preserved reviewed character identity PNGs in the active output folder.");
}
