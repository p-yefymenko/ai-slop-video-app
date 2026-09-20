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
try {
  fs.renameSync(source, destination);
  const archivedCharacters = path.join(destination, "characters");
  if (fs.existsSync(archivedCharacters)) {
    const activeCharacters = path.join(source, "characters");
    fs.mkdirSync(source, { recursive: true });
    fs.cpSync(archivedCharacters, activeCharacters, { recursive: true });
  }
} catch (error) {
  if (error.code !== "EPERM") {
    throw error;
  }
  // Windows can refuse to rename a directory while Cursor previews a file in it.
  // Copy the archive, then clear only generated episode folders in place.
  fs.cpSync(source, destination, { recursive: true });
  for (const entry of fs.readdirSync(source, { withFileTypes: true })) {
    if (entry.name === "characters") {
      continue;
    }
    fs.rmSync(path.join(source, entry.name), { recursive: true, force: true });
  }
}
console.log(`Archived ${source}`);
console.log(`      to ${destination}`);
if (fs.existsSync(path.join(source, "characters"))) {
  console.log("Preserved reviewed character identity PNGs in the active output folder.");
}
