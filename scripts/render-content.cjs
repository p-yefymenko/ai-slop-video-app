const { spawnSync } = require("node:child_process");

const extraArgs = process.argv.slice(2);
const stages = [
  {
    name: "content:previs",
    args: ["scripts/run-python.cjs", "content-pipeline/scripts/spatial_previs.py"],
  },
  {
    name: "content:frames",
    args: [
      "scripts/run-python.cjs",
      "content-pipeline/scripts/generate_batch.py",
      "--stage",
      "frames",
    ],
  },
  {
    name: "content:generate",
    args: [
      "scripts/run-python.cjs",
      "content-pipeline/scripts/generate_batch.py",
      "--stage",
      "video",
    ],
  },
];

for (const stage of stages) {
  console.log(`\n=== ${stage.name} ===\n`);
  const result = spawnSync(process.execPath, [...stage.args, ...extraArgs], {
    cwd: process.cwd(),
    env: process.env,
    stdio: "inherit",
  });
  if (result.error) {
    console.error(result.error.message);
    process.exit(1);
  }
  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
}
