const { spawn } = require("node:child_process");
const path = require("node:path");

const args = process.argv.slice(2);
let show = "";
for (let index = 0; index < args.length; index += 1) {
  const arg = args[index];
  if (arg === "--show") {
    show = args[index + 1] || "";
    index += 1;
  } else if (arg.startsWith("--show=")) {
    show = arg.slice("--show=".length);
  }
}

if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(show)) {
  console.error("Usage: pnpm run view -- --show <show-id>");
  process.exit(1);
}

const child = spawn("pnpm", ["--filter", "@reelshort/viewer", "run", "dev"], {
  stdio: "inherit",
  shell: true,
  cwd: path.resolve(__dirname, ".."),
  env: { ...process.env, VIEW_SHOW: show },
});

child.on("exit", (code) => {
  process.exit(code ?? 0);
});
