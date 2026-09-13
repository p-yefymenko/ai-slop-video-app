"use strict";

const { spawn } = require("node:child_process");

const ES_CONTINUOUS = 0x80000000;
const ES_SYSTEM_REQUIRED = 0x00000001;
const FLAGS = ES_CONTINUOUS | ES_SYSTEM_REQUIRED;

function powershellStayAwakeScript() {
  return `
Add-Type -TypeDefinition @"
using System.Runtime.InteropServices;
public static class StayAwake {
  [DllImport("kernel32.dll")]
  public static extern uint SetThreadExecutionState(uint esFlags);
}
"@
$flags = [uint32]${FLAGS}
[void][StayAwake]::SetThreadExecutionState($flags)
try {
  while ($true) {
    Start-Sleep -Seconds 30
    [void][StayAwake]::SetThreadExecutionState($flags)
  }
} finally {
  [void][StayAwake]::SetThreadExecutionState([uint32]${ES_CONTINUOUS})
}
`.trim();
}

function startStayAwake() {
  if (process.platform !== "win32") {
    return () => {};
  }
  const encoded = Buffer.from(powershellStayAwakeScript(), "utf16le").toString("base64");
  const child = spawn(
    "powershell.exe",
    ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-EncodedCommand", encoded],
    { stdio: "ignore", windowsHide: true },
  );
  let stopped = false;
  const stop = () => {
    if (stopped) {
      return;
    }
    stopped = true;
    if (!child.killed) {
      child.kill();
    }
  };
  child.on("error", (err) => {
    console.warn(`Could not inhibit Windows sleep (${err.message}).`);
  });
  process.on("exit", stop);
  return stop;
}

module.exports = { startStayAwake };
