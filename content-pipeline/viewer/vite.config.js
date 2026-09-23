import path from "node:path";
import { fileURLToPath } from "node:url";

import { defineConfig } from "vite";

import { pipelineApi } from "./pipeline-api.js";

const pipelineRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

export default defineConfig({
  plugins: [pipelineApi(pipelineRoot)],
  server: {
    host: "127.0.0.1",
    port: 5174,
    strictPort: true,
  },
});
