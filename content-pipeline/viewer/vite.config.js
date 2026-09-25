import path from "node:path";
import { fileURLToPath } from "node:url";

import { defineConfig } from "vite";

import { pipelineApi } from "./pipeline-api.js";

const pipelineRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const showId = process.env.VIEW_SHOW || "";

export default defineConfig({
  plugins: [pipelineApi(pipelineRoot, showId)],
  server: {
    host: "127.0.0.1",
    port: 5174,
    strictPort: true,
  },
});
