import fs from "node:fs";
import path from "node:path";

const ALLOWED = [/^output\/.+\.(glb|gltf|png|json|jpe?g|mp4)$/];

export function resolvePipelineFile(pipelineRoot, relative) {
  const normalized = String(relative || "").replaceAll("\\", "/").replace(/^\/+/, "");
  if (!normalized || normalized.includes("..")) return null;
  if (!ALLOWED.some((pattern) => pattern.test(normalized))) return null;
  const root = path.resolve(pipelineRoot);
  const absolute = path.resolve(root, normalized);
  const prefix = root.endsWith(path.sep) ? root : root + path.sep;
  if (absolute !== root && !absolute.startsWith(prefix)) return null;
  return absolute;
}

export function buildCatalog(pipelineRoot, showId) {
  if (!showId) return { showId: null, locations: [], scenes: [] };
  const locations = new Map();
  const assetRoot = path.join(pipelineRoot, "output", "assets", showId);
  for (const directory of childDirs(assetRoot)) {
    const locationId = path.basename(directory);
    if (locationId === "props") continue;
    const recordPath = path.join(directory, "location.json");
    const model = path.join(directory, "model.glb");
    const record = fs.existsSync(recordPath) ? JSON.parse(fs.readFileSync(recordPath, "utf8")) : {};
    const landmarks = [];
    for (const child of childDirs(directory)) {
      const childModel = path.join(child, "model.glb");
      const childRecordPath = path.join(child, "landmark.json");
      if (!fs.existsSync(childModel) && !fs.existsSync(childRecordPath)) continue;
      const childRecord = fs.existsSync(childRecordPath)
        ? JSON.parse(fs.readFileSync(childRecordPath, "utf8"))
        : {};
      landmarks.push({
        id: childRecord.landmarkId || path.basename(child),
        modelUrl: fs.existsSync(childModel) ? fileUrl(pipelineRoot, childModel) : null,
        position: childRecord.position || null,
      });
    }
    locations.set(locationId, {
      locationId,
      title: record.title || locationId,
      sizeMeters: record.sizeMeters || null,
      modelUrl: landmarks.length || !fs.existsSync(model) ? null : fileUrl(pipelineRoot, model),
      plateUrl: null,
      landmarks,
    });
  }
  const plateRoot = path.join(pipelineRoot, "output", "plates", showId);
  for (const directory of childDirs(plateRoot)) {
    const locationId = path.basename(directory);
    const plate = path.join(directory, "plate.png");
    const entry = locations.get(locationId) || {
      locationId,
      title: locationId,
      sizeMeters: null,
      modelUrl: null,
      plateUrl: null,
      landmarks: [],
    };
    if (fs.existsSync(plate)) entry.plateUrl = fileUrl(pipelineRoot, plate);
    locations.set(locationId, entry);
  }
  const scenes = [];
  const previsRoot = path.join(pipelineRoot, "output", "previs", showId);
  for (const episodeDir of childDirs(previsRoot)) {
    if (!/^\d+$/.test(path.basename(episodeDir))) continue;
    for (const sceneDir of childDirs(episodeDir)) {
      if (!/^scene_\d+$/.test(path.basename(sceneDir))) continue;
      const file = path.join(sceneDir, "scene.json");
      if (!fs.existsSync(file)) continue;
      scenes.push({
        episodeNumber: Number(path.basename(episodeDir)),
        sceneNumber: Number(path.basename(sceneDir).slice("scene_".length)),
        locationId: null,
        url: fileUrl(pipelineRoot, file),
      });
    }
  }
  scenes.sort((a, b) => a.episodeNumber - b.episodeNumber || a.sceneNumber - b.sceneNumber);
  const listed = [...locations.values()].sort((a, b) => a.locationId.localeCompare(b.locationId));
  return { showId, locations: listed, scenes };
}

export function pipelineApi(pipelineRoot, showId) {
  return {
    name: "pipeline-api",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = new URL(req.url || "/", "http://127.0.0.1");
        if (url.pathname === "/api/catalog") {
          return sendJson(res, buildCatalog(pipelineRoot, showId || ""));
        }
        if (url.pathname.startsWith("/pipeline/")) {
          const relativePath = decodeURIComponent(url.pathname.slice("/pipeline/".length));
          const file = resolvePipelineFile(pipelineRoot, relativePath);
          if (!file || !fs.existsSync(file) || !fs.statSync(file).isFile()) {
            res.statusCode = 404;
            res.end("Not found");
            return;
          }
          res.setHeader("content-type", contentType(file));
          fs.createReadStream(file).pipe(res);
          return;
        }
        next();
      });
    },
  };
}

function fileUrl(pipelineRoot, file) {
  return "/pipeline/" + relative(pipelineRoot, file).split("/").map(encodeURIComponent).join("/");
}

function relative(pipelineRoot, file) {
  return path.relative(pipelineRoot, file).replaceAll("\\", "/");
}

function childDirs(directory) {
  if (!fs.existsSync(directory)) return [];
  return fs
    .readdirSync(directory, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => path.join(directory, entry.name));
}

function contentType(file) {
  if (file.endsWith(".json")) return "application/json";
  if (file.endsWith(".png")) return "image/png";
  if (file.endsWith(".jpg") || file.endsWith(".jpeg")) return "image/jpeg";
  if (file.endsWith(".glb")) return "model/gltf-binary";
  if (file.endsWith(".mp4")) return "video/mp4";
  return "application/octet-stream";
}

function sendJson(res, payload) {
  res.setHeader("content-type", "application/json");
  res.end(JSON.stringify(payload));
}
