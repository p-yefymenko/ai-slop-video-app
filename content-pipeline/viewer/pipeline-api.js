import fs from "node:fs";
import path from "node:path";

const ALLOWED = [
  /^library\/prefabs\/.+\.(glb|gltf|png|json)$/,
  /^library\/(lock|sources)\.json$/,
  /^shows\/[^/]+\/assets\/.+\.(glb|gltf|png|json)$/,
  /^output\/.+\.(glb|gltf|png|json)$/,
];

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

export function buildCatalog(pipelineRoot) {
  const prefabs = [];
  for (const file of walk(path.join(pipelineRoot, "library", "prefabs"), (name) => name === "prefab.json")) {
    prefabs.push(prefabRecord(pipelineRoot, file, "library"));
  }
  for (const showDir of childDirs(path.join(pipelineRoot, "shows"))) {
    for (const file of walk(path.join(showDir, "assets"), (name) => name === "prefab.json")) {
      prefabs.push(prefabRecord(pipelineRoot, file, "show"));
    }
  }
  for (const showDir of childDirs(path.join(pipelineRoot, "output"))) {
    for (const file of walk(path.join(showDir, "assets", "fallback"), (name) => name === "prefab.json")) {
      prefabs.push(prefabRecord(pipelineRoot, file, "fallback"));
    }
  }
  const sets = [];
  for (const showDir of childDirs(path.join(pipelineRoot, "output"))) {
    for (const file of walk(path.join(showDir, "sets"), (name) => name === "set.json")) {
      const locationId = path.basename(path.dirname(file));
      sets.push({
        showId: path.basename(showDir),
        locationId,
        path: relative(pipelineRoot, file),
        url: fileUrl(pipelineRoot, file),
      });
    }
  }
  const shots = [];
  for (const file of walk(path.join(pipelineRoot, "output"), (name) => name === "shot.json")) {
    const sceneDir = path.basename(path.dirname(file));
    const previsDir = path.basename(path.dirname(path.dirname(file)));
    if (previsDir !== "01_previs" || !/^scene_\d+$/.test(sceneDir)) continue;
    const episodeDir = path.dirname(path.dirname(path.dirname(file)));
    shots.push({
      showId: path.basename(path.dirname(episodeDir)),
      episodeNumber: Number(path.basename(episodeDir)),
      sceneNumber: Number(sceneDir.slice("scene_".length)),
      path: relative(pipelineRoot, file),
      url: fileUrl(pipelineRoot, file),
    });
  }
  const reviews = [];
  for (const file of walk(path.join(pipelineRoot, "output"), (name) => name === "candidates.json")) {
    const hash = path.basename(path.dirname(file));
    const reviewRoot = path.dirname(path.dirname(file));
    if (path.basename(reviewRoot) !== "asset-review") continue;
    reviews.push({
      showId: path.basename(path.dirname(reviewRoot)),
      needHash: hash,
      path: relative(pipelineRoot, file),
      url: fileUrl(pipelineRoot, file),
    });
  }
  shots.sort((a, b) => a.showId.localeCompare(b.showId) || a.episodeNumber - b.episodeNumber || a.sceneNumber - b.sceneNumber);
  return { prefabs, sets, shots, reviews };
}

export function pickNeed(pipelineRoot, body) {
  const needHash = String(body?.needHash || "");
  const prefabId = String(body?.prefabId || "");
  if (!/^[a-f0-9]{64}$/.test(needHash) || !prefabId || prefabId.includes("..")) {
    throw new Error("needHash and prefabId are required");
  }
  const lockPath = path.join(pipelineRoot, "library", "lock.json");
  const lock = JSON.parse(fs.readFileSync(lockPath, "utf8"));
  lock.needs = lock.needs || {};
  const entry = lock.needs[needHash] || { need: null, sizeMeters: null };
  entry.prefabId = prefabId;
  entry.origin = "override";
  entry.provisional = false;
  if (body.source) entry.source = String(body.source);
  if (body.sourceId) entry.sourceId = String(body.sourceId);
  lock.needs[needHash] = entry;
  fs.writeFileSync(lockPath, JSON.stringify(lock, null, 2) + "\n", "utf8");
  return entry;
}

export function pipelineApi(pipelineRoot) {
  return {
    name: "pipeline-api",
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        const url = new URL(req.url || "/", "http://127.0.0.1");
        if (url.pathname === "/api/catalog") {
          return sendJson(res, buildCatalog(pipelineRoot));
        }
        if (url.pathname === "/api/pick" && req.method === "POST") {
          try {
            const body = JSON.parse(await readBody(req));
            return sendJson(res, pickNeed(pipelineRoot, body));
          } catch (error) {
            res.statusCode = 400;
            return sendJson(res, { error: error.message });
          }
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

function prefabRecord(pipelineRoot, file, origin) {
  const record = JSON.parse(fs.readFileSync(file, "utf8"));
  const directory = path.dirname(file);
  const model = path.join(directory, "model.glb");
  const thumb = path.join(directory, "thumb.png");
  return {
    id: record.id || path.basename(directory),
    origin: record.origin || origin,
    title: record.title || record.id || path.basename(directory),
    license: record.license || "",
    author: record.author || "",
    source: record.source || "",
    pageUrl: record.pageUrl || "",
    sizeMeters: record.sizeMeters || null,
    modelUrl: fs.existsSync(model) ? fileUrl(pipelineRoot, model) : null,
    thumbUrl: fs.existsSync(thumb) ? fileUrl(pipelineRoot, thumb) : null,
    path: relative(pipelineRoot, file),
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

function walk(directory, predicate) {
  if (!fs.existsSync(directory)) return [];
  const found = [];
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    const full = path.join(directory, entry.name);
    if (entry.isDirectory()) found.push(...walk(full, predicate));
    else if (predicate(entry.name)) found.push(full);
  }
  return found;
}

function contentType(file) {
  if (file.endsWith(".json")) return "application/json";
  if (file.endsWith(".png")) return "image/png";
  if (file.endsWith(".glb")) return "model/gltf-binary";
  return "application/octet-stream";
}

function sendJson(res, payload) {
  res.setHeader("content-type", "application/json");
  res.end(JSON.stringify(payload));
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on("data", (chunk) => chunks.push(chunk));
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8") || "{}"));
    req.on("error", reject);
  });
}
