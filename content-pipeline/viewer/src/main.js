import { Stage } from "./stage.js";

const app = document.querySelector("#app");
const stageHost = document.createElement("div");
stageHost.className = "viewport";
const canvas = document.createElement("canvas");
stageHost.append(canvas);

const sidebar = document.createElement("aside");
sidebar.className = "sidebar";
const inspector = document.createElement("aside");
inspector.className = "inspector";
const toolbar = document.createElement("div");
toolbar.className = "toolbar";
stageHost.append(toolbar);
app.append(sidebar, stageHost, inspector);

const stage = new Stage(canvas);
let catalog = { showId: null, locations: [], scenes: [] };

window.addEventListener("popstate", () => {
  void renderRoute();
});

const catalogResponse = await fetch("/api/catalog");
catalog = await catalogResponse.json();
drawSidebar();
await renderRoute();

function drawSidebar() {
  sidebar.innerHTML = "";
  sidebar.append(heading(catalog.showId || "Show"));
  sidebar.append(link("/", "Overview"));
  addGroup("Locations", catalog.locations, (location) =>
    link(`/location/${encodeURIComponent(location.locationId)}`, location.locationId),
  );
  addGroup("Scenes", catalog.scenes, (scene) =>
    link(
      `/scene/${scene.episodeNumber}/${scene.sceneNumber}`,
      `${scene.episodeNumber}.${String(scene.sceneNumber).padStart(2, "0")}`,
    ),
  );
}

function addGroup(title, items, renderItem) {
  const block = document.createElement("section");
  block.append(heading(title));
  if (!items.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "None yet";
    block.append(empty);
  } else {
    for (const item of items) block.append(renderItem(item));
  }
  sidebar.append(block);
}

function heading(text) {
  const node = document.createElement("h2");
  node.textContent = text;
  return node;
}

function link(href, label) {
  const node = document.createElement("a");
  node.href = href;
  node.textContent = label;
  node.addEventListener("click", (event) => {
    event.preventDefault();
    history.pushState({}, "", href);
    void renderRoute();
  });
  return node;
}

async function renderRoute() {
  const route = parseRoute(location.pathname);
  markActive(location.pathname);
  toolbar.replaceChildren();
  inspector.replaceChildren();
  try {
    if (route.kind === "location") await showLocation(route.location);
    else if (route.kind === "scene") await showScene(route.episode, route.scene);
    else showHome();
  } catch (error) {
    stage.clear();
    inspector.append(note(error.message || "Could not open this view"));
  }
}

function parseRoute(pathname) {
  const parts = pathname.split("/").filter(Boolean).map(decodeURIComponent);
  if (parts[0] === "location") return { kind: "location", location: parts.slice(1).join("/") };
  if (parts[0] === "scene") return { kind: "scene", episode: Number(parts[1]), scene: Number(parts[2]) };
  return { kind: "home" };
}

function markActive(pathname) {
  for (const anchor of sidebar.querySelectorAll("a")) {
    anchor.classList.toggle("active", anchor.getAttribute("href") === pathname);
  }
}

function showHome() {
  stage.clear();
  inspector.append(heading(catalog.showId || "Viewer"));
  inspector.append(
    note(
      catalog.showId
        ? "Locations and scenes are glTF Y-up. This view does not convert coordinates. A landmark marker appears only after that point has a position."
        : "Start this viewer with pnpm run view -- --show <show-id>.",
    ),
  );
}

async function showLocation(locationId) {
  const listed = catalog.locations.find((item) => item.locationId === locationId);
  if (!listed) throw new Error(`No location ${locationId}`);
  if (listed.landmarks?.length) await stage.showLandmarks(listed.landmarks);
  else if (listed.modelUrl) await stage.showModel(listed.modelUrl);
  else stage.clear();
  fillFacts("Location", [
    ["Show", catalog.showId || ""],
    ["Location", locationId],
    ["Size", formatSize(listed.sizeMeters)],
    ["Mesh", listed.landmarks?.length ? `${listed.landmarks.length} landmarks` : listed.modelUrl ? "model.glb" : "not built"],
    ["Plate", listed.plateUrl ? "plate.png" : "not drawn"],
  ]);
  if (listed.plateUrl) {
    const image = document.createElement("img");
    image.alt = `${locationId} plate`;
    image.src = listed.plateUrl;
    inspector.append(image);
  }
}

async function showScene(episodeNumber, sceneNumber) {
  const listed = catalog.scenes.find(
    (item) => item.episodeNumber === episodeNumber && item.sceneNumber === sceneNumber,
  );
  if (!listed) throw new Error(`No scene ${episodeNumber}/${sceneNumber}`);
  const scene = await fetchJson(listed.url);
  const location = catalog.locations.find((item) => item.locationId === scene.locationId);
  const modelUrl = scene.location?.model
    ? `/pipeline/output/${scene.location.model}`
    : location?.modelUrl || null;
  await stage.showScene(scene, modelUrl);
  const [start, finish] = scene.timeRangeSeconds || [0, 0];
  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = String(start);
  slider.max = String(finish);
  slider.step = "0.01";
  slider.value = String(start);
  const play = document.createElement("button");
  play.type = "button";
  play.textContent = "Play";
  play.addEventListener("click", () => {
    const playing = play.textContent === "Play";
    play.textContent = playing ? "Pause" : "Play";
    stage.play(playing);
  });
  const god = document.createElement("button");
  god.type = "button";
  god.textContent = "God camera";
  god.classList.add("active");
  const lens = document.createElement("button");
  lens.type = "button";
  lens.textContent = "Scene camera";
  for (const [button, mode] of [
    [god, "god"],
    [lens, "shot"],
  ]) {
    button.addEventListener("click", () => {
      stage.setMode(mode);
      god.classList.toggle("active", mode === "god");
      lens.classList.toggle("active", mode === "shot");
    });
  }
  slider.addEventListener("input", () => {
    stage.play(false);
    play.textContent = "Play";
    stage.setTime(Number(slider.value));
  });
  stage.onTime = (timeSeconds) => {
    slider.value = String(timeSeconds);
  };
  toolbar.append(play, god, lens, slider);
  fillFacts("Scene", [
    ["Show", catalog.showId || scene.showId || ""],
    ["Episode", String(episodeNumber)],
    ["Scene", String(sceneNumber)],
    ["Location", scene.locationId || ""],
    ["Space", scene.space || "gltf-y-up"],
    ["Range", `${start}s – ${finish}s`],
    ["Landmarks", String((scene.landmarks || []).length)],
  ]);
}

function fillFacts(title, rows) {
  inspector.append(heading(title));
  const list = document.createElement("dl");
  for (const [label, value] of rows) {
    const term = document.createElement("dt");
    term.textContent = label;
    const detail = document.createElement("dd");
    detail.textContent = value;
    list.append(term, detail);
  }
  inspector.append(list);
}

function note(text) {
  const node = document.createElement("p");
  node.textContent = text;
  return node;
}

function formatSize(size) {
  if (!Array.isArray(size)) return "unknown";
  return size.map((value) => `${value}m`).join(" × ");
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Missing ${url}`);
  return response.json();
}
