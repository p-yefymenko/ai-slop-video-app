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
let catalog = { prefabs: [], sets: [], shots: [], reviews: [] };
const prefabs = new Map();

window.addEventListener("popstate", () => {
  void renderRoute();
});

const catalogResponse = await fetch("/api/catalog");
catalog = await catalogResponse.json();
for (const prefab of catalog.prefabs) prefabs.set(prefab.id, prefab);
drawSidebar();
await renderRoute();

function drawSidebar() {
  sidebar.innerHTML = "";
  sidebar.append(heading("Stage"));
  sidebar.append(link("/", "Library"));
  addGroup("Prefabs", catalog.prefabs, (prefab) => link(`/prefab/${encodePath(prefab.id)}`, prefab.id));
  addGroup("Sets", catalog.sets, (set) => link(`/set/${set.showId}/${encodePath(set.locationId)}`, `${set.showId} / ${set.locationId}`));
  addGroup(
    "Shots",
    catalog.shots,
    (shot) => link(`/shot/${shot.showId}/${shot.episodeNumber}/${shot.sceneNumber}`, `${shot.showId} ${shot.episodeNumber}.${String(shot.sceneNumber).padStart(2, "0")}`),
  );
  addGroup("Review", catalog.reviews, (review) => link(`/review/${review.showId}/${review.needHash}`, `${review.showId} ${review.needHash.slice(0, 8)}`));
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

function encodePath(value) {
  return String(value).split("/").map(encodeURIComponent).join("/");
}

async function renderRoute() {
  const route = parseRoute(location.pathname);
  markActive(location.pathname);
  toolbar.replaceChildren();
  inspector.replaceChildren();
  try {
    if (route.kind === "prefab") await showPrefab(route.id);
    else if (route.kind === "set") await showSet(route.show, route.location);
    else if (route.kind === "shot") await showShot(route.show, route.episode, route.scene);
    else if (route.kind === "review") await showReview(route.show, route.hash);
    else showHome();
  } catch (error) {
    stage.clear();
    inspector.append(note(error.message || "Could not open this stage"));
  }
}

function parseRoute(pathname) {
  const parts = pathname.split("/").filter(Boolean).map(decodeURIComponent);
  if (parts[0] === "prefab") return { kind: "prefab", id: parts.slice(1).join("/") };
  if (parts[0] === "set") return { kind: "set", show: parts[1], location: parts.slice(2).join("/") };
  if (parts[0] === "shot") return { kind: "shot", show: parts[1], episode: Number(parts[2]), scene: Number(parts[3]) };
  if (parts[0] === "review") return { kind: "review", show: parts[1], hash: parts[2] };
  return { kind: "home" };
}

function markActive(pathname) {
  for (const anchor of sidebar.querySelectorAll("a")) {
    anchor.classList.toggle("active", anchor.getAttribute("href") === pathname);
  }
}

function showHome() {
  stage.clear();
  inspector.append(heading("Y-up stage"));
  inspector.append(
    note(
      "Prefabs, sets, and shots are already glTF Y-up. This view does not convert coordinates. An editor GLB preview is optional and is not required.",
    ),
  );
}

async function showPrefab(id) {
  const prefab = prefabs.get(id);
  if (!prefab) throw new Error(`No prefab ${id}`);
  if (!prefab.modelUrl) throw new Error(`${id} has no model.glb`);
  await stage.showPrefab(prefab.modelUrl);
  fillFacts("Prefab", [
    ["Id", prefab.id],
    ["Origin", prefab.origin],
    ["License", prefab.license || "unset"],
    ["Author", prefab.author || "unset"],
    ["Source", prefab.source || "unset"],
    ["Size", formatSize(prefab.sizeMeters)],
  ]);
  if (prefab.pageUrl) inspector.append(externalLink(prefab.pageUrl));
}

async function showSet(showId, locationId) {
  const listed = catalog.sets.find((item) => item.showId === showId && item.locationId === locationId);
  if (!listed) throw new Error(`No set ${showId}/${locationId}`);
  const setDocument = await fetchJson(listed.url);
  await stage.showSet(setDocument, prefabs);
  fillFacts("Set", [
    ["Show", showId],
    ["Location", locationId],
    ["Space", setDocument.space || "gltf-y-up"],
    ["Instances", String((setDocument.instances || []).length)],
  ]);
  const list = document.createElement("ul");
  for (const instance of setDocument.instances || []) {
    const item = document.createElement("li");
    item.textContent = `${instance.id} · ${instance.origin || "unknown"} · ${instance.prefabId || ""}`;
    list.append(item);
  }
  inspector.append(list);
}

async function showShot(showId, episodeNumber, sceneNumber) {
  const listed = catalog.shots.find(
    (item) => item.showId === showId && item.episodeNumber === episodeNumber && item.sceneNumber === sceneNumber,
  );
  if (!listed) throw new Error(`No shot ${showId} ${episodeNumber}/${sceneNumber}`);
  const shot = await fetchJson(listed.url);
  const setDocument = shot.set ? await fetchJson(`/pipeline/output/${showId}/sets/${shot.locationId}/set.json`) : null;
  await stage.showShot(shot, setDocument, prefabs);
  const [start, finish] = shot.timeRangeSeconds || [0, 0];
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
  lens.textContent = "Shot camera";
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
  fillFacts("Shot", [
    ["Show", showId],
    ["Scene", String(sceneNumber)],
    ["Space", shot.space],
    ["Range", `${start}s – ${finish}s`],
    ["Camera", "God view shows the shot frustum"],
  ]);
}

async function showReview(showId, needHash) {
  const listed = catalog.reviews.find((item) => item.showId === showId && item.needHash === needHash);
  if (!listed) throw new Error("No review packet. Run pnpm content:assets -- --review");
  const packet = await fetchJson(listed.url);
  stage.clear();
  fillFacts("Review", [
    ["Show", showId],
    ["Need", packet.need?.query || needHash],
    ["Chosen", packet.chosen || "unset"],
  ]);
  const gallery = document.createElement("div");
  gallery.className = "gallery";
  for (const candidate of packet.candidates || []) {
    const card = document.createElement("article");
    const prefab = prefabs.get(candidate.prefabId);
    if (prefab?.thumbUrl) {
      const image = document.createElement("img");
      image.alt = candidate.title || candidate.prefabId;
      image.src = prefab.thumbUrl;
      card.append(image);
    }
    const title = document.createElement("strong");
    title.textContent = candidate.title || candidate.prefabId;
    const meta = document.createElement("p");
    meta.textContent = `${candidate.license || "unknown"} · ${candidate.source || ""}`;
    const open = document.createElement("button");
    open.type = "button";
    open.textContent = "View";
    open.addEventListener("click", () => {
      if (prefab?.modelUrl) void stage.showPrefab(prefab.modelUrl);
    });
    const pick = document.createElement("button");
    pick.type = "button";
    pick.textContent = "Lock this";
    pick.addEventListener("click", async () => {
      const response = await fetch("/api/pick", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          needHash,
          prefabId: candidate.prefabId,
          source: candidate.source,
          sourceId: candidate.prefabId,
        }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not update the lock");
      pick.textContent = "Locked";
    });
    card.append(title, meta, open, pick);
    gallery.append(card);
  }
  if (!(packet.candidates || []).length) gallery.append(note("This packet has no candidates."));
  inspector.append(gallery);
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

function externalLink(href) {
  const node = document.createElement("a");
  node.href = href;
  node.target = "_blank";
  node.rel = "noreferrer";
  node.textContent = "Source page";
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
