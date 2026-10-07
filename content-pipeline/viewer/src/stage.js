import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const CLAY = 0xb4b6b8;
const BODY_PARTS = ["hair", "face", "eyes", "neck", "torso", "arms", "hands", "legs", "feet"];
const PART_COLORS = {
  hair: [0.77, 0.24, 0.24],
  face: [0.94, 0.78, 0.47],
  eyes: [0.24, 0.47, 0.94],
  neck: [0.82, 0.47, 0.78],
  torso: [0.24, 0.71, 0.31],
  arms: [0.94, 0.63, 0.16],
  hands: [0.63, 0.31, 0.16],
  legs: [0.31, 0.63, 0.78],
  feet: [0.78, 0.78, 0.31],
};
export { BODY_PARTS, PART_COLORS };
const ORIGIN_COLOR = {
  show: "#7dcea0",
  library: "#85c1e9",
  downloaded: "#f7dc6f",
  fallback: "#e6b089",
  override: "#d2b4de",
};

export class Stage {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setClearColor(0x2a2d31, 1);
    this.scene = new THREE.Scene();
    this.godCamera = new THREE.PerspectiveCamera(40, 1, 0.05, 5000);
    this.godCamera.position.set(4, 3, 6);
    this.shotCamera = new THREE.PerspectiveCamera(40, 1, 0.05, 5000);
    this.viewCamera = this.godCamera;
    this.controls = new OrbitControls(this.godCamera, canvas);
    this.controls.enableDamping = true;
    this.loader = new GLTFLoader();
    this.root = new THREE.Group();
    this.scene.add(this.root);
    this.scene.add(new THREE.HemisphereLight(0xf4f1ea, 0x3a3530, 1.15));
    const sun = new THREE.DirectionalLight(0xfff4e0, 1.4);
    sun.position.set(8, 14, 6);
    this.scene.add(sun);
    this.grid = new THREE.GridHelper(20, 20, 0x6e737a, 0x3e4348);
    this.axes = new THREE.AxesHelper(1.5);
    this.scene.add(this.grid, this.axes);
    this.frustum = null;
    this.mode = "god";
    this.clock = 0;
    this.playing = false;
    this.shot = null;
    this.onTime = null;
    this._frame = this._frame.bind(this);
    this._frame();
    window.addEventListener("resize", () => this.resize());
  }

  resize() {
    const width = this.canvas.clientWidth || 1;
    const height = this.canvas.clientHeight || 1;
    this.renderer.setSize(width, height, false);
    for (const camera of [this.godCamera, this.shotCamera]) {
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    }
  }

  clear() {
    this.playing = false;
    this.shot = null;
    this.root.clear();
    if (this.frustum) {
      this.scene.remove(this.frustum);
      this.frustum = null;
    }
  }

  async showModel(url) {
    this.clear();
    const object = await this._load(url);
    this.root.add(object);
    const box = new THREE.Box3().setFromObject(object);
    this.root.add(new THREE.Box3Helper(box, 0xf0d8a8));
    this._frameView(box);
  }

  async showLandmarks(landmarks) {
    this.clear();
    const pieces = [];
    for (const landmark of landmarks || []) {
      if (!landmark.modelUrl || !landmark.position) continue;
      const object = await this._load(landmark.modelUrl);
      object.position.set(landmark.position[0], landmark.position[1], landmark.position[2]);
      this.root.add(object);
      pieces.push(object);
    }
    if (!pieces.length) return;
    const box = new THREE.Box3();
    for (const object of pieces) box.expandByObject(object);
    this._frameView(box);
  }

  /**
   * Replay the world previs rendered: every mesh once, posed from each sample.
   * Between samples position, yaw, and camera are interpolated.
   */
  async showScene(scene) {
    this.clear();
    this.shot = scene;
    this.entities = new Map();
    this.characters = new Map();
    for (const [id, spec] of Object.entries(scene.models || {})) {
      const holder = new THREE.Group();
      holder.add(await this._load(`/pipeline/output/${spec.model}`));
      if (spec.kind !== "landmark") {
        const height = new THREE.Box3().setFromObject(holder).max.y;
        holder.add(this._label(id, "show", 0, height + 0.15, 0));
      }
      this.root.add(holder);
      this.entities.set(id, holder);
      if (spec.kind === "character") this.characters.set(id, holder);
    }
    this.setMode("god");
    this.setTime((scene.timeRangeSeconds || [0, 0])[0]);
    this._frameView(new THREE.Box3().setFromObject(this.root));
  }

  setMode(mode) {
    this.mode = mode;
    if (mode === "shot" && this.shot) {
      this.controls.enabled = false;
      this.viewCamera = this.shotCamera;
    } else {
      this.controls.enabled = true;
      this.viewCamera = this.godCamera;
    }
    this._updateFrustum();
  }

  setTime(timeSeconds) {
    this.clock = timeSeconds;
    if (!this.shot) return;
    const { before, after, amount } = bracket(this.shot.frames || [], timeSeconds);
    if (!before) return;
    const camera = after ? mixCamera(before.camera, after.camera, amount) : before.camera;
    this.shotCamera.position.set(...camera.position);
    this.shotCamera.up.set(0, 1, 0);
    this.shotCamera.lookAt(...camera.lookAt);
    const roll = Number(camera.rollDegrees) || 0;
    if (roll) {
      const axis = new THREE.Vector3(...camera.lookAt).sub(this.shotCamera.position).normalize();
      this.shotCamera.up.applyAxisAngle(axis, THREE.MathUtils.degToRad(roll));
      this.shotCamera.lookAt(...camera.lookAt);
    }
    this.shotCamera.fov = Number(camera.verticalFovDegrees) || 40;
    this.shotCamera.updateProjectionMatrix();
    this.shotCamera.updateMatrixWorld();
    const next = new Map((after?.entities || []).map((item) => [item.id, item]));
    const present = new Set();
    for (const item of before.entities) {
      const holder = this.entities?.get(item.id);
      if (!holder) continue;
      const later = next.get(item.id);
      const position = later ? mix(item.position, later.position, amount) : item.position;
      const yaw = later ? mixAngle(item.yawDegrees, later.yawDegrees, amount) : item.yawDegrees;
      holder.position.set(position[0], position[1], position[2]);
      // Schema yaw turns +Y toward +X around Z up; in glTF that is -yaw around Y.
      holder.rotation.y = -THREE.MathUtils.degToRad(yaw);
      present.add(item.id);
    }
    for (const [id, holder] of this.entities || []) holder.visible = present.has(id);
    this._updateFrustum();
    if (this.onTime) this.onTime(timeSeconds);
  }

  play(enabled) {
    this.playing = enabled && Boolean(this.shot);
  }

  _updateFrustum() {
    if (this.frustum) {
      this.scene.remove(this.frustum);
      this.frustum = null;
    }
    if (this.mode === "god" && this.shot) {
      this.frustum = new THREE.CameraHelper(this.shotCamera);
      this.scene.add(this.frustum);
    }
  }

  _load(url) {
    return new Promise((resolve, reject) => {
      this.loader.load(
        url,
        (gltf) => {
          gltf.scene.traverse((node) => {
            if (node.isMesh) {
              node.geometry.computeVertexNormals();
              const plateColor = Boolean(node.geometry.getAttribute("color"));
              node.material = new THREE.MeshStandardMaterial({
                color: plateColor ? 0xffffff : CLAY,
                vertexColors: plateColor,
                roughness: plateColor ? 1 : 0.82,
                metalness: 0.04,
                side: THREE.DoubleSide,
              });
            }
          });
          resolve(gltf.scene);
        },
        undefined,
        reject,
      );
    });
  }

  colorParts(root = this.root) {
    if (this.characters?.size) {
      for (const holder of this.characters.values()) colorMeshByParts(holder);
      return;
    }
    colorMeshByParts(root);
  }

  _label(text, origin, x, y, z) {
    const canvas = document.createElement("canvas");
    canvas.width = 256;
    canvas.height = 64;
    const draw = canvas.getContext("2d");
    draw.fillStyle = colorFor(origin);
    draw.fillRect(0, 0, 256, 64);
    draw.fillStyle = "#1c1915";
    draw.font = "28px sans-serif";
    draw.textAlign = "center";
    draw.textBaseline = "middle";
    draw.fillText(text, 128, 32);
    const sprite = new THREE.Sprite(
      new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(canvas), depthTest: false }),
    );
    sprite.position.set(x, y, z);
    sprite.scale.set(1.4, 0.35, 1);
    return sprite;
  }

  _frameView(box) {
    if (box.isEmpty()) {
      this.controls.target.set(0, 0, 0);
      this.godCamera.position.set(4, 3, 6);
      return;
    }
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const span = Math.max(size.length(), 1);
    this.grid.scale.setScalar(Math.max(span / 10, 1));
    this.controls.target.copy(center);
    this.godCamera.position.copy(center).add(new THREE.Vector3(span * 0.7, span * 0.45, span * 0.9));
    this.godCamera.near = Math.max(span / 500, 0.01);
    this.godCamera.far = span * 20;
    this.godCamera.updateProjectionMatrix();
    this.controls.update();
  }

  _frame() {
    requestAnimationFrame(this._frame);
    if (this.playing && this.shot) {
      const [start, finish] = this.shot.timeRangeSeconds || [0, 1];
      const span = Math.max(finish - start, 0.001);
      const next = this.clock + 1 / 60;
      this.setTime(next > finish ? start : next);
    }
    this.controls.update();
    this.resize();
    this.renderer.render(this.scene, this.viewCamera);
  }
}

function colorFor(origin) {
  return ORIGIN_COLOR[origin] || "#d5d8dc";
}

function partNameForVertex(x, y, z, minY, height, maxAbsX, headCenterZ) {
  const t = (y - minY) / Math.max(height, 1e-6);
  const absX = Math.abs(x) / Math.max(maxAbsX, 1e-6);
  const front = z <= headCenterZ;
  if (t < 0.07) return "feet";
  if (t < 0.48) {
    if (t >= 0.42 && t < 0.52 && absX >= 0.5) return "hands";
    return "legs";
  }
  if (t < 0.76) return absX >= 0.55 ? "arms" : "torso";
  if (t < 0.82) return "neck";
  if (front && t >= 0.86 && t < 0.91) return "eyes";
  if (front && t < 0.92) return "face";
  return "hair";
}

function colorMeshByParts(root) {
  root.traverse((child) => {
    if (!child.isMesh || !child.geometry?.getAttribute("position")) return;
    child.geometry.computeBoundingBox();
    const box = child.geometry.boundingBox;
    if (!box) return;
    const height = box.max.y - box.min.y;
    const minY = box.min.y;
    const pos = child.geometry.attributes.position;
    let maxAbsX = 0;
    let headMinZ = Infinity;
    let headMaxZ = -Infinity;
    for (let index = 0; index < pos.count; index += 1) {
      maxAbsX = Math.max(maxAbsX, Math.abs(pos.getX(index)));
      const t = (pos.getY(index) - minY) / Math.max(height, 1e-6);
      if (t >= 0.82) {
        const z = pos.getZ(index);
        headMinZ = Math.min(headMinZ, z);
        headMaxZ = Math.max(headMaxZ, z);
      }
    }
    const headCenterZ = Number.isFinite(headMinZ) ? (headMinZ + headMaxZ) / 2 : 0;
    const colors = new Float32Array(pos.count * 3);
    for (let index = 0; index < pos.count; index += 1) {
      const name = partNameForVertex(
        pos.getX(index),
        pos.getY(index),
        pos.getZ(index),
        minY,
        height,
        maxAbsX,
        headCenterZ,
      );
      const rgb = PART_COLORS[name];
      colors[index * 3] = rgb[0];
      colors[index * 3 + 1] = rgb[1];
      colors[index * 3 + 2] = rgb[2];
    }
    child.geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    child.material = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      vertexColors: true,
      roughness: 0.7,
      metalness: 0.04,
      side: THREE.DoubleSide,
    });
  });
}

function bracket(frames, timeSeconds) {
  if (!frames.length) return { before: null, after: null, amount: 0 };
  let index = 0;
  while (index < frames.length - 1 && frames[index + 1].timeSeconds <= timeSeconds) index += 1;
  const before = frames[index];
  const after = frames[index + 1] || null;
  if (!after) return { before, after: null, amount: 0 };
  const span = Math.max(after.timeSeconds - before.timeSeconds, 1e-6);
  return { before, after, amount: Math.min(Math.max((timeSeconds - before.timeSeconds) / span, 0), 1) };
}

function mix(a, b, amount) {
  return a.map((value, axis) => value + (b[axis] - value) * amount);
}

function mixAngle(a, b, amount) {
  return a + ((((b - a + 180) % 360) + 360) % 360 - 180) * amount;
}

function mixCamera(a, b, amount) {
  return {
    position: mix(a.position, b.position, amount),
    lookAt: mix(a.lookAt, b.lookAt, amount),
    verticalFovDegrees: a.verticalFovDegrees + (b.verticalFovDegrees - a.verticalFovDegrees) * amount,
    rollDegrees: mixAngle(a.rollDegrees || 0, b.rollDegrees || 0, amount),
  };
}
