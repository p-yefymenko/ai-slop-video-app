import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const CLAY = 0xb4b6b8;
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

  async showPrefab(url) {
    this.clear();
    const object = await this._load(url);
    this.root.add(object);
    const box = new THREE.Box3().setFromObject(object);
    this.root.add(new THREE.Box3Helper(box, 0xf0d8a8));
    this._frameView(box);
  }

  async showSet(setDocument, prefabs) {
    this.clear();
    const group = new THREE.Group();
    for (const instance of setDocument.instances || []) {
      const prefab = prefabs.get(instance.prefabId);
      const object = await this._instanceObject(instance, prefab);
      const position = instance.position || [0, 0, 0];
      object.position.set(position[0], position[1], position[2]);
      group.add(object);
      const height = (instance.sizeMeters || [1, 1, 1])[2];
      group.add(this._label(instance.id, instance.origin, position[0], position[1] + height + 0.2, position[2]));
    }
    this.root.add(group);
    this._frameView(new THREE.Box3().setFromObject(group));
  }

  async showShot(shot, setDocument, prefabs) {
    this.clear();
    this.shot = shot;
    if (setDocument) await this.showSet(setDocument, prefabs);
    this.shot = shot;
    this.characters = new Map();
    for (const character of shot.characters || []) {
      const height = Number(character.proxy?.heightMeters) || 1.72;
      const build = character.proxy?.build || "average";
      const radius = 0.22 * ({ slim: 0.85, average: 1, broad: 1.15 }[build] || 1);
      const mesh = new THREE.Mesh(
        new THREE.CapsuleGeometry(radius, Math.max(height - radius * 2, 0.2), 4, 8),
        new THREE.MeshStandardMaterial({ color: 0xd9c7b0, roughness: 0.7 }),
      );
      mesh.position.y = height / 2;
      const holder = new THREE.Group();
      holder.add(mesh);
      holder.add(this._label(character.id, "show", 0, height + 0.15, 0));
      this.root.add(holder);
      this.characters.set(character.id, holder);
    }
    this.props = [];
    for (const prop of shot.props || []) {
      const prefab = prefabs.get(prop.prefabId);
      let object;
      if (prefab?.modelUrl) object = await this._load(prefab.modelUrl);
      else {
        object = new THREE.Mesh(
          new THREE.BoxGeometry(0.3, 0.3, 0.3),
          new THREE.MeshStandardMaterial({ color: 0xc4a484 }),
        );
      }
      this.root.add(object);
      this.props.push({ spec: prop, object });
    }
    this.setMode("god");
    this.setTime((shot.timeRangeSeconds || [0, 0])[0]);
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
    const camera = sampleCamera(this.shot.camera?.keyframes || [], timeSeconds);
    if (camera) {
      this.shotCamera.position.set(...camera.position);
      this.shotCamera.up.set(0, 1, 0);
      this.shotCamera.lookAt(...camera.lookAt);
      const roll = Number(camera.rollDegrees) || 0;
      if (roll) {
        const axis = new THREE.Vector3(...camera.lookAt).sub(this.shotCamera.position).normalize();
        this.shotCamera.up.applyAxisAngle(axis, THREE.MathUtils.degToRad(roll));
      }
      this.shotCamera.fov = Number(camera.verticalFovDegrees) || 40;
      this.shotCamera.updateProjectionMatrix();
      this.shotCamera.updateMatrixWorld();
    }
    for (const [id, holder] of this.characters || []) {
      const character = (this.shot.characters || []).find((item) => item.id === id);
      const frame = sampleTrack(character?.keyframes || [], timeSeconds);
      if (!frame?.position) continue;
      holder.position.set(frame.position[0], frame.position[1], frame.position[2]);
      holder.rotation.y = Math.PI - THREE.MathUtils.degToRad(Number(frame.bodyYawDegrees) || 0);
    }
    for (const prop of this.props || []) {
      const frame = sampleTrack(prop.spec.keyframes || [], timeSeconds);
      if (frame?.position) {
        prop.object.position.set(frame.position[0], frame.position[1], frame.position[2]);
        prop.object.visible = true;
      } else if (frame?.heldByCharacterId && this.characters?.has(frame.heldByCharacterId)) {
        const holder = this.characters.get(frame.heldByCharacterId);
        prop.object.position.copy(holder.position);
        prop.object.position.y += 1.1;
        prop.object.visible = true;
      } else {
        prop.object.visible = false;
      }
    }
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

  async _instanceObject(instance, prefab) {
    if (prefab?.modelUrl) return this._load(prefab.modelUrl);
    const size = instance.sizeMeters || [1, 1, 1];
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(size[0], size[2], size[1]),
      new THREE.MeshStandardMaterial({ color: colorFor(instance.origin), roughness: 0.8 }),
    );
    mesh.position.y = size[2] / 2;
    return mesh;
  }

  _load(url) {
    return new Promise((resolve, reject) => {
      this.loader.load(
        url,
        (gltf) => {
          gltf.scene.traverse((node) => {
            if (node.isMesh) {
              node.geometry.computeVertexNormals();
              node.material = new THREE.MeshStandardMaterial({
                color: CLAY,
                roughness: 0.82,
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

function sampleTrack(keyframes, timeSeconds) {
  if (!keyframes.length) return null;
  const frames = [...keyframes].sort((a, b) => a.timeSeconds - b.timeSeconds);
  if (timeSeconds <= frames[0].timeSeconds) return frames[0];
  if (timeSeconds >= frames[frames.length - 1].timeSeconds) return frames[frames.length - 1];
  for (let index = 0; index < frames.length - 1; index += 1) {
    const before = frames[index];
    const after = frames[index + 1];
    if (timeSeconds < before.timeSeconds || timeSeconds > after.timeSeconds) continue;
    const span = Math.max(after.timeSeconds - before.timeSeconds, 1e-6);
    const amount = (timeSeconds - before.timeSeconds) / span;
    if (!before.position || !after.position) return before;
    const sampled = {
      ...before,
      position: before.position.map((value, axis) => value + (after.position[axis] - value) * amount),
    };
    if (before.lookAt && after.lookAt) {
      sampled.lookAt = before.lookAt.map((value, axis) => value + (after.lookAt[axis] - value) * amount);
    }
    return sampled;
  }
  return frames[0];
}

function sampleCamera(keyframes, timeSeconds) {
  const frame = sampleTrack(keyframes, timeSeconds);
  if (!frame?.position || !frame.lookAt) return frame;
  return frame;
}
