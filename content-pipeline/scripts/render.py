"""Draw entities on the GPU.

``render`` is one draw of every entity into four buffers at once: shaded clay,
plate color, which entity and region owns each pixel, and camera-forward depth.
``coverage`` draws one entity's regions with no depth test, so a region counts
the pixels it would cover if nothing (not even the same body) were in front.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

WIDTH = 768
HEIGHT = 1360
NEAR = 0.05
FAR = 10_000.0
# Empty viewport in the clay playblast.
VIEWPORT_GRAY = (148, 149, 152)
KIND_GRAY = {"landmark": 176, "prop": 190, "character": 208, "floor": 156}

_VERTEX = """
#version 330 core
in vec3 in_position;
in vec3 in_color;
uniform vec2 u_yaw;
uniform vec3 u_offset;
out vec3 v_position;
out vec3 v_color;
void main() {
    vec3 p = in_position;
    v_position = vec3(p.x * u_yaw.x + p.y * u_yaw.y, -p.x * u_yaw.y + p.y * u_yaw.x, p.z) + u_offset;
    v_color = in_color;
    gl_Position = vec4(v_position, 1.0);
}
"""

_GEOMETRY = """
#version 330 core
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;
uniform vec3 u_cam;
uniform vec3 u_right;
uniform vec3 u_up;
uniform vec3 u_forward;
uniform float u_focal;
uniform vec2 u_viewport;
uniform float u_near;
uniform float u_far;
// Window of the full frame to fill the target: center in NDC, then magnification.
uniform vec3 u_crop;
in vec3 v_position[];
in vec3 v_color[];
flat out vec3 g_normal;
out vec3 g_color;
void main() {
    vec3 normal = cross(v_position[1] - v_position[0], v_position[2] - v_position[0]);
    float span = u_far - u_near;
    float depth_a = (u_far + u_near) / span;
    float depth_b = -2.0 * u_far * u_near / span;
    for (int index = 0; index < 3; index++) {
        vec3 relative = v_position[index] - u_cam;
        float z = dot(relative, u_forward);
        float x = dot(relative, u_right) * 2.0 * u_focal / u_viewport.x;
        float y = dot(relative, u_up) * 2.0 * u_focal / u_viewport.y;
        gl_Position = vec4(
            (x - u_crop.x * z) * u_crop.z,
            (y - u_crop.y * z) * u_crop.z,
            depth_a * z + depth_b,
            z
        );
        g_normal = normal;
        g_color = v_color[index];
        EmitVertex();
    }
    EndPrimitive();
}
"""

_FRAGMENT = """
#version 330 core
uniform int u_gray;
uniform int u_entity;
uniform int u_region;
flat in vec3 g_normal;
in vec3 g_color;
layout(location = 0) out vec4 out_clay;
layout(location = 1) out vec4 out_albedo;
layout(location = 2) out vec4 out_ident;
void main() {
    float value = float(u_gray);
    float length_n = length(g_normal);
    if (length_n >= 1e-6) {
        vec3 normal = g_normal / length_n;
        float facing = abs(dot(normal, normalize(vec3(0.28, 0.42, 0.86))));
        value = floor(float(u_gray) * (0.62 + 0.38 * facing) + 18.0 * max(0.0, normal.z) + 0.5);
    }
    out_clay = vec4(vec3(clamp(value, 88.0, 232.0) / 255.0), 1.0);
    out_albedo = vec4(pow(clamp(g_color, 0.0, 1.0), vec3(1.0 / 2.2)), 1.0);
    out_ident = vec4(float(u_entity) / 255.0, float(u_region) / 255.0, 0.0, 1.0);
}
"""


def camera_basis(camera: dict, height: float = HEIGHT):
    """Position, right, up, forward, and focal length in pixels."""
    position = np.asarray(camera["position"], dtype=np.float64)
    forward = np.asarray(camera["lookAt"], dtype=np.float64) - position
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0]) if abs(forward[2]) <= 0.999 else np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    roll = math.radians(float(camera.get("rollDegrees") or 0.0))
    if abs(roll) > 1e-8:
        right, up = right * math.cos(roll) + up * math.sin(roll), -right * math.sin(roll) + up * math.cos(roll)
    focal = (height / 2.0) / math.tan(math.radians(float(camera["verticalFovDegrees"])) / 2.0)
    return position, right, up, forward, focal


def screen_box(entity, camera: dict, width: int = WIDTH, height: int = HEIGHT) -> tuple[float, float, float, float] | None:
    """Where the whole entity lands in the frame, in pixels, even past its edges. None if behind the lens."""
    position, right, up, forward, focal = camera_basis(camera, height)
    points = entity.to_world(entity.mesh.vertices[:: max(1, len(entity.mesh.vertices) // 200_000)]) - position
    depth = points @ forward
    ahead = depth > NEAR
    if not ahead.any():
        return None
    xs = width / 2 + focal * (points[ahead] @ right) / depth[ahead]
    ys = height / 2 - focal * (points[ahead] @ up) / depth[ahead]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


@dataclass
class Frame:
    clay: np.ndarray  # (H, W, 3) uint8, viewport gray where empty
    entity: np.ndarray  # (H, W) uint8, index into the entity list + 1, 0 where empty
    region: np.ndarray  # (H, W) uint8 region index + 1
    # Read back only when asked (full=True): plate color, and camera-forward meters (inf where empty).
    albedo: np.ndarray | None = None
    depth: np.ndarray | None = None


class _Renderer:
    def __init__(self) -> None:
        try:
            import moderngl
        except ImportError as exc:
            raise RuntimeError("Previs draws on the GPU. Run `pnpm run content:asset-deps`.") from exc
        self.gl = moderngl
        self.ctx = moderngl.create_context(standalone=True, require=330)
        self.program = self.ctx.program(vertex_shader=_VERTEX, geometry_shader=_GEOMETRY, fragment_shader=_FRAGMENT)
        self.size: tuple[int, int] | None = None
        self.meshes: dict[str, tuple] = {}

    def _target(self, width: int, height: int) -> None:
        if self.size == (width, height):
            return
        if self.size is not None:
            for item in (self.fbo, self.cover_fbo, *self.textures, self.cover, self.depth):
                item.release()
        self.size = (width, height)
        self.textures = [
            self.ctx.texture((width, height), 3),
            self.ctx.texture((width, height), 3),
            self.ctx.texture((width, height), 2),
        ]
        self.depth = self.ctx.depth_texture((width, height))
        self.fbo = self.ctx.framebuffer(color_attachments=self.textures, depth_attachment=self.depth)
        self.cover = self.ctx.texture((width, height), 1)
        self.cover_fbo = self.ctx.framebuffer(color_attachments=[self.cover])

    def _vao(self, mesh):
        cached = self.meshes.get(mesh.key)
        if cached is not None:
            return cached[0]
        colors = mesh.colors if mesh.colors is not None else np.zeros_like(mesh.vertices)
        interleaved = np.ascontiguousarray(np.concatenate((mesh.vertices, colors), axis=1), dtype=np.float32)
        vbo = self.ctx.buffer(interleaved.tobytes())
        ibo = self.ctx.buffer(mesh.faces.tobytes())
        vao = self.ctx.vertex_array(
            self.program, [(vbo, "3f 3f", "in_position", "in_color")], ibo, index_element_size=4
        )
        self.meshes[mesh.key] = (vao, vbo, ibo)
        return vao

    def _camera(self, camera: dict, width: int, height: int, crop=None) -> None:
        """``crop`` is (x0, y0, x1, y1) of the full frame, in pixels, with the frame's aspect."""
        position, right, up, forward, focal = camera_basis(camera, height)
        if crop is None:
            self.program["u_crop"].value = (0.0, 0.0, 1.0)
        else:
            x0, y0, x1, y1 = crop
            self.program["u_crop"].value = (
                ((x0 + x1) / 2 - width / 2) / (width / 2),
                (height / 2 - (y0 + y1) / 2) / (height / 2),
                width / (x1 - x0),
            )
        for name, value in (("u_cam", position), ("u_right", right), ("u_up", up), ("u_forward", forward)):
            self.program[name].value = tuple(float(v) for v in value)
        self.program["u_focal"].value = float(focal)
        self.program["u_viewport"].value = (float(width), float(height))
        self.program["u_near"].value = NEAR
        self.program["u_far"].value = FAR

    def _place(self, entity) -> None:
        yaw = math.radians(entity.yaw_degrees)
        self.program["u_yaw"].value = (math.cos(yaw), math.sin(yaw))
        self.program["u_offset"].value = tuple(float(v) for v in entity.offset)
        self.program["u_gray"].value = KIND_GRAY.get(entity.kind, 176)

    def render(self, entities, camera: dict, width: int, height: int, crop=None, full: bool = True) -> Frame:
        self._target(width, height)
        self._camera(camera, width, height, crop)
        self.fbo.use()
        self.ctx.viewport = (0, 0, width, height)
        self.ctx.enable_only(self.gl.DEPTH_TEST)
        self.ctx.depth_func = "<"
        self.fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        for index, entity in enumerate(entities):
            vao = self._vao(entity.mesh)
            self._place(entity)
            self.program["u_entity"].value = index + 1
            for region, first, count in entity.mesh.ranges:
                self.program["u_region"].value = region + 1
                vao.render(self.gl.TRIANGLES, vertices=count * 3, first=first * 3)

        def read(attachment: int, components: int) -> np.ndarray:
            raw = self.fbo.read(components=components, attachment=attachment)
            return np.flipud(np.frombuffer(raw, dtype=np.uint8).reshape(height, width, components)).copy()

        ident = read(2, 2)
        clay = read(0, 3)
        clay[ident[:, :, 0] == 0] = VIEWPORT_GRAY
        frame = Frame(clay, ident[:, :, 0], ident[:, :, 1])
        if full:
            frame.albedo = read(1, 3)
            frame.depth = _linear_depth(self.depth, width, height)
        return frame

    def coverage(self, entity, camera: dict, width: int, height: int, regions: set[int]) -> tuple[dict[int, int], int]:
        """Pixels each of these regions covers with no depth test, and pixels those regions cover together."""
        self._target(width, height)
        self._camera(camera, width, height)
        self.cover_fbo.use()
        self.ctx.viewport = (0, 0, width, height)
        self.ctx.disable(self.gl.DEPTH_TEST)
        vao = self._vao(entity.mesh)
        self._place(entity)
        self.program["u_entity"].value = 1
        union = np.zeros(width * height, dtype=bool)
        found: dict[int, int] = {}
        for region, first, count in entity.mesh.ranges:
            if region not in regions:
                continue
            self.cover_fbo.clear(0.0, 0.0, 0.0, 0.0)
            self.program["u_region"].value = region + 1
            vao.render(self.gl.TRIANGLES, vertices=count * 3, first=first * 3)
            covered = np.frombuffer(self.cover_fbo.read(components=1), dtype=np.uint8) > 0
            union |= covered
            found[region] = int(np.count_nonzero(covered))
        return found, int(np.count_nonzero(union))


def _linear_depth(texture, width: int, height: int) -> np.ndarray:
    raw = texture.read()
    count = width * height
    if len(raw) == count * 4:
        window = np.frombuffer(raw, dtype=np.float32)
    else:
        window = np.frombuffer(raw, dtype=np.uint16).astype(np.float32) / np.float32(65535.0)
    window = np.flipud(window.reshape(height, width))
    ndc = window * np.float32(2.0) - np.float32(1.0)
    depth_a = (FAR + NEAR) / (FAR - NEAR)
    depth_b = -2.0 * FAR * NEAR / (FAR - NEAR)
    depth = np.full(window.shape, np.inf, dtype=np.float32)
    valid = window < np.float32(1.0 - 1e-6)
    depth[valid] = np.float32(depth_b) / (ndc[valid] - np.float32(depth_a))
    return depth


_RENDERER: _Renderer | None = None


def _renderer() -> _Renderer:
    global _RENDERER
    if _RENDERER is None:
        _RENDERER = _Renderer()
    return _RENDERER


def render(entities, camera: dict, width: int = WIDTH, height: int = HEIGHT, crop=None, full: bool = True) -> Frame:
    """Every entity, one draw. ``crop`` magnifies a window of the frame to fill the image.

    ``full=False`` reads back only clay and ids, for frames that are only checked.
    """
    return _renderer().render(entities, camera, width, height, crop, full)


def coverage(entity, camera: dict, regions: set[int] | None = None, width: int = WIDTH, height: int = HEIGHT) -> tuple[dict[int, int], int]:
    every = {region for region, _first, _count in entity.mesh.ranges}
    return _renderer().coverage(entity, camera, width, height, every if regions is None else regions)


def thumbnail(vertices: np.ndarray, faces: np.ndarray, destination: Path, width: int = 256) -> None:
    """Clay thumbnail of one mesh."""
    from world import Entity, _region_mesh

    points = np.asarray(vertices, dtype=np.float64)
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) / 2.0
    span = max(float(np.max(high - low)), 0.1)
    camera = {
        "position": [center[0] + span, center[1] - span * 2.2, center[2] + span * 0.85],
        "lookAt": list(center),
        "verticalFovDegrees": 35.0,
    }
    mesh = _region_mesh(f"thumb:{destination}", points, np.asarray(faces), None, np.zeros(len(points), dtype=np.int16))
    frame = render([Entity("thumb", "landmark", mesh, (0.0, 0.0, 0.0), 0.0, ("whole",))], camera)
    height = max(1, round(width * HEIGHT / WIDTH))
    Image.fromarray(frame.clay).resize((width, height), Image.Resampling.LANCZOS).save(destination)
