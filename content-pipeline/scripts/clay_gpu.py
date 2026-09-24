"""Clay frames on the GPU. One draw per mesh, into a color image and a depth buffer.

The gray matches ``_shade_gray`` in ``spatial_previs``: a flat face normal, a key
light, and a sky term. The normal comes from the triangle winding, so the sky
term stays on the face that points up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

FAR_CLIP = 10_000.0

_VERTEX = """
#version 330 core
in vec3 in_position;
out vec3 v_position;
void main() {
    v_position = in_position;
    gl_Position = vec4(in_position, 1.0);
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
uniform vec3 u_offset;
uniform float u_focal;
uniform vec2 u_viewport;
uniform float u_near;
uniform float u_far;

in vec3 v_position[];
flat out vec3 g_normal;

void main() {
    vec3 p0 = v_position[0] + u_offset;
    vec3 p1 = v_position[1] + u_offset;
    vec3 p2 = v_position[2] + u_offset;
    vec3 normal = cross(p1 - p0, p2 - p0);
    vec3 points[3];
    points[0] = p0;
    points[1] = p1;
    points[2] = p2;
    float span = u_far - u_near;
    float depth_a = (u_far + u_near) / span;
    float depth_b = -2.0 * u_far * u_near / span;
    float x_scale = 2.0 * u_focal / u_viewport.x;
    float y_scale = 2.0 * u_focal / u_viewport.y;
    for (int index = 0; index < 3; index++) {
        vec3 relative = points[index] - u_cam;
        float x = dot(relative, u_right);
        float y = dot(relative, u_up);
        float z = dot(relative, u_forward);
        gl_Position = vec4(x * x_scale, y * y_scale, depth_a * z + depth_b, z);
        g_normal = normal;
        EmitVertex();
    }
    EndPrimitive();
}
"""

_FRAGMENT = """
#version 330 core
uniform int u_base;
flat in vec3 g_normal;
out vec4 frag_color;

void main() {
    float length_n = length(g_normal);
    float value = float(u_base);
    if (length_n >= 1e-6) {
        vec3 normal = g_normal / length_n;
        vec3 key = normalize(vec3(0.28, 0.42, 0.86));
        float facing = abs(dot(normal, key));
        float sky = max(0.0, normal.z);
        value = float(u_base) * (0.62 + 0.38 * facing) + 18.0 * sky;
        value = floor(value + 0.5);
    }
    value = clamp(value, 88.0, 232.0);
    frag_color = vec4(vec3(value / 255.0), 1.0);
}
"""


@dataclass
class ClayBatch:
    """Indexed triangles in schema space. ``key`` keeps the GPU buffer across frames."""

    vertices: np.ndarray
    faces: np.ndarray
    base: int
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    key: tuple | None = None

    def __post_init__(self) -> None:
        self.vertices = np.ascontiguousarray(self.vertices, dtype=np.float32)
        self.faces = np.ascontiguousarray(self.faces, dtype=np.uint32)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("Clay vertices must have shape (N, 3)")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("Clay faces must have shape (M, 3)")
        self.base = int(self.base)
        self.offset = (float(self.offset[0]), float(self.offset[1]), float(self.offset[2]))


class _Renderer:
    def __init__(self) -> None:
        try:
            import moderngl
        except ImportError as exc:
            raise RuntimeError(
                "Clay previs draws on the GPU. Run `pnpm run content:asset-deps`."
            ) from exc
        self._moderngl = moderngl
        try:
            self.ctx = moderngl.create_context(standalone=True, require=330)
        except Exception as exc:
            raise RuntimeError(
                "Clay previs could not open an OpenGL context. "
                "Run `pnpm run content:asset-deps` on this GPU machine."
            ) from exc
        self.program = self.ctx.program(
            vertex_shader=_VERTEX,
            geometry_shader=_GEOMETRY,
            fragment_shader=_FRAGMENT,
        )
        self.ctx.enable_only(moderngl.DEPTH_TEST)
        self.ctx.depth_func = "<"
        self.width = 0
        self.height = 0
        self.color = None
        self.depth = None
        self.fbo = None
        self._static: dict[tuple, tuple] = {}

    def _ensure_target(self, width: int, height: int) -> None:
        if self.fbo is not None and self.width == width and self.height == height:
            return
        if self.fbo is not None:
            self.fbo.release()
            self.color.release()
            self.depth.release()
        self.width = width
        self.height = height
        self.color = self.ctx.texture((width, height), 3)
        self.depth = self.ctx.depth_texture((width, height))
        self.fbo = self.ctx.framebuffer(color_attachments=[self.color], depth_attachment=self.depth)

    def _vao(self, batch: ClayBatch):
        cached = self._static.get(batch.key) if batch.key is not None else None
        if cached is not None:
            return cached[0], None
        vbo = self.ctx.buffer(batch.vertices.tobytes())
        ibo = self.ctx.buffer(batch.faces.tobytes())
        vao = self.ctx.vertex_array(
            self.program,
            [(vbo, "3f", "in_position")],
            ibo,
            index_element_size=4,
        )
        if batch.key is None:
            return vao, (vbo, ibo)
        if isinstance(batch.key, tuple) and batch.key:
            stale = [
                existing
                for existing in self._static
                if isinstance(existing, tuple)
                and existing
                and existing[0] == batch.key[0]
                and existing != batch.key
            ]
            for existing in stale:
                old_vao, old_vbo, old_ibo = self._static.pop(existing)
                old_vao.release()
                old_vbo.release()
                old_ibo.release()
        self._static[batch.key] = (vao, vbo, ibo)
        return vao, None

    def draw(
        self,
        batches: list[ClayBatch],
        basis: tuple,
        *,
        width: int,
        height: int,
        near: float,
        background: tuple[int, int, int],
    ) -> tuple[Image.Image, np.ndarray]:
        self._ensure_target(width, height)
        assert self.fbo is not None and self.depth is not None
        position, right, up, forward, focal = basis
        self.fbo.use()
        self.ctx.viewport = (0, 0, width, height)
        self.program["u_cam"].value = position
        self.program["u_right"].value = right
        self.program["u_up"].value = up
        self.program["u_forward"].value = forward
        self.program["u_focal"].value = float(focal)
        self.program["u_viewport"].value = (float(width), float(height))
        self.program["u_near"].value = float(near)
        self.program["u_far"].value = float(FAR_CLIP)
        red, green, blue = (channel / 255.0 for channel in background)
        self.fbo.clear(red, green, blue, 1.0, depth=1.0)
        for batch in batches:
            if batch.faces.size == 0:
                continue
            vao, ephemeral = self._vao(batch)
            self.program["u_base"].value = batch.base
            self.program["u_offset"].value = batch.offset
            vao.render(self._moderngl.TRIANGLES)
            if ephemeral is not None:
                vbo, ibo = ephemeral
                vao.release()
                vbo.release()
                ibo.release()
        color = np.frombuffer(self.fbo.read(components=3), dtype=np.uint8).reshape(height, width, 3)
        color = np.ascontiguousarray(np.flipud(color))
        depth = _linear_depth(_read_window_depth(self.depth, width, height), near, FAR_CLIP)
        return Image.fromarray(color, "RGB"), depth


def _read_window_depth(depth, width: int, height: int) -> np.ndarray:
    raw = depth.read()
    count = width * height
    if len(raw) == count * 4:
        values = np.frombuffer(raw, dtype=np.float32)
    elif len(raw) == count * 2:
        values = np.frombuffer(raw, dtype=np.uint16).astype(np.float32) / np.float32(65535.0)
    else:
        raise RuntimeError(f"Unexpected depth buffer of {len(raw)} bytes for {width}x{height}")
    return np.flipud(values.reshape(height, width).copy())


def _linear_depth(window_depth: np.ndarray, near: float, far: float) -> np.ndarray:
    """Camera-forward meters. Cleared pixels stay infinite, matching the old z-buffer."""
    ndc = window_depth.astype(np.float64) * 2.0 - 1.0
    span = far - near
    depth_a = (far + near) / span
    depth_b = -2.0 * far * near / span
    depth = np.full(window_depth.shape, np.inf, dtype=np.float32)
    valid = window_depth < np.float32(1.0 - 1e-6)
    depth[valid] = (depth_b / (ndc[valid] - depth_a)).astype(np.float32)
    return depth


_RENDERER: _Renderer | None = None


def raster_clay(
    batches: list[ClayBatch],
    basis: tuple,
    *,
    width: int,
    height: int,
    near: float,
    background: tuple[int, int, int],
) -> tuple[Image.Image, np.ndarray]:
    global _RENDERER
    if _RENDERER is None:
        _RENDERER = _Renderer()
    return _RENDERER.draw(
        batches,
        basis,
        width=width,
        height=height,
        near=near,
        background=background,
    )
