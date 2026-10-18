"""What the video model gets from the 3D scene and the plates: a depth video and one reference picture.

The less the model is given, the more natural the take. Two inputs proved
necessary, and each is the least that does its job:

- **Depth video** (every shot): where things are and how big, so the framing is
  the script's. People are flat silhouettes at their own distance: the clay's
  face relief, copied, freezes the face. One fixed log scale for every shot, so a
  lone person far away is not stretched to white.
- **Reference sheet**: the reference photo of each of the shot's ``subjects``,
  side by side on white. VACE copies a reference's backdrop into the take unless
  it is white, and paints a face for anyone on the sheet, so only the people
  whose faces the shot shows are subjects (``describe.reference_people``).

A still is drawn person by person, because VACE cannot tell whose photo goes where
when the sheet holds more than one. Everyone with any pixel in the start frame is drawn
alone, upright and full height, from only their own depth, seen through a long lens from
the direction the shot sees them from (``drawing_camera``, ``person_depth``): the model is
never asked to fill a small piece of a costume, nor to draw a figure stretched by a wide
view. The shot's piece is the drawing laid on the frame by the one scale, turn and shift
that best matches where the part the shot sees falls in both (``fit_similarity``,
``shot_piece``): a whole-picture move, so nothing in the drawing is torn apart where it
does not follow its mesh. A background-removal model cuts them out of their piece, and a
person away from the focus (the first subject) is blurred as a lens would at their distance
(``blur_diameter``, ``out_of_focus``); the 3D scene only says who is in front of whom
(``layer_people``). The
final pass keeps them by their cut-outs and draws everything around them
(``write_kept_control``).
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
from PIL import Image

from ffmpeg_tools import encode_rgb_frames
from observe import backdrop_depth
from render import NEAR, camera_basis, render
from world import camera_at, entities_at

# Depth gray: 0.5 m and nearer is brightest, 60 m and farther is darkest, in log distance.
DEPTH_NEAR_METERS = 0.5
DEPTH_FAR_METERS = 60.0
# Gray range of a surface; black is reserved for nothing at all.
DEPTH_GRAY_FAR, DEPTH_GRAY_NEAR = 48, 255
# A person fills at most this share of the sheet's height.
SHEET_FILL = 0.95
# Room around a person drawn whole, as a share of their height on each side, so
# the model sees a whole figure and not one cut by the picture's edge.
WINDOW_MARGIN = 0.06
# The lens a person is drawn alone through (vertical view, degrees): a long portrait lens, as for
# a full-length photo, so the figure has its true proportions. Seen whole from the shot's own
# camera, a person a hand's width away is stretched toward the edges of the view (a leg 58°
# off-axis drawn 3.6 times as tall), which the picture model redraws with normal proportions,
# so the part the shot sees landed elsewhere in the drawing.
DRAWING_FOV_DEGREES = 24.0
# The lens a still is seen through, for its depth of field: the frame's height is a Super 35
# sensor's long side (frames are upright), and the aperture is a cinema prime's wide open, the
# shallow depth of field close shots are asked for in words. A person the shot looks past, as
# sharp as its subject, was taken for part of the subject and moved with him.
SENSOR_HEIGHT_MM = 24.89
F_NUMBER = 2.0


def depth_gray(depth: np.ndarray) -> np.ndarray:
    """Meters to gray on the fixed log scale; ``inf`` (nothing there) is black."""
    gray = np.zeros(depth.shape, dtype=np.uint8)
    valid = np.isfinite(depth)
    span = np.log(DEPTH_FAR_METERS) - np.log(DEPTH_NEAR_METERS)
    near = np.clip((np.log(DEPTH_FAR_METERS) - np.log(np.maximum(depth[valid], DEPTH_NEAR_METERS))) / span, 0.0, 1.0)
    gray[valid] = np.rint(DEPTH_GRAY_FAR + near * (DEPTH_GRAY_NEAR - DEPTH_GRAY_FAR)).astype(np.uint8)
    return gray


def flatten_people(depth: np.ndarray, entity_ids: np.ndarray, people: list[int]) -> np.ndarray:
    """Each person (entity number in ``entity_ids``) at one depth, their median: a flat silhouette."""
    flat = depth.copy()
    for number in people:
        body = (entity_ids == number) & np.isfinite(flat)
        if body.any():
            flat[body] = np.median(flat[body])
    return flat


def depth_frame(show: dict, episode: dict, scene: dict, time_seconds: float, width: int, height: int, flat_people: bool,
                people: bool = True) -> np.ndarray:
    """The shot at one moment as a gray depth picture, rendered at the video's own size; people
    flat (``flat_people``, for a clip VACE animates) or with their real relief, or left out
    (``people`` False: a still's people are drawn already, and their mesh's edge, a little off
    the drawing's, would be forced onto the pixels around them)."""
    camera = camera_at(scene, time_seconds)
    entities = [entity for entity in entities_at(show, episode, scene, time_seconds) if people or entity.kind != "character"]
    frame = render(entities, camera, width, height)
    depth = frame.depth
    if flat_people:
        depth = flatten_people(depth, frame.entity, [number + 1 for number, entity in enumerate(entities) if entity.kind == "character"])
    # Past every mesh the ground runs on to the surround: shaded at its own distance, never empty.
    backdrop = show["locations"][scene["locationId"]]
    beyond = backdrop_depth(camera, depth.shape, backdrop["spatial"]["sizeMeters"], backdrop["backdrop"]["surroundHeightMeters"])[0]
    depth = np.where(np.isfinite(depth), depth, beyond)
    return depth_gray(depth)


def write_depth_video(show: dict, episode: dict, scene: dict, width: int, height: int, length: int, fps: int, dest: Path,
                      flat_people: bool, people: bool = True, pace: float = 1.0) -> Path:
    """The shot's first ``length`` frames at ``fps`` as a depth video (``people`` False: without them).
    ``pace`` is how many seconds of the script's timeline pass in one second of the take: below 1
    when the scene was lengthened to hold its speech, so its blocking is spread over the longer take
    and never runs on into the next scene's."""
    start = float(scene["timeRangeSeconds"][0])
    frames = (depth_frame(show, episode, scene, start + index / fps * pace, width, height, flat_people, people) for index in range(length))
    raw = b"".join(np.repeat(gray[:, :, None], 3, axis=2).tobytes() for gray in frames)
    dest.parent.mkdir(parents=True, exist_ok=True)
    encode_rgb_frames(raw, width, height, fps, dest, crf=14)
    return dest


def reference_sheet(photos: list[Path], width: int, height: int, dest: Path) -> Path:
    """The reference photos side by side on pure white, as drawn, one picture: VACE takes a single reference."""
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    slot = width // len(photos)
    for index, path in enumerate(photos):
        person = Image.open(path).convert("RGB")
        scale = min(slot / person.width, height * SHEET_FILL / person.height)
        person = person.resize((max(1, int(person.width * scale)), max(1, int(person.height * scale))), Image.Resampling.LANCZOS)
        sheet.paste(person, (index * slot + (slot - person.width) // 2, (height - person.height) // 2))
    dest.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dest)
    return dest


def tinted_layout(clay: np.ndarray, entity_ids: np.ndarray, tints: dict[int, tuple[int, int, int]]) -> np.ndarray:
    """The shot's clay render with each person (by entity number) in their color, shaded as the clay
    is: which shape is whom, from the meshes, with no word for a side of the frame."""
    out = clay.astype(np.float32)
    shade = out.mean(axis=2, keepdims=True) / 255.0
    for number, rgb in tints.items():
        where = entity_ids == number
        out[where] = (shade * np.asarray(rgb, dtype=np.float32))[where]
    return np.rint(out).clip(0, 255).astype(np.uint8)


def prompt_sheet(layout: Image.Image, photos: list[tuple[Path, tuple[int, int, int]]], border: int, gap: int) -> Image.Image:
    """One picture: the layout, then each reference photo framed in its person's color, all as tall
    as the layout, on white."""
    height = layout.height
    framed = []
    for path, rgb in photos:
        photo = Image.open(path).convert("RGB")
        inner = height - 2 * border
        photo = photo.resize((max(1, round(photo.width * inner / photo.height)), inner), Image.Resampling.LANCZOS)
        frame = Image.new("RGB", (photo.width + 2 * border, height), tuple(rgb))
        frame.paste(photo, (border, border))
        framed.append(frame)
    pictures = [layout, *framed]
    sheet = Image.new("RGB", (sum(picture.width for picture in pictures) + gap * (len(pictures) + 1), height + 2 * gap), (255, 255, 255))
    x = gap
    for picture in pictures:
        sheet.paste(picture, (x, gap))
        x += picture.width + gap
    return sheet


def start_frame(show: dict, episode: dict, scene: dict, width: int, height: int):
    """The start frame at the still's size: (everyone with any pixel in it, in the script's order of
    characters; the entities; the render)."""
    start = float(scene["timeRangeSeconds"][0])
    entities = entities_at(show, episode, scene, start)
    frame = render(entities, camera_at(scene, start), width, height)
    numbers = {entity.id: number + 1 for number, entity in enumerate(entities) if entity.kind == "character"}
    seen = set(np.unique(frame.entity).tolist())
    return [cid for cid in show["characters"] if numbers.get(cid) in seen], entities, frame


def window_around(low: tuple[float, float], high: tuple[float, float], width: int, height: int) -> tuple[float, float, float, float]:
    """The window of the frame (x0, y0, x1, y1 in frame pixels) around the box ``low``-``high``,
    with ``WINDOW_MARGIN`` of its height on each side, widened to the frame's aspect."""
    pad = WINDOW_MARGIN * (high[1] - low[1])
    w, h = high[0] - low[0] + 2 * pad, high[1] - low[1] + 2 * pad
    h = max(h, w * height / width)
    w = h * width / height
    cx, cy = (low[0] + high[0]) / 2.0, (low[1] + high[1]) / 2.0
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def figure_box(person, camera: dict, width: int, height: int) -> tuple[float, float, float, float]:
    """A person's whole figure seen from ``camera`` (every vertex of their mesh in front of it,
    projected): (x0, y0, x1, y1) in frame pixels."""
    position, right, up, forward, focal = camera_basis(camera, height)
    points = person.to_world(person.mesh.vertices) - position
    ahead = points @ forward
    points, ahead = points[ahead > NEAR], ahead[ahead > NEAR]
    xs = width / 2.0 + focal * (points @ right) / ahead
    ys = height / 2.0 - focal * (points @ up) / ahead
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def window_in_render(window, frame: tuple[int, int], size: tuple[int, int]) -> tuple[float, float, float, float]:
    """A window of the frame (frame pixels) as ``render``'s crop for a picture of ``size``: the same
    camera, its frame scaled to the picture's height and centered on it."""
    scale = size[1] / frame[1]
    shift = (size[0] - frame[0] * scale) / 2.0
    x0, y0, x1, y1 = window
    return (x0 * scale + shift, y0 * scale, x1 * scale + shift, y1 * scale)


def person_depth(person, camera: dict, window, size: tuple[int, int], frame: tuple[int, int]) -> Image.Image:
    """A person alone, seen from ``camera`` through ``window`` of its ``frame`` (frame pixels), as a
    depth picture of ``size``: their own nearest point brightest and farthest darkest, on black."""
    depth = render([person], camera, size[0], size[1], crop=window_in_render(window, frame, size)).depth
    solid = np.isfinite(depth)
    gray = np.zeros(solid.shape, dtype=np.uint8)
    if solid.any():
        near, far = float(depth[solid].min()), float(depth[solid].max())
        share = (far - depth[solid]) / max(far - near, 1e-6)
        gray[solid] = np.rint(DEPTH_GRAY_FAR + share * (DEPTH_GRAY_NEAR - DEPTH_GRAY_FAR)).astype(np.uint8)
    return Image.fromarray(np.repeat(gray[..., None], 3, axis=2))


def seen_surface(entity_ids: np.ndarray, depth: np.ndarray, camera: dict, number: int) -> tuple[np.ndarray, np.ndarray]:
    """The corners of every pixel of the frame where person ``number`` is seen, each at its pixel's
    depth: (frame pixels (N, 2), world points (N, 3)). Corners, not centers, so even one pixel
    says how big and which way up the person is there."""
    height, width = entity_ids.shape
    position, right, up, forward, focal = camera_basis(camera, height)
    ys, xs = np.nonzero(entity_ids == number)
    meters = depth[ys, xs].astype(np.float64)[:, None]
    pixels, points = [], []
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        px, py = (xs + dx).astype(np.float64), (ys + dy).astype(np.float64)
        ray = right * ((px - width / 2.0) / focal)[:, None] + up * ((height / 2.0 - py) / focal)[:, None] + forward
        pixels.append(np.stack([px, py], axis=1))
        points.append(position + ray * meters)
    return np.concatenate(pixels), np.concatenate(points)


def drawing_camera(person, seen: np.ndarray, camera: dict) -> dict:
    """The camera a person is drawn alone through: it sees the part of them the shot sees (world
    points ``seen``) from the same direction the shot does, from as far as a ``DRAWING_FOV_DEGREES``
    lens needs to hold their whole height, upright, looking at the middle of their figure."""
    vertices = person.to_world(person.mesh.vertices)
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    target = seen.mean(axis=0)
    direction = np.asarray(camera["position"], dtype=np.float64) - target
    direction /= np.linalg.norm(direction)
    distance = (high[2] - low[2]) / 2.0 / np.tan(np.radians(DRAWING_FOV_DEGREES) / 2.0)
    return {"position": (target + direction * distance).tolist(), "lookAt": ((low + high) / 2.0).tolist(),
            "verticalFovDegrees": DRAWING_FOV_DEGREES}


def view_degrees(person, camera: dict) -> float:
    """How far round from a person's front ``camera`` stands, in degrees seen from above: 0 in front
    of them, 180 behind. A mesh faces its own +Y, turned by the person's yaw."""
    yaw = np.radians(person.yaw_degrees)
    facing = np.array([np.sin(yaw), np.cos(yaw)])
    toward = np.asarray(camera["position"], dtype=np.float64)[:2] - np.asarray(person.offset, dtype=np.float64)[:2]
    return float(np.degrees(np.arccos(np.clip(facing @ toward / np.linalg.norm(toward), -1.0, 1.0))))


def drawing_pixels(points: np.ndarray, camera: dict, window, size: tuple[int, int]) -> np.ndarray:
    """World ``points`` in a picture of ``size`` that shows ``window`` of ``camera``'s frame of that size."""
    position, right, up, forward, focal = camera_basis(camera, size[1])
    offset = points - position
    ahead = offset @ forward
    xs = size[0] / 2.0 + focal * (offset @ right) / ahead
    ys = size[1] / 2.0 - focal * (offset @ up) / ahead
    x0, y0, x1, y1 = window
    return np.stack([(xs - x0) * size[0] / (x1 - x0), (ys - y0) * size[1] / (y1 - y0)], axis=1)


def fit_similarity(source: np.ndarray, target: np.ndarray) -> tuple[complex, complex]:
    """The scale, turn and shift that best lay points ``source`` on ``target`` (least squares),
    as complex numbers (x + iy): target ≈ turn * source + shift."""
    a, b = source[:, 0] + 1j * source[:, 1], target[:, 0] + 1j * target[:, 1]
    a_mean, b_mean = a.mean(), b.mean()
    turn = np.vdot(a - a_mean, b - b_mean) / np.vdot(a - a_mean, a - a_mean)
    return complex(turn), complex(b_mean - turn * a_mean)


def shot_piece(drawing: Image.Image, turn: complex, shift: complex, frame: tuple[int, int], reach: int = 0) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """The drawing laid on the frame by ``turn`` and ``shift`` (``fit_similarity``): the part of the
    frame, and of ``reach`` pixels around it (what a blur pulls in from past the frame's edge),
    that it covers, at the frame's own pixels, and where that goes (left, top, right, bottom; past
    the frame by up to ``reach``). The drawing is first resized by the scale (Lanczos, so a shrink
    does not alias), then turned; past its edge its edge pixels go on, its plain backdrop."""
    scale = abs(turn)
    sized = drawing.resize((max(1, round(drawing.width * scale)), max(1, round(drawing.height * scale))), Image.Resampling.LANCZOS)
    # A point s of the sized drawing is the drawing's (s.x * W / w, s.y * H / h), which lands at
    # turn * that + shift: frame = linear @ s + offset.
    linear = np.array([[turn.real, -turn.imag], [turn.imag, turn.real]]) @ np.diag([drawing.width / sized.width, drawing.height / sized.height])
    offset = np.array([shift.real, shift.imag])
    corners = (linear @ np.array([[0, sized.width, 0, sized.width], [0, 0, sized.height, sized.height]], dtype=np.float64)).T + offset
    left, top = (max(-reach, int(np.floor(corners[:, axis].min()))) for axis in (0, 1))
    right, bottom = (min(frame[axis] + reach, int(np.ceil(corners[:, axis].max()))) for axis in (0, 1))
    # The piece's pixel p samples the sized drawing at inverse @ (p + (left, top) - offset).
    inverse = np.linalg.inv(linear)
    origin = inverse @ (np.array([left, top], dtype=np.float64) - offset)
    reach = (inverse @ (np.array([[left, right, left, right], [top, top, bottom, bottom]], dtype=np.float64) - offset[:, None])).T
    pad = int(np.ceil(max(0.0, -reach.min(), (reach - [sized.width, sized.height]).max()))) + 2
    padded = Image.fromarray(np.pad(np.asarray(sized), ((pad, pad), (pad, pad), (0, 0)), mode="edge"))
    coefficients = (inverse[0, 0], inverse[0, 1], origin[0] + pad, inverse[1, 0], inverse[1, 1], origin[1] + pad)
    return padded.transform((right - left, bottom - top), Image.Transform.AFFINE, coefficients, Image.Resampling.BICUBIC), (left, top, right, bottom)


def own_colors(picture: np.ndarray, cut: np.ndarray) -> np.ndarray:
    """A cut-out person's own colors: where the cut-out is soft (hair, fur), a drawn pixel is the
    person mixed with the plain backdrop they were drawn on, and laid in as it is, that backdrop
    rims them with its gray. The person's color is solved for at every pixel (pymatting's
    multi-level foreground estimation: the person's colors and the backdrop's each vary smoothly)."""
    from pymatting import estimate_foreground_ml

    colors = estimate_foreground_ml(picture.astype(np.float64) / 255.0, cut.astype(np.float64))
    return np.rint(np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint8)


def placed(part: np.ndarray, box: tuple[int, int, int, int], frame: tuple[int, int]) -> np.ndarray:
    """``part`` laid at ``box`` of an empty frame."""
    canvas = np.zeros((frame[1], frame[0], *part.shape[2:]), dtype=part.dtype)
    left, top, right, bottom = box
    canvas[top:bottom, left:right] = part
    return canvas


def blur_diameter(distance: float, focus: float, vertical_fov_degrees: float, height: int) -> float:
    """How wide, in frame pixels, a thin lens focused at ``focus`` meters spreads a point ``distance``
    meters away (its circle of confusion), for the frame's view and ``SENSOR_HEIGHT_MM`` and ``F_NUMBER``."""
    focal = SENSOR_HEIGHT_MM / 2.0 / np.tan(np.radians(vertical_fov_degrees) / 2.0) / 1000.0
    spread = focal / F_NUMBER * focal / (focus - focal) * abs(distance - focus) / distance
    return float(spread * 1000.0 / SENSOR_HEIGHT_MM * height)


def blur_diameters(scene: dict, people: list[str], numbers: dict[str, int], frame, camera: dict) -> dict[str, float]:
    """How out of focus each of ``people`` is in the start ``frame`` (where person ``cid`` is entity
    ``numbers[cid]``): the lens focuses on the first of the shot's subjects in the frame, and each
    person is at the median distance of their pixels. A shot with no subject in it is all in focus."""
    def distance(cid: str) -> float:
        return float(np.median(frame.depth[frame.entity == numbers[cid]]))

    subject = next((cid for cid in (scene.get("shot") or {}).get("subjects") or [] if cid in people), None)
    if subject is None:
        return {cid: 0.0 for cid in people}
    focus = distance(subject)
    return {cid: blur_diameter(distance(cid), focus, float(camera["verticalFovDegrees"]), frame.depth.shape[0]) for cid in people}


def blur_reach(diameter: float) -> int:
    """How far, in pixels, a blur of ``diameter`` pulls a pixel's neighbors in."""
    return int(np.ceil(diameter / 2.0))


def lens_blur(picture: np.ndarray, cut: np.ndarray, diameter: float) -> tuple[np.ndarray, np.ndarray]:
    """A layer (colors, cut-out 0..1) out of focus: both spread over a disc ``diameter`` pixels
    wide, the colors weighted by the cut-out, so the backdrop around the person does not bleed in."""
    from scipy.signal import fftconvolve

    radius = diameter / 2.0
    ys, xs = np.mgrid[-blur_reach(diameter):blur_reach(diameter) + 1, -blur_reach(diameter):blur_reach(diameter) + 1]
    disc = (xs * xs + ys * ys <= radius * radius).astype(np.float64)
    disc /= disc.sum()
    cut = cut.astype(np.float64)
    weight = np.clip(fftconvolve(cut, disc, mode="same"), 0.0, 1.0)
    spread = np.stack([fftconvolve(picture[..., channel] * cut, disc, mode="same") for channel in range(3)], axis=2)
    colors = np.divide(spread, weight[..., None], out=np.zeros_like(spread), where=weight[..., None] > 0)
    return np.clip(colors, 0.0, 255.0).astype(np.float32), weight.astype(np.float32)


def out_of_focus(colors: np.ndarray, cut: np.ndarray, box: tuple[int, int, int, int], frame: tuple[int, int], diameter: float) -> tuple[np.ndarray, np.ndarray]:
    """A person's piece (colors, cut-out) laid at ``box``, which may run past the frame by the blur's
    reach (the drawing goes on past the frame's edge, and the blur there pulls it in), blurred by
    ``diameter``, and cropped to the frame: (colors, cut-out)."""
    margin = max(0, -box[0], -box[1], box[2] - frame[0], box[3] - frame[1])
    canvas = (frame[0] + 2 * margin, frame[1] + 2 * margin)
    shifted = (box[0] + margin, box[1] + margin, box[2] + margin, box[3] + margin)
    blurred, weight = lens_blur(placed(colors, shifted, canvas).astype(np.float32), placed(cut, shifted, canvas), diameter)
    window = (slice(margin, margin + frame[1]), slice(margin, margin + frame[0]))
    return blurred[window], weight[window]


def hidden(entity_ids: np.ndarray, depth: np.ndarray, number: int, own: np.ndarray, floors: list[int], people: list[int]) -> np.ndarray:
    """Where a thing the final pass draws is in front of person ``number`` (whose own depth alone
    is ``own``): on their surface, whatever the full render shows instead of them; past it (hair
    or cloth a drawing adds), anything nearer than their nearest point. The floor hides nothing,
    and neither does another person: a drawing does not follow its mesh, so a nearer person's
    mesh would cut a hole that their own drawing does not fill. People cover each other only by
    their cut-outs, nearer over farther (``layer_people``)."""
    other = (entity_ids > 0) & (entity_ids != number) & ~np.isin(entity_ids, floors + people)
    solid = np.isfinite(own)
    if not solid.any():
        return other
    return other & (solid | (depth < own[solid].min()))


def layer_people(entity_ids: np.ndarray, depth: np.ndarray, people: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
                 floors: list[int]) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """The people laid over each other by their cut-outs, farthest (by their nearest point) first,
    each (number, own depth, picture, cut-out 0..1) where no thing is in front of them: (the
    people's own colors, how much of each pixel is a person, 0..1, and by number how much of each
    pixel is that person, as seen past everyone nearer). The colors are not yet
    multiplied by how much of the pixel is a person (every user lays them over something by
    ``alpha``); laid over black and then by ``alpha`` again, every soft edge got a dark rim."""
    picture = np.zeros((*entity_ids.shape, 3), dtype=np.float32)
    alpha = np.zeros(entity_ids.shape, dtype=np.float32)
    numbers = [person[0] for person in people]
    order = sorted(people, key=lambda person: -float(person[1][np.isfinite(person[1])].min()) if np.isfinite(person[1]).any() else 0.0)
    shares: dict[int, np.ndarray] = {}
    for number, own, drawing, cut in order:
        cover = cut.astype(np.float32) * ~hidden(entity_ids, depth, number, own, floors, numbers)
        picture = picture * (1.0 - cover[..., None]) + drawing.astype(np.float32) * cover[..., None]
        alpha = alpha * (1.0 - cover) + cover
        shares = {seen: share * (1.0 - cover) for seen, share in shares.items()}
        shares[number] = cover
    picture = np.divide(picture, alpha[..., None], out=np.zeros_like(picture), where=alpha[..., None] > 0)
    return np.rint(picture).clip(0, 255).astype(np.uint8), alpha, shares


def write_kept_control(depth: Path, picture: np.ndarray, alpha: np.ndarray, length: int, fps: int, control_dest: Path, mask_dest: Path) -> None:
    """The final pass's control: the depth video with the people laid over every frame by their
    cut-outs (``alpha``), and the mask that keeps them as much as their cut-outs cover each pixel
    (0 kept, 1 drawn from the depth), in every frame. Left to the depth alone, the band around a
    person was filled with a ghost of one; with the people's colors in all of it, a near-empty
    edge's meaningless color was drawn as a black outline.

    Every frame, not only the first: VACE shrinks the mask to its latent frames by sampling frames
    in time, and the first frame's own mask is never among them (a first-frame-only mask was read
    as "draw everything", and the frame was copied as it was, depth gray and all)."""
    height, width = alpha.shape
    weight = alpha[..., None]
    with av.open(str(depth)) as container:
        frames = [np.rint(picture * weight + frame.to_ndarray(format="rgb24") * (1.0 - weight)).astype(np.uint8)
                  for frame in container.decode(video=0)][:length]
    encode_rgb_frames(b"".join(frame.tobytes() for frame in frames), width, height, fps, control_dest, lossless=True)
    mask = np.repeat(np.rint((1.0 - alpha) * 255).astype(np.uint8)[..., None], 3, axis=2)
    encode_rgb_frames(mask.tobytes() * len(frames), width, height, fps, mask_dest, lossless=True)
