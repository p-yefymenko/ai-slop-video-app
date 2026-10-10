"""Check what the models made against the 3D scene, the same reality the script is checked against.

Previs checks the script before anything is drawn. This checks every still and
clip after it is made, so a model's own mistake is caught on the first run
instead of being found by eye:

- nobody is in the picture whom the 3D scene does not put there;
- each face in the picture is the character's face (matched against its plate).

A check returns problems as sentences; none means the output matches the world.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline_paths import ROOT

QUALITY_MODELS = ROOT / ".comfyui" / "models" / "quality"
FACE_DETECTOR = QUALITY_MODELS / "face_detection_yunet_2023mar.onnx"
FACE_MATCHER = QUALITY_MODELS / "face_recognition_sface_2021dec.onnx"
# A detection this sure is a person.
PERSON_SCORE = 0.7
# A detected person mostly outside where the 3D scene has people was invented.
PERSON_INSIDE_SHARE = 0.5
# Smaller detections are too few pixels to call.
PERSON_MIN_PIXELS = 1500
# A face smaller than this many pixels high cannot be matched.
FACE_MIN_HEIGHT = 48
# Cosine similarity below this is another person (SFace's own threshold is 0.363).
FACE_SAME_PERSON = 0.30


@cache
def _person_model():
    import torch
    import torchvision

    weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=weights).eval()
    return model, torch


def people(image: Image.Image) -> list[np.ndarray]:
    """Boolean masks of every person a detector finds in the picture."""
    model, torch = _person_model()
    tensor = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)
    with torch.no_grad():
        found = model([tensor])[0]
    masks = []
    for label, score, mask in zip(found["labels"], found["scores"], found["masks"]):
        if int(label) == 1 and float(score) >= PERSON_SCORE:
            solid = mask[0].numpy() > 0.5
            if solid.sum() >= PERSON_MIN_PIXELS:
                masks.append(solid)
    return _distinct(masks)


def _distinct(masks: list[np.ndarray]) -> list[np.ndarray]:
    """One mask per person: a detection mostly inside a bigger one is a part of that person found again."""
    kept: list[np.ndarray] = []
    for mask in sorted(masks, key=lambda item: -int(item.sum())):
        if all(float((mask & other).sum()) / float(mask.sum()) < PERSON_INSIDE_SHARE for other in kept):
            kept.append(mask)
    return kept


def invented_people(
    image: Image.Image, expected: np.ndarray, count: int | None = None, light: np.ndarray | None = None
) -> list[str]:
    """People in the picture the 3D scene does not have.

    ``expected`` is where the scene has people (True), at the picture's size.
    ``count`` caps how many people there may be at all, for a clip whose people
    move freely. ``light`` is where the shot has effects (fire, glow): a
    detection mostly inside it is not judged, since a detector sees figures in
    flames, and fire is only light, so nobody can be invented there.
    """
    found = people(image)
    if light is not None:
        found = [mask for mask in found if float((mask & light).sum()) / float(mask.sum()) < PERSON_INSIDE_SHARE]
    problems = []
    for mask in found:
        inside = float((mask & expected).sum()) / float(mask.sum())
        if inside < PERSON_INSIDE_SHARE:
            ys, xs = np.nonzero(mask)
            problems.append(
                f"a person at x={int(xs.mean())}, y={int(ys.mean())} ({int(mask.sum())} px) stands where the "
                f"3D scene has nobody ({inside:.0%} inside its people)"
            )
    if count is not None and len(found) > count:
        problems.append(f"{len(found)} people in the picture, but the 3D scene has at most {count}")
    return problems


@cache
def _face_models():
    import cv2

    detector = cv2.FaceDetectorYN.create(str(FACE_DETECTOR), "", (320, 320), 0.8)
    matcher = cv2.FaceRecognizerSF.create(str(FACE_MATCHER), "")
    return cv2, detector, matcher


def face_embedding(image: Image.Image, region: np.ndarray | None = None) -> np.ndarray | None:
    """The largest face in the picture (inside ``region`` when given), as an identity vector."""
    cv2, detector, matcher = _face_models()
    bgr = np.ascontiguousarray(np.asarray(image.convert("RGB"))[..., ::-1])
    detector.setInputSize((bgr.shape[1], bgr.shape[0]))
    _ok, faces = detector.detect(bgr)
    if faces is None:
        return None
    best = None
    for face in faces:
        x, y, w, h = (int(round(v)) for v in face[:4])
        if h < FACE_MIN_HEIGHT:
            continue
        if region is not None:
            cx, cy = min(max(x + w // 2, 0), bgr.shape[1] - 1), min(max(y + h // 2, 0), bgr.shape[0] - 1)
            if not region[cy, cx]:
                continue
        if best is None or w * h > best[2] * best[3]:
            best = face
    if best is None:
        return None
    return matcher.feature(matcher.alignCrop(bgr, best)).ravel()


@cache
def plate_face(plate: Path) -> np.ndarray | None:
    """The character's face as its plate shows it: the identity every picture of them must keep."""
    return face_embedding(Image.open(plate))


def face_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def wrong_face(image: Image.Image, region: np.ndarray, plate: Path, who: str) -> list[str]:
    """Whether the face in ``region`` of the picture is the face on the plate. Silent when no face can be measured."""
    reference = plate_face(plate)
    drawn = face_embedding(image, region)
    if reference is None or drawn is None:
        return []
    similarity = face_similarity(reference, drawn)
    if similarity < FACE_SAME_PERSON:
        return [f"{who}'s face does not match their plate (similarity {similarity:.2f}, needs {FACE_SAME_PERSON})"]
    return []


def _mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L").resize(size, Image.Resampling.NEAREST)) > 127


def check_still(still: Path, people: Path, faces: dict[str, tuple[Path, Path]]) -> list[str]:
    """A still against its start frame: nobody where the scene has nobody, and every face its character's.

    ``people`` is previs's mask of where the start frame has people. ``faces`` is
    characterId -> (where the shot shows them, their plate). A face must match its
    own plate better than any other character's, so a drawn swap is caught too.
    """
    image = Image.open(still).convert("RGB")
    problems = invented_people(image, _mask(people, image.size))
    plates = {cid: plate for cid, (_shown, plate) in faces.items()}
    for cid, (shown, plate) in faces.items():
        problems += wrong_face(image, _mask(shown, image.size), plate, cid)
        drawn = face_embedding(image, _mask(shown, image.size))
        reference = plate_face(plate)
        if drawn is None or reference is None:
            continue
        own = face_similarity(drawn, reference)
        for other, other_plate in plates.items():
            face = plate_face(other_plate)
            if other != cid and face is not None and face_similarity(drawn, face) > own:
                problems.append(f"{cid}'s face looks more like {other}'s plate than their own")
    return problems


def clip_frames(clip: Path, fps: float = 3.0) -> list[Image.Image]:
    """Frames of a clip, sampled ``fps`` times a second."""
    import subprocess

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(clip)],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(",")
    width, height = int(probe[0]), int(probe[1])
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(clip), "-vf", f"fps={fps}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3)
    return [Image.fromarray(frame) for frame in frames]


def check_clip(clip: Path, people: Image.Image, count: int, light: Image.Image | None = None) -> list[str]:
    """A clip against its shot: in every sampled frame, nobody where the shot has nobody, and no more people than it has.

    ``people`` is where the shot has people at any moment (grown for acting room),
    ``light`` where it has effects, both already fitted to the clip's frame.
    """
    problems = []
    for index, frame in enumerate(clip_frames(clip)):
        expected = np.asarray(people.convert("L").resize(frame.size, Image.Resampling.NEAREST)) > 127
        lit = None if light is None else np.asarray(light.convert("L").resize(frame.size, Image.Resampling.NEAREST)) > 127
        found = invented_people(frame, expected, count, lit)
        if found:
            problems.append(f"at {index / 3.0:.1f}s: " + "; ".join(found))
    return problems
