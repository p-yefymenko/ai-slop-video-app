from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from asset_generate import generate_asset_mesh, object_graph, pixal3d_graph, plate_prompt  # noqa: E402
from generate_batch import latent_size  # noqa: E402
from asset_resolver import (  # noqa: E402
    AssetResolver,
    collect_requests,
    export_credits,
    description_hash,
    write_plates,
)
from asset_sources import materialize_mesh  # noqa: E402
from coords import schema_to_gltf  # noqa: E402
from mesh_io import (  # noqa: E402
    DECIMATOR,
    TRIANGLE_BUDGET,
    box_mesh,
    cap_holes,
    drop_thin_side_protrusions,
    face_schema_forward,
    fit_to_size,
    primitive_mesh,
    proportions_match,
    read_schema_mesh,
    read_vertex_colors,
    standing_proportion_warning,
    write_schema_glb,
)


class CountingGenerator:
    def __init__(self, cube: Path) -> None:
        self.cube = cube
        self.calls = 0

    def __call__(
        self,
        appearance: str,
        raw_dir: Path,
        triangle_budget: int = TRIANGLE_BUDGET,
        character: bool = False,
    ) -> Path:
        del appearance, character
        self.calls += 1
        self.triangle_budget = triangle_budget
        raw_dir.mkdir(parents=True, exist_ok=True)
        dest = raw_dir / "model.glb"
        dest.write_bytes(self.cube.read_bytes())
        return dest


class UprightTests(unittest.TestCase):
    def test_a_leaning_body_is_stood_up_without_turning_it(self) -> None:
        from mesh_io import stand_upright

        rings = [(x, y, z) for z in np.linspace(0, 1.8, 40) for x, y in ((0.2, 0), (0, 0.12), (-0.2, 0), (0, -0.12))]
        body = np.array(rings, dtype=np.float64)
        nose = len(body)
        body = np.vstack([body, [[0.0, 0.2, 1.6]]])  # faces +Y
        faces = np.array([[i, i + 1, i + 4] for i in range(len(rings) - 5)] + [[nose, nose - 1, nose - 2]])
        lean = np.radians(12)
        tilt = np.array([[1, 0, 0], [0, np.cos(lean), -np.sin(lean)], [0, np.sin(lean), np.cos(lean)]])
        upright = stand_upright(body @ tilt.T, faces)
        spine = upright[38 * 4] - upright[0]
        self.assertLess(np.degrees(np.arctan2(np.hypot(spine[0], spine[1]), spine[2])), 1.0)
        self.assertGreater(upright[nose, 1] - upright[38 * 4:39 * 4, 1].mean(), 0.1, "still faces +Y")


class AssetTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        root = Path(self._temp.name)
        self.library = root / "library"
        self.shows = root / "shows"
        self.output = root / "output"
        self.cube = root / "cube.glb"
        vertices, faces = box_mesh((1.0, 1.0, 1.0))
        write_schema_glb(self.cube, vertices + np.array([5.0, -3.0, 4.0]), faces)

    def tearDown(self) -> None:
        self._temp.cleanup()

    def test_mesh_round_trip_and_primitive_bounds(self) -> None:
        vertices, faces = box_mesh((2.0, 3.0, 4.0))
        path = self.library / "roundtrip.glb"
        write_schema_glb(path, vertices, faces)
        restored, restored_faces = read_schema_mesh(path)
        np.testing.assert_allclose(restored, vertices, atol=1e-5)
        np.testing.assert_array_equal(restored_faces, faces)
        for kind, size in (("column", (1.0, 1.0, 4.0)), ("window", (1.2, 0.1, 1.6)), ("seat", (0.6, 0.6, 0.9))):
            mesh, mesh_faces = primitive_mesh(kind, size)
            self.assertGreater(len(mesh_faces), 0)
            self.assertAlmostEqual(float(mesh[:, 2].min()), 0.0, places=5)
        self.assertFalse(proportions_match(np.array([1.0, 1.0, 1.0]), (8.0, 0.2, 0.2)))
        self.assertTrue(proportions_match(np.array([2.0, 2.0, 2.0]), (1.0, 1.0, 1.0)))

    def test_glb_saved_as_zip_is_not_unzipped(self) -> None:
        mislabeled = self.library / "model.zip"
        mislabeled.parent.mkdir(parents=True, exist_ok=True)
        mislabeled.write_bytes(self.cube.read_bytes())
        mesh = materialize_mesh(mislabeled)
        self.assertEqual(mesh.suffix, ".glb")
        _vertices, faces = read_schema_mesh(mesh)
        self.assertGreater(len(faces), 0)

    def test_zip_extracts_the_first_mesh(self) -> None:
        archive_path = self.library / "pack.zip"
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.write(self.cube, "models/chair.glb")
        extracted = materialize_mesh(archive_path)
        self.assertEqual(extracted.suffix, ".glb")
        vertices, faces = read_schema_mesh(extracted)
        self.assertGreater(len(faces), 0)
        self.assertEqual(len(vertices), 8)

    def test_objects_stay_upright_and_characters_are_pixel_aligned(self) -> None:
        upright = [node["class_type"] for node in object_graph("plate.png", 7).values()]
        self.assertIn("Trellis2Conditioning", upright)
        self.assertNotIn("Pixal3DConditioning", upright)
        self.assertIn("PaintMesh", upright)
        graph = pixal3d_graph("plate.png", 7)
        classes = [node["class_type"] for node in graph.values()]
        self.assertIn("SaveGLB", classes)
        decimate_id = next(node_id for node_id, node in graph.items() if node["class_type"] == "DecimateMesh")
        self.assertEqual(graph[decimate_id]["inputs"]["target_face_count"], TRIANGLE_BUDGET)
        self.assertEqual(graph[decimate_id]["inputs"]["placement_mode"], "midpoint")
        custom = pixal3d_graph("plate.png", 7, triangle_budget=12_000)
        custom_decimate = next(node for node in custom.values() if node["class_type"] == "DecimateMesh")
        self.assertEqual(custom_decimate["inputs"]["target_face_count"], 12_000)
        self.assertEqual(latent_size(graph), (1024, 1024, 1))
        self.assertIn("a stone bench", plate_prompt("  a   stone bench "))
        landmark = plate_prompt("a stone bench")
        self.assertNotIn("movie set", landmark.lower())
        self.assertNotIn("place", landmark.lower())
        self.assertIn("a stone bench", landmark)
        person = plate_prompt("black hair", character=True)
        self.assertIn("full-body", person)
        self.assertIn("black hair", person)
        self.assertIn("no holes", person)
        self.assertNotIn("movie set", person.lower())
        aligned = pixal3d_graph("plate.png", 7)
        aligned_classes = [node["class_type"] for node in aligned.values()]
        self.assertIn("Pixal3DConditioning", aligned_classes)
        self.assertNotIn("Trellis2Conditioning", aligned_classes)
        self.assertNotIn("FillHoles", aligned_classes)
        self.assertNotIn("RemeshMesh", aligned_classes)
        aligned_unet = next(node for node in aligned.values() if node["class_type"] == "UNETLoader")
        self.assertEqual(aligned_unet["inputs"]["unet_name"], "pixal3d_int8_convrot.safetensors")
        aligned_crop = next(node for node in aligned.values() if node["class_type"] == "ImageCropToMask")
        self.assertEqual(aligned_crop["inputs"]["pad_factor"], 1.1)
        aligned_dino = next(node for node in aligned.values() if node["class_type"] == "CLIPVisionLoader")
        self.assertEqual(aligned_dino["inputs"]["clip_name"], "dino_v3_L_naf_fp32.safetensors")
        aligned_vae = next(node for node in aligned.values() if node["class_type"] == "VAELoader")
        self.assertEqual(aligned_vae["inputs"]["vae_name"], "trellis_2_shape_vae_bf16.safetensors")
        fov = next(node for node in aligned.values() if node["class_type"] == "MoGeGeometryToFOV")
        self.assertEqual(fov["inputs"]["axis"], "horizontal")
        self.assertEqual(fov["inputs"]["unit"], "degrees")
        conditioning = next(node for node in aligned.values() if node["class_type"] == "Pixal3DConditioning")
        self.assertEqual(conditioning["inputs"]["camera_angle_x"], ["32", 0])
        self.assertEqual(conditioning["inputs"]["image"], ["4", 0])
        structure = next(node for node in aligned.values() if node["class_type"] == "VaeDecodeStructureTrellis2")
        self.assertEqual(structure["inputs"]["resolution"], "32")
        texture = next(node for node in aligned.values() if node["class_type"] == "Trellis2TextureStage")
        self.assertEqual(texture["inputs"]["shape_latent"], ["20", 0])
        paint = next(node for node in aligned.values() if node["class_type"] == "PaintMesh")
        self.assertEqual(paint["inputs"]["mesh"], ["22", 0])
        aligned_save = next(node for node in aligned.values() if node["class_type"] == "SaveGLB")
        self.assertEqual(aligned_save["inputs"]["mesh"], ["37", 0])
        texture_vae = [
            node for node in aligned.values() if node["class_type"] == "VAELoader"
        ]
        self.assertEqual(
            [node["inputs"]["vae_name"] for node in texture_vae],
            ["trellis_2_shape_vae_bf16.safetensors", "trellis_2_texture_vae_bf16.safetensors"],
        )
        small, small_faces = box_mesh((0.01, 0.01, 0.01))
        opened = small_faces[:-1]
        capped, capped_faces = cap_holes(small, opened)
        np.testing.assert_array_equal(capped, small)
        self.assertEqual(len(capped_faces), len(small_faces))
        wide, wide_faces = box_mesh((1.0, 1.0, 1.0))
        _kept, kept_faces = cap_holes(wide, wide_faces[:-1])
        self.assertEqual(len(kept_faces), len(wide_faces) - 1)
        slit_points = np.array(
            [[0.0, 0.0, 0.0], [0.008, 0.0, 0.0], [0.016, 0.0, 0.0], [0.008, 0.004, 0.0]],
            dtype=np.float64,
        )
        slit_faces = np.array([[0, 1, 3], [1, 2, 3]], dtype=np.int64)
        sealed, sealed_faces = cap_holes(slit_points, slit_faces)
        np.testing.assert_array_equal(sealed, slit_points)
        self.assertGreater(len(sealed_faces), len(slit_faces))

    def test_a_thin_side_sheet_is_removed_and_a_wide_body_is_kept(self) -> None:
        def column(width: float, depth: float, height: float) -> tuple[np.ndarray, np.ndarray]:
            rings, around = 40, 8
            vertices = []
            for z in np.linspace(0.0, height, rings):
                for index in range(around):
                    angle = 2.0 * np.pi * index / around
                    vertices.append([width / 2.0 * np.cos(angle), depth / 2.0 * np.sin(angle), z])
            points = np.asarray(vertices, dtype=np.float64)
            faces = []
            for ring in range(rings - 1):
                for index in range(around):
                    a = ring * around + index
                    b = ring * around + (index + 1) % around
                    c = a + around
                    d = b + around
                    faces.append([a, b, d])
                    faces.append([a, d, c])
            return points, np.asarray(faces, dtype=np.int64)

        body, body_faces = column(0.4, 0.25, 1.8)
        sheet = []
        for x in np.linspace(-0.9, 0.9, 16):
            sheet.append([x, 0.0, 1.42])
            sheet.append([x, 0.01, 1.42])
        sheet_points = np.asarray(sheet, dtype=np.float64)
        sheet_faces = []
        for index in range(0, len(sheet_points) - 2, 2):
            sheet_faces.append([index, index + 1, index + 2])
        sheet_faces = np.asarray(sheet_faces, dtype=np.int64) + len(body)
        vertices = np.vstack((body, sheet_points))
        faces = np.vstack((body_faces, sheet_faces))
        colors = np.tile(np.array([[20, 30, 40]], dtype=np.uint8), (len(vertices), 1))
        cleaned, _faces, cleaned_colors, lost_vertices, lost_triangles = drop_thin_side_protrusions(
            vertices, faces, colors
        )
        self.assertGreater(lost_vertices, 0)
        self.assertGreater(lost_triangles, 0)
        self.assertLess(float(cleaned[:, 0].max() - cleaned[:, 0].min()), 0.6)
        self.assertEqual(len(cleaned_colors), len(cleaned))
        np.testing.assert_array_equal(cleaned_colors[0], [20, 30, 40])
        wide, wide_faces = column(0.9, 0.3, 1.8)
        _kept, kept_faces, _colors, lost_vertices, lost_triangles = drop_thin_side_protrusions(
            wide, wide_faces, None
        )
        self.assertEqual(lost_vertices, 0)
        self.assertEqual(lost_triangles, 0)
        self.assertEqual(len(kept_faces), len(wide_faces))
        fitted = fit_to_size(cleaned, (1.8, 1.8, 1.8))
        self.assertIsNone(standing_proportion_warning("sela", fitted))
        self.assertIn("vardan", standing_proportion_warning("vardan", vertices) or "")

    def test_triangle_limit_is_recorded_on_the_landmark(self) -> None:
        generator = CountingGenerator(self.cube)
        show = self._show()
        self._write_plates(show)
        resolved = self._resolver(generator, triangle_budget=12_000).resolve_show(show)
        record = json.loads((resolved["landmark:room:bench"].glb.parent / "landmark.json").read_text(encoding="utf-8"))
        self.assertEqual(record["triangleBudget"], 12_000)
        self.assertEqual(generator.triangle_budget, 12_000)
        with self.assertRaises(ValueError):
            self._resolver(generator, triangle_budget=0)

    def test_generation_is_fitted_and_a_second_resolve_does_not_generate(self) -> None:
        generator = CountingGenerator(self.cube)
        show = self._show()
        self._write_plates(show)
        item = self._resolver(generator).resolve_show(show)["landmark:room:bench"]
        self.assertEqual(item.source, "trellis2-textured")
        self.assertEqual(generator.calls, 1)
        vertices, faces = read_schema_mesh(item.glb)
        np.testing.assert_allclose(vertices.min(axis=0), [-0.25, -0.25, 0.0], atol=1e-4)
        np.testing.assert_allclose(vertices.max(axis=0) - vertices.min(axis=0), [0.5, 0.5, 0.5], atol=1e-4)
        self.assertLessEqual(len(faces), TRIANGLE_BUDGET)
        again = self._resolver(generator).resolve_show(show)
        self.assertEqual(again["landmark:room:bench"].glb, item.glb)
        self.assertEqual(generator.calls, 1)
        credits = self.output / "CREDITS.md"
        export_credits(self.output, credits)
        text = credits.read_text(encoding="utf-8")
        self.assertIn("TRELLIS.2", text)
        self.assertIn("demo/room/bench", text)

    def test_a_landmark_plate_carries_its_location_materials(self) -> None:
        from asset_resolver import collect_requests

        (request,) = [r for r in collect_requests(self._show()) if r.landmark_id]
        self.assertIn("Built of worn gray stone, like everything else in its place.", request.appearance)
        self.assertIn("Built of worn gray stone", plate_prompt(request.appearance))

    def test_plates_stop_before_the_mesh(self) -> None:
        show = self._show()
        drawn: list[Path] = []

        def writer(appearance: str, raw_dir: Path, character: bool = False) -> Path:
            del appearance, character
            raw_dir.mkdir(parents=True, exist_ok=True)
            plate = raw_dir / "plate.png"
            plate.write_bytes(b"x" * 2048)
            drawn.append(plate)
            return plate

        plates = write_plates(show, writer, output_dir=self.output)
        self.assertEqual(plates, drawn)
        self.assertEqual(len(drawn), 1)
        mesh = self.output / "assets" / "demo" / "room" / "bench" / "model.glb"
        mesh.parent.mkdir(parents=True, exist_ok=True)
        mesh.write_bytes(b"mesh")
        self.assertEqual(write_plates(show, writer, output_dir=self.output), plates)
        self.assertEqual(len(drawn), 1)
        self.assertTrue(mesh.is_file())
        write_plates(show, writer, output_dir=self.output, refresh=True)
        self.assertEqual(len(drawn), 2)
        self.assertFalse(mesh.exists())

    def test_an_entity_with_effects_gets_a_look_picture_beside_its_plate(self) -> None:
        show = self._show()
        landmark = next(iter(show["locations"]["room"]["spatial"]["landmarks"].values()))
        landmark["effects"] = [{"appearance": "white-gold flames", "size": [1, 1, 1], "offset": [0, 0, 1]}]
        looks: list[tuple[Path, list[str]]] = []

        def writer(appearance: str, raw_dir: Path, character: bool = False) -> Path:
            del appearance, character
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / "plate.png").write_bytes(b"x" * 2048)
            return raw_dir / "plate.png"

        def look_writer(plate: Path, effects: list[str], dest: Path) -> Path:
            looks.append((plate, effects))
            dest.write_bytes(b"y" * 2048)
            return dest

        (plate,) = write_plates(show, writer, output_dir=self.output, look_writer=look_writer)
        self.assertTrue(plate.with_name("look.png").is_file())
        self.assertEqual(looks, [(plate, ["white-gold flames"])])
        write_plates(show, writer, output_dir=self.output, look_writer=look_writer)
        self.assertEqual(len(looks), 1, "an unchanged plate and effects keep their look")
        del landmark["effects"]
        write_plates(show, writer, output_dir=self.output, look_writer=look_writer)
        self.assertFalse(plate.with_name("look.png").exists())

    def test_mesh_step_requires_a_reviewed_plate(self) -> None:
        with self.assertRaises(RuntimeError) as caught:
            generate_asset_mesh("a stone bench", self.output / "plates" / "demo" / "room" / "bench")
        self.assertIn("content:plates", str(caught.exception))
        with self.assertRaises(RuntimeError) as missing:
            self._resolver(CountingGenerator(self.cube)).resolve_show(self._show())
        self.assertIn("content:plates", str(missing.exception))

    def test_every_landmark_and_prop_is_its_own_mesh(self) -> None:
        show = self._show()
        show["locations"]["room"]["spatial"]["landmarks"]["column"] = {
            "position": [2, 1, 0],
            "size": [0.4, 0.4, 3.0],
            "appearance": "One marble column standing alone.",
        }
        show["props"] = {"cup": {"appearance": "One clay cup", "size": [0.1, 0.1, 0.12]}}
        requests = collect_requests(show)
        self.assertEqual([item.consumer_id for item in requests], ["landmark:room:bench", "landmark:room:column", "prop:cup"])
        self._write_plates(show)
        generator = CountingGenerator(self.cube)
        resolved = self._resolver(generator).resolve_show(show)
        self.assertEqual(generator.calls, 3)
        cup = resolved["prop:cup"]
        self.assertEqual(cup.glb, self.output / "assets" / "demo" / "props" / "cup" / "model.glb")
        record = json.loads((cup.glb.parent / "prop.json").read_text(encoding="utf-8"))
        self.assertEqual(record["propId"], "cup")
        vertices, _faces = read_schema_mesh(cup.glb)
        np.testing.assert_allclose(vertices.max(axis=0) - vertices.min(axis=0), [0.1, 0.1, 0.1], atol=1e-4)

    def test_identical_landmarks_are_generated_once(self) -> None:
        show = self._show()
        appearance = "One twisted marble column standing alone."
        show["locations"]["room"]["spatial"]["landmarks"] = {
            "gate_column_l": {"position": [-4.2, -12.4, 0], "size": [0.8, 0.8, 4.2], "appearance": appearance},
            "gate_column_r": {"position": [4.2, -12.4, 0], "size": [0.8, 0.8, 4.2], "appearance": appearance},
        }
        drawn: list[Path] = []

        def writer(text: str, raw_dir: Path, character: bool = False) -> Path:
            del text, character
            raw_dir.mkdir(parents=True, exist_ok=True)
            plate = raw_dir / "plate.png"
            plate.write_bytes(b"column" * 400)
            drawn.append(plate)
            return plate

        plates = write_plates(show, writer, output_dir=self.output)
        self.assertEqual(len(drawn), 1)
        self.assertEqual(plates[0].read_bytes(), plates[1].read_bytes())
        generator = CountingGenerator(self.cube)
        resolved = self._resolver(generator).resolve_show(show)
        self.assertEqual(generator.calls, 1)
        right = json.loads(
            (resolved["landmark:room:gate_column_r"].glb.parent / "landmark.json").read_text(encoding="utf-8")
        )
        self.assertEqual(right["schemaPosition"], [4.2, -12.4, 0.0])
        self._resolver(generator).resolve_show(show)
        self.assertEqual(generator.calls, 1)
        show["locations"]["room"]["spatial"]["landmarks"]["gate_column_r"]["size"] = [1.0, 1.0, 4.2]
        write_plates(show, writer, output_dir=self.output)
        self.assertEqual(len(drawn), 2)

    def test_fit_keeps_proportions_inside_the_requested_box(self) -> None:
        fitted = fit_to_size(box_mesh((2.0, 1.0, 4.0))[0], (8.0, 0.2, 0.2))
        extent = fitted.max(axis=0) - fitted.min(axis=0)
        np.testing.assert_allclose(extent, [0.1, 0.05, 0.2], atol=1e-6)
        self.assertAlmostEqual(float(fitted[:, 2].min()), 0.0, places=5)
        self.assertAlmostEqual(float(fitted[:, 0].min()), -float(fitted[:, 0].max()), places=5)

    def test_a_character_is_plated_and_fitted_to_their_height(self) -> None:
        show = self._show()
        show["characters"] = {
            "ada": {
                "body": "Adult woman, 24 years old",
                "attributes": ["black hair", "a soot-stained ivory wrap", "amber irises"],
                "heightMeters": 1.6,
            }
        }
        drawn: list[tuple[bool, Path]] = []

        def writer(text: str, raw_dir: Path, character: bool = False) -> Path:
            del text
            raw_dir.mkdir(parents=True, exist_ok=True)
            plate = raw_dir / "plate.png"
            plate.write_bytes(b"x" * 2048)
            drawn.append((character, plate))
            return plate

        write_plates(show, writer, output_dir=self.output)
        character_plates = [plate for character, plate in drawn if character]
        self.assertEqual([plate.parent.name for plate in character_plates], ["ada"])
        generator = CountingGenerator(self.cube)
        item = self._resolver(generator).resolve_show(show)["character:ada"]
        self.assertEqual(item.source, "pixal3d")
        vertices, _faces = read_schema_mesh(item.glb)
        np.testing.assert_allclose(vertices.max(axis=0) - vertices.min(axis=0), [1.6, 1.6, 1.6], atol=1e-3)
        self.assertTrue(item.glb.with_name("model.precleanup.glb").is_file())
        record_path = item.glb.parent / "character.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        self.assertEqual(record["characterId"], "ada")
        self.assertEqual(record["meshModel"], "pixal3d")
        self.assertEqual(record["facing"], "schema-plus-y")
        self.assertEqual(record["meshRepair"], "upright-cap-holes")
        self.assertEqual(record["surface"], "vertex-color")
        self.assertIn("amber irises", record["title"])
        np.testing.assert_allclose(face_schema_forward(np.array([[0.2, -0.5, 1.0]])), [[-0.2, 0.5, 1.0]])
        calls = generator.calls
        self._resolver(generator).resolve_show(show)
        self.assertEqual(generator.calls, calls)
        record["meshModel"] = "trellis2"
        record_path.write_text(json.dumps(record), encoding="utf-8")
        self._resolver(generator).resolve_show(show)
        self.assertEqual(generator.calls, calls + 1)
        painted_vertices, painted_faces = box_mesh((1.0, 1.0, 1.0))
        painted_colors = np.tile(np.array([[0.2, 0.4, 0.8]], dtype=np.float64), (len(painted_vertices), 1))
        painted = self.library / "painted.glb"
        write_schema_glb(painted, painted_vertices, painted_faces, painted_colors)
        item.glb.unlink()
        record_path.unlink()
        kept = read_vertex_colors(self._resolver(CountingGenerator(painted)).resolve_show(show)["character:ada"].glb)
        assert kept is not None
        np.testing.assert_allclose(kept, painted_colors)

    def test_offline_never_generates(self) -> None:
        generator = CountingGenerator(self.cube)
        show = self._show()
        self._write_plates(show)
        with self.assertRaises(RuntimeError):
            self._resolver(generator, offline=True).resolve_show(show)
        self.assertEqual(generator.calls, 0)

    def test_landmark_record_is_gltf_y_up(self) -> None:
        show = self._show()
        self._write_plates(show)
        self._resolver(CountingGenerator(self.cube)).resolve_show(show)
        record = json.loads(
            (self.output / "assets" / "demo" / "room" / "bench" / "landmark.json").read_text(encoding="utf-8")
        )
        self.assertEqual(record["space"], "gltf-y-up")
        self.assertEqual(record["schemaPosition"], [0.0, 1.0, 0.0])
        self.assertEqual(record["position"], list(schema_to_gltf((0.0, 1.0, 0.0))))

    def _resolver(self, generator: CountingGenerator, **kwargs) -> AssetResolver:
        return AssetResolver("demo", output_dir=self.output, generator=generator, **kwargs)

    def _show(self) -> dict:
        return {
            "id": "demo",
            "characters": {},
            "locations": {
                "room": {
                    "look": {"materials": "worn gray stone", "light": "soft overcast daylight"},
                    "spatial": {
                        "sizeMeters": [8.0, 10.0, 4.0],
                        "landmarks": {
                            "bench": {
                                "position": [0, 1, 0],
                                "size": [1.6, 0.6, 0.5],
                                "appearance": "One stone bench, a single object, no room.",
                            }
                        },
                    }
                }
            },
        }

    def _write_plates(self, show: dict) -> None:
        for request in collect_requests(show):
            plate = request.directory(self.output, "plates", "demo") / "plate.png"
            plate.parent.mkdir(parents=True, exist_ok=True)
            plate.write_bytes(b"x" * 2048)
            plate.with_name("plate.json").write_text(
                json.dumps({"descriptionHash": description_hash(request)}), encoding="utf-8"
            )


if __name__ == "__main__":
    unittest.main()
