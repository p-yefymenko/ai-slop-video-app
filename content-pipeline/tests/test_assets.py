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

from asset_generate import generate_asset_mesh, plate_prompt, trellis_graph  # noqa: E402
from generate_batch import latent_size  # noqa: E402
from asset_resolver import (  # noqa: E402
    AssetResolver,
    collect_requests,
    export_credits,
    location_scene_description,
    write_location_plates,
)
from asset_sources import materialize_mesh  # noqa: E402
from coords import schema_to_gltf  # noqa: E402
from mesh_io import (  # noqa: E402
    DECIMATOR,
    TRIANGLE_BUDGET,
    box_mesh,
    fit_to_size,
    primitive_mesh,
    proportions_match,
    read_schema_mesh,
    write_schema_glb,
)


class CountingGenerator:
    def __init__(self, cube: Path) -> None:
        self.cube = cube
        self.calls = 0

    def __call__(self, appearance: str, raw_dir: Path, triangle_budget: int = TRIANGLE_BUDGET) -> Path:
        del appearance
        self.calls += 1
        self.triangle_budget = triangle_budget
        raw_dir.mkdir(parents=True, exist_ok=True)
        dest = raw_dir / "model.glb"
        dest.write_bytes(self.cube.read_bytes())
        return dest


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

    def test_trellis_graph_uses_the_int8_shape_pipeline(self) -> None:
        graph = trellis_graph("plate.png", 7)
        classes = [node["class_type"] for node in graph.values()]
        self.assertIn("Trellis2Conditioning", classes)
        self.assertIn("SaveGLB", classes)
        self.assertIn("DecimateMesh", classes)
        self.assertNotIn("Pixal3DConditioning", classes)
        decimate_id = next(node_id for node_id, node in graph.items() if node["class_type"] == "DecimateMesh")
        decimate_node = graph[decimate_id]
        self.assertEqual(decimate_node["inputs"]["target_face_count"], TRIANGLE_BUDGET)
        custom = trellis_graph("plate.png", 7, triangle_budget=12_000)
        custom_decimate = next(node for node in custom.values() if node["class_type"] == "DecimateMesh")
        self.assertEqual(custom_decimate["inputs"]["target_face_count"], 12_000)
        self.assertEqual(decimate_node["inputs"]["placement_mode"], "midpoint")
        save = next(node for node in graph.values() if node["class_type"] == "SaveGLB")
        self.assertEqual(save["inputs"]["mesh"], [decimate_id, 0])
        unet = next(node for node in graph.values() if node["class_type"] == "UNETLoader")
        self.assertEqual(unet["inputs"]["unet_name"], "trellis_2_int8_convrot.safetensors")
        crop = next(node for node in graph.values() if node["class_type"] == "ImageCropToMask")
        self.assertEqual(crop["inputs"]["pad_factor"], 1.0)
        self.assertEqual(latent_size(graph), (1024, 1024, 1))
        self.assertIn("a stone bench", plate_prompt("  a   stone bench "))
        place = plate_prompt("a stone bench")
        landmark = plate_prompt("a stone bench", landmark=True)
        self.assertIn("movie set", place.lower())
        self.assertNotIn("movie set", landmark.lower())
        self.assertNotIn("place", landmark.lower())
        self.assertIn("a stone bench", landmark)
        person = plate_prompt("black hair", character=True)
        self.assertIn("full-body", person)
        self.assertIn("black hair", person)
        self.assertNotIn("movie set", person.lower())

    def test_triangle_limit_is_recorded_on_the_location(self) -> None:
        generator = CountingGenerator(self.cube)
        show = self._show()
        self._write_plate(show)
        resolved = self._resolver(generator, triangle_budget=12_000).resolve_show(show)
        record = json.loads((resolved["location:room"].glb.parent / "location.json").read_text(encoding="utf-8"))
        self.assertEqual(record["triangleBudget"], 12_000)
        self.assertEqual(generator.triangle_budget, 12_000)
        with self.assertRaises(ValueError):
            self._resolver(generator, triangle_budget=0)

    def test_generation_is_fitted_and_a_second_resolve_does_not_generate(self) -> None:
        generator = CountingGenerator(self.cube)
        show = self._show()
        self._write_plate(show)
        resolved = self._resolver(generator).resolve_show(show)
        item = resolved["location:room"]
        self.assertEqual(item.source, "trellis2")
        self.assertEqual(generator.calls, 1)
        vertices, faces = read_schema_mesh(item.glb)
        np.testing.assert_allclose(vertices.min(axis=0), [-2.0, -2.0, 0.0], atol=1e-4)
        np.testing.assert_allclose(vertices.max(axis=0) - vertices.min(axis=0), [4.0, 4.0, 4.0], atol=1e-4)
        self.assertLessEqual(len(faces), TRIANGLE_BUDGET)
        again = self._resolver(generator).resolve_show(show)
        self.assertEqual(again["location:room"].glb, item.glb)
        self.assertEqual(generator.calls, 1)
        credits = self.output / "CREDITS.md"
        export_credits(self.output, credits)
        text = credits.read_text(encoding="utf-8")
        self.assertIn("MIT", text)
        self.assertIn("TRELLIS.2", text)
        self.assertIn("demo/room", text)

    def test_plates_stop_before_the_mesh(self) -> None:
        show = self._show()
        drawn: list[Path] = []

        def writer(appearance: str, raw_dir: Path) -> Path:
            del appearance
            raw_dir.mkdir(parents=True, exist_ok=True)
            plate = raw_dir / "plate.png"
            plate.write_bytes(b"x" * 2048)
            drawn.append(plate)
            return plate

        plates = write_location_plates(show, writer, output_dir=self.output)
        self.assertEqual(plates, drawn)
        self.assertEqual(len(drawn), 1)
        mesh = self.output / "assets" / "demo" / "room" / "model.glb"
        mesh.parent.mkdir(parents=True, exist_ok=True)
        mesh.write_bytes(b"mesh")
        again = write_location_plates(show, writer, output_dir=self.output)
        self.assertEqual(again, plates)
        self.assertEqual(len(drawn), 1)
        self.assertTrue(mesh.is_file())
        write_location_plates(show, writer, output_dir=self.output, refresh=True)
        self.assertEqual(len(drawn), 2)
        self.assertFalse(mesh.exists())
        self.assertTrue(plates[0].is_file())

    def test_mesh_step_requires_a_reviewed_plate(self) -> None:
        raw_dir = self.output / "plates" / "demo" / "room"
        with self.assertRaises(RuntimeError) as caught:
            generate_asset_mesh("a stone bench", raw_dir)
        self.assertIn("content:plates", str(caught.exception))
        with self.assertRaises(RuntimeError) as missing:
            self._resolver(CountingGenerator(self.cube)).resolve_show(self._show())
        self.assertIn("content:plates", str(missing.exception))

    def test_place_description_is_the_location_text(self) -> None:
        text = location_scene_description(
            "court",
            {
                "promptBlock": "An open basalt court.",
                "spatial": {
                    "sizeMeters": [22, 30, 18],
                    "landmarks": {"sun_well": {}},
                },
            },
        )
        lowered = text.lower()
        self.assertIn("open basalt court", lowered)
        self.assertIn("bare floor", lowered)
        self.assertIn("no human figures", lowered)
        self.assertNotIn("sun well", lowered)
        self.assertNotIn("several people", lowered)

    def test_a_location_with_people_builds_one_mesh_per_landmark(self) -> None:
        show = self._show()
        show["episodes"] = [
            {
                "scenes": [{"locationId": "room", "characterIds": ["ada"]}],
                "spatialTimeline": {"characterTracks": {}},
            }
        ]
        show["locations"]["room"]["spatial"]["landmarks"] = {
            "bench": {
                "position": [0, 1, 0],
                "size": [2.0, 1.0, 0.5],
                "appearance": "One stone bench, a single object, no room.",
            },
            "column": {
                "position": [2, 1, 0],
                "size": [0.4, 0.4, 3.0],
                "appearance": "One marble column standing alone.",
            },
        }
        requests = collect_requests(show)
        self.assertEqual({item.landmark_id for item in requests}, {"bench", "column"})
        self.assertNotIn("location:room", [item.consumer_id for item in requests])
        generator = CountingGenerator(self.cube)
        for request in requests:
            plate = self.output / "plates" / "demo" / request.location_id / request.landmark_id / "plate.png"
            plate.parent.mkdir(parents=True, exist_ok=True)
            plate.write_bytes(b"x" * 2048)
            from asset_resolver import description_hash

            digest = description_hash(request)
            plate.with_name("plate.json").write_text(
                json.dumps({"descriptionHash": digest}),
                encoding="utf-8",
            )
        resolved = self._resolver(generator).resolve_show(show)
        self.assertEqual(generator.calls, 2)
        bench = resolved["landmark:room:bench"]
        vertices, _faces = read_schema_mesh(bench.glb)
        extent = vertices.max(axis=0) - vertices.min(axis=0)
        np.testing.assert_allclose(sorted(extent), sorted([0.5, 0.5, 0.5]), atol=1e-3)
        self.assertTrue((bench.glb.parent / "landmark.json").is_file())

    def test_identical_landmarks_are_generated_once(self) -> None:
        show = self._show()
        show["episodes"] = [
            {
                "scenes": [{"locationId": "room", "characterIds": ["ada"]}],
                "spatialTimeline": {"characterTracks": {}},
            }
        ]
        appearance = "One twisted marble column standing alone."
        show["locations"]["room"]["spatial"]["landmarks"] = {
            "gate_column_l": {
                "position": [-4.2, -12.4, 0],
                "size": [0.8, 0.8, 4.2],
                "appearance": appearance,
            },
            "gate_column_r": {
                "position": [4.2, -12.4, 0],
                "size": [0.8, 0.8, 4.2],
                "appearance": appearance,
            },
        }
        drawn: list[Path] = []

        def writer(text: str, raw_dir: Path, landmark: bool = False) -> Path:
            del text, landmark
            raw_dir.mkdir(parents=True, exist_ok=True)
            plate = raw_dir / "plate.png"
            plate.write_bytes(b"column" * 400)
            drawn.append(plate)
            return plate

        plates = write_location_plates(show, writer, output_dir=self.output)
        self.assertEqual(len(drawn), 1)
        self.assertEqual(len(plates), 2)
        self.assertEqual(plates[0].read_bytes(), plates[1].read_bytes())
        generator = CountingGenerator(self.cube)
        resolved = self._resolver(generator).resolve_show(show)
        self.assertEqual(generator.calls, 1)
        left = json.loads(
            (resolved["landmark:room:gate_column_l"].glb.parent / "landmark.json").read_text(encoding="utf-8")
        )
        right = json.loads(
            (resolved["landmark:room:gate_column_r"].glb.parent / "landmark.json").read_text(encoding="utf-8")
        )
        self.assertEqual(left["schemaPosition"], [-4.2, -12.4, 0.0])
        self.assertEqual(right["schemaPosition"], [4.2, -12.4, 0.0])
        self.assertEqual(generator.calls, 1)
        self._resolver(generator).resolve_show(show)
        self.assertEqual(generator.calls, 1)
        show["locations"]["room"]["spatial"]["landmarks"]["gate_column_r"]["size"] = [1.0, 1.0, 4.2]
        write_location_plates(show, writer, output_dir=self.output)
        self.assertEqual(len(drawn), 2)

    def test_one_location_is_one_mesh(self) -> None:
        generator = CountingGenerator(self.cube)
        show = self._show()
        show["locations"]["room"]["spatial"]["landmarks"]["column_r"] = {}
        self._write_plate(show)
        resolved = self._resolver(generator).resolve_show(show)
        self.assertEqual(list(resolved), ["location:room"])
        self.assertEqual(generator.calls, 1)

    def test_fit_keeps_proportions_inside_the_requested_box(self) -> None:
        fitted = fit_to_size(box_mesh((2.0, 1.0, 4.0))[0], (8.0, 0.2, 0.2))
        extent = fitted.max(axis=0) - fitted.min(axis=0)
        np.testing.assert_allclose(extent, [0.1, 0.05, 0.2], atol=1e-6)
        self.assertAlmostEqual(float(fitted[:, 2].min()), 0.0, places=5)
        self.assertAlmostEqual(float(fitted[:, 0].min()), -float(fitted[:, 0].max()), places=5)

    def test_a_generated_mesh_keeps_its_shape_inside_the_location(self) -> None:
        generator = CountingGenerator(self.cube)
        show = self._show(location_size=(8.0, 0.2, 0.2))
        self._write_plate(show)
        resolved = self._resolver(generator).resolve_show(show)
        vertices, _faces = read_schema_mesh(resolved["location:room"].glb)
        np.testing.assert_allclose(vertices.max(axis=0) - vertices.min(axis=0), [0.2, 0.2, 0.2], atol=1e-3)

    def test_a_character_is_plated_and_fitted_like_a_landmark(self) -> None:
        show = self._show()
        show["characters"] = {
            "ada": {
                "promptBlock": "Adult woman, black hair.",
                "proxy": {"heightMeters": 1.6, "build": "slim"},
            }
        }
        drawn: list[tuple[bool, Path]] = []

        def writer(text: str, raw_dir: Path, landmark: bool = False, character: bool = False) -> Path:
            del text, landmark
            raw_dir.mkdir(parents=True, exist_ok=True)
            plate = raw_dir / "plate.png"
            plate.write_bytes(b"x" * 2048)
            drawn.append((character, plate))
            return plate

        write_location_plates(show, writer, output_dir=self.output)
        character_plates = [plate for character, plate in drawn if character]
        self.assertEqual(len(character_plates), 1)
        self.assertEqual(character_plates[0].parent.name, "ada")
        resolved = self._resolver(CountingGenerator(self.cube)).resolve_show(show)
        item = resolved["character:ada"]
        vertices, _faces = read_schema_mesh(item.glb)
        extent = vertices.max(axis=0) - vertices.min(axis=0)
        np.testing.assert_allclose(extent, [1.6, 1.6, 1.6], atol=1e-3)
        record = json.loads((item.glb.parent / "character.json").read_text(encoding="utf-8"))
        self.assertEqual(record["characterId"], "ada")
        self.assertEqual(record["fit"], "uniform")

    def test_offline_never_generates(self) -> None:
        generator = CountingGenerator(self.cube)
        show = self._show()
        self._write_plate(show)
        with self.assertRaises(RuntimeError):
            self._resolver(generator, offline=True).resolve_show(show)
        self.assertEqual(generator.calls, 0)

    def test_location_mesh_is_gltf_y_up(self) -> None:
        show = self._show()
        self._write_plate(show)
        self._resolver(CountingGenerator(self.cube)).resolve_show(show)
        document = json.loads(
            (self.output / "assets" / "demo" / "room" / "location.json").read_text(encoding="utf-8")
        )
        self.assertEqual(document["space"], "gltf-y-up")
        self.assertEqual(document["locationId"], "room")
        self.assertEqual(document["sizeMeters"], [8.0, 10.0, 4.0])
        self.assertTrue((self.output / "assets" / "demo" / "room" / "model.glb").is_file())
        origin = schema_to_gltf((0.0, 0.0, 0.0))
        self.assertEqual(list(origin), [0.0, 0.0, 0.0])

    def _resolver(self, generator: CountingGenerator, **kwargs) -> AssetResolver:
        return AssetResolver("demo", output_dir=self.output, generator=generator, **kwargs)

    def _show(
        self,
        appearance: str = "A stone room with one bench.",
        location_size: tuple[float, float, float] = (8.0, 10.0, 4.0),
    ) -> dict:
        return {
            "id": "demo",
            "locations": {
                "room": {
                    "promptBlock": appearance,
                    "spatial": {
                        "sizeMeters": list(location_size),
                        "landmarks": {"bench": {}},
                    },
                }
            },
        }

    def _write_plate(self, show: dict) -> None:
        request = collect_requests(show)[0]
        from asset_resolver import appearance_source_id, asset_hash

        plate = self.output / "plates" / "demo" / request.location_id / "plate.png"
        plate.parent.mkdir(parents=True, exist_ok=True)
        plate.write_bytes(b"x" * 2048)
        digest = asset_hash("trellis2", appearance_source_id(request.appearance), request.size)
        plate.with_name("plate.json").write_text(
            json.dumps({"descriptionHash": digest}),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
