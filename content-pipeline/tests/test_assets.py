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

from asset_generate import asset_plate_prompt, trellis_graph  # noqa: E402
from asset_resolver import (  # noqa: E402
    GENERATED_SOURCE,
    AssetResolver,
    appearance_source_id,
    apply_lock_override,
    asset_hash,
    export_credits,
    prune_unused_library,
)
from asset_sources import materialize_mesh  # noqa: E402
from coords import schema_to_gltf  # noqa: E402
from mesh_io import (  # noqa: E402
    TRIANGLE_BUDGET,
    box_mesh,
    decimate,
    primitive_mesh,
    proportions_match,
    read_schema_mesh,
    write_schema_glb,
)


class CountingGenerator:
    def __init__(self, cube: Path) -> None:
        self.cube = cube
        self.calls = 0

    def __call__(self, appearance: str, raw_dir: Path) -> Path:
        del appearance
        self.calls += 1
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

    def test_decimate_stays_under_the_previs_budget(self) -> None:
        xs = np.linspace(0.0, 1.0, 360)
        ys = np.linspace(0.0, 1.0, 360)
        grid_x, grid_y = np.meshgrid(xs, ys)
        vertices = np.column_stack((grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size)))
        faces = []
        width = len(xs)
        for y in range(len(ys) - 1):
            for x in range(width - 1):
                index = y * width + x
                faces.append((index, index + 1, index + width))
                faces.append((index + 1, index + width + 1, index + width))
        dense = np.array(faces, dtype=np.int64)
        self.assertGreater(len(dense), TRIANGLE_BUDGET)
        reduced_vertices, reduced_faces = decimate(vertices, dense)
        self.assertLessEqual(len(reduced_faces), TRIANGLE_BUDGET)
        self.assertGreater(len(reduced_faces), 0)
        self.assertLess(float(reduced_vertices[:, 0].min()), 0.05)
        self.assertGreater(float(reduced_vertices[:, 0].max()), 0.95)

    def test_decimate_welds_a_triangle_soup(self) -> None:
        xs = np.linspace(0.0, 1.0, 40)
        ys = np.linspace(0.0, 1.0, 40)
        grid_x, grid_y = np.meshgrid(xs, ys)
        vertices = np.column_stack((grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size)))
        faces = []
        width = len(xs)
        for y in range(len(ys) - 1):
            for x in range(width - 1):
                index = y * width + x
                faces.append((index, index + 1, index + width))
                faces.append((index + 1, index + width + 1, index + width))
        indexed = np.array(faces, dtype=np.int64)
        soup_vertices = vertices[indexed].reshape(-1, 3)
        soup_faces = np.arange(len(soup_vertices), dtype=np.int64).reshape(-1, 3)
        reduced_vertices, reduced_faces = decimate(soup_vertices, soup_faces, budget=200)
        self.assertLessEqual(len(reduced_faces), 200)
        self.assertLess(float(reduced_vertices[:, 0].min()), 0.05)
        self.assertGreater(float(reduced_vertices[:, 0].max()), 0.95)

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
        self.assertNotIn("Pixal3DConditioning", classes)
        unet = next(node for node in graph.values() if node["class_type"] == "UNETLoader")
        self.assertEqual(unet["inputs"]["unet_name"], "trellis_2_int8_convrot.safetensors")
        crop = next(node for node in graph.values() if node["class_type"] == "ImageCropToMask")
        self.assertEqual(crop["inputs"]["pad_factor"], 1.0)
        self.assertIn("a stone bench", asset_plate_prompt("  a   stone bench "))

    def test_show_prefab_beats_the_library_and_skips_generation(self) -> None:
        appearance = "a stone bench"
        size = (1.0, 1.0, 1.0)
        digest = asset_hash(GENERATED_SOURCE, appearance_source_id(appearance), size)
        self._plant(self.shows / "demo" / "assets", "blocks/show-chair", digest, "from show", "show")
        self._plant(self.library / "prefabs", "blocks/library-chair", digest, "from library", "library")
        generator = CountingGenerator(self.cube)
        resolved = self._resolver(generator).resolve_show(self._show(appearance))
        item = resolved["landmark:room/bench"]
        self.assertEqual(item.origin, "show")
        self.assertEqual(item.title, "from show")
        self.assertEqual(generator.calls, 0)

    def test_pinned_prefab_skips_generation(self) -> None:
        self._plant(self.shows / "demo" / "assets", "blocks/custom-chair", None, "pinned", "show")
        generator = CountingGenerator(self.cube)
        show = self._show("a stone bench", prefab_id="blocks/custom-chair")
        resolved = self._resolver(generator).resolve_show(show)
        self.assertEqual(resolved["landmark:room/bench"].prefab_id, "blocks/custom-chair")
        self.assertEqual(generator.calls, 0)

    def test_generation_is_fitted_and_a_second_resolve_does_not_generate(self) -> None:
        generator = CountingGenerator(self.cube)
        first = self._resolver(generator)
        show = self._show("a stone bench")
        resolved = first.resolve_show(show)
        item = resolved["landmark:room/bench"]
        self.assertEqual(item.origin, "generated")
        self.assertEqual(item.source, GENERATED_SOURCE)
        self.assertEqual(generator.calls, 1)
        vertices, faces = read_schema_mesh(item.glb)
        np.testing.assert_allclose(vertices.min(axis=0), [-0.5, -0.5, 0.0], atol=1e-4)
        np.testing.assert_allclose(vertices.max(axis=0) - vertices.min(axis=0), [1.0, 1.0, 1.0], atol=1e-4)
        self.assertLessEqual(len(faces), TRIANGLE_BUDGET)
        raw_copy = self.library / "raw" / GENERATED_SOURCE / appearance_source_id("a stone bench")[:16] / "model.glb"
        self.assertTrue(raw_copy.is_file())
        again = self._resolver(generator).resolve_show(show)
        self.assertEqual(again["landmark:room/bench"].prefab_id, item.prefab_id)
        self.assertEqual(generator.calls, 1)
        credits = self.output / "CREDITS.md"
        export_credits(self.library / "sources.json", credits)
        text = credits.read_text(encoding="utf-8")
        self.assertIn("MIT", text)
        self.assertIn("TRELLIS.2", text)

    def test_same_appearance_shares_one_prefab(self) -> None:
        generator = CountingGenerator(self.cube)
        show = self._show("one twisted marble column")
        show["locations"]["room"]["spatial"]["landmarks"]["column_r"] = dict(
            show["locations"]["room"]["spatial"]["landmarks"]["bench"]
        )
        resolved = self._resolver(generator).resolve_show(show)
        left = resolved["landmark:room/bench"].prefab_id
        right = resolved["landmark:room/column_r"].prefab_id
        self.assertEqual(left, right)
        self.assertEqual(generator.calls, 1)

    def test_a_generated_mesh_is_scaled_to_the_landmark(self) -> None:
        generator = CountingGenerator(self.cube)
        resolved = self._resolver(generator).resolve_show(self._show("a long spear", size=(8.0, 0.2, 0.2)))
        vertices, _faces = read_schema_mesh(resolved["landmark:room/bench"].glb)
        np.testing.assert_allclose(vertices.max(axis=0) - vertices.min(axis=0), [8.0, 0.2, 0.2], atol=1e-3)

    def test_offline_never_generates_and_a_missing_appearance_fails(self) -> None:
        generator = CountingGenerator(self.cube)
        with self.assertRaises(RuntimeError):
            self._resolver(generator, offline=True).resolve_show(self._show("a wooden chair"))
        self.assertEqual(generator.calls, 0)
        with self.assertRaises(RuntimeError):
            self._resolver(generator).resolve_show(self._show(None))

    def test_set_is_gltf_y_up_and_has_no_characters(self) -> None:
        show = self._show("a stone bench")
        show["characters"] = {"ada": {"name": "Ada"}}
        show["props"] = {"cup": {"appearance": "one small cup", "sizeMeters": [0.1, 0.1, 0.12]}}
        self._resolver(CountingGenerator(self.cube)).resolve_show(show)
        document = json.loads((self.output / "demo" / "sets" / "room" / "set.json").read_text(encoding="utf-8"))
        self.assertEqual(document["space"], "gltf-y-up")
        self.assertEqual([item["id"] for item in document["instances"]], ["bench"])
        self.assertEqual(document["instances"][0]["position"], list(schema_to_gltf((1.0, 2.0, 0.0))))
        self.assertEqual(document["instances"][0]["schemaPosition"], [1.0, 2.0, 0.0])
        self.assertNotIn("ada", json.dumps(document["instances"]))
        self.assertTrue((self.output / "demo" / "sets" / "room" / "set.glb").is_file())
        self.assertIn("prop:cup", json.loads((self.output / "demo" / "assets" / "resolved.json").read_text())["prefabs"])

    def test_unused_library_prefabs_are_removed(self) -> None:
        generator = CountingGenerator(self.cube)
        show = self._show("a stone bench")
        resolved = self._resolver(generator).resolve_show(show)
        used = resolved["landmark:room/bench"].prefab_id
        self._plant(self.library / "prefabs", "blocks/unused-table", "unused", "Unused", "library")
        self._plant(self.library / "prefabs", "blocks/other-show", None, "Other", "library")
        other = self._show(prefab_id="blocks/other-show")
        raw = self.library / "raw" / "library" / "unused-table"
        raw.mkdir(parents=True)
        (raw / "model.glb").write_bytes(self.cube.read_bytes())
        removed = prune_unused_library(self.library, [show, other])
        self.assertIn("blocks/unused-table", removed)
        self.assertFalse((self.library / "prefabs" / "blocks" / "unused-table").exists())
        self.assertFalse(raw.exists())
        self.assertTrue((self.library / "prefabs" / "blocks" / "other-show" / "prefab.json").is_file())
        self.assertTrue((self.library / "prefabs" / Path(*used.split("/")) / "prefab.json").is_file())

    def test_lock_override_pins_a_known_asset(self) -> None:
        self._resolver(CountingGenerator(self.cube)).resolve_show(self._show("a stone bench"))
        lock_path = self.library / "lock.json"
        digest = next(iter(json.loads(lock_path.read_text(encoding="utf-8"))["needs"]))
        apply_lock_override(lock_path, digest, "blocks/picked")
        updated = json.loads(lock_path.read_text(encoding="utf-8"))["needs"][digest]
        self.assertEqual(updated["prefabId"], "blocks/picked")
        self.assertEqual(updated["origin"], "override")
        self.assertFalse(updated["provisional"])

    def _resolver(self, generator: CountingGenerator, **kwargs) -> AssetResolver:
        return AssetResolver(
            "demo",
            library_dir=self.library,
            shows_dir=self.shows,
            output_dir=self.output,
            generator=generator,
            **kwargs,
        )

    def _show(
        self,
        appearance: str | None = "a stone bench",
        size: tuple[float, float, float] = (1.0, 1.0, 1.0),
        prefab_id: str | None = None,
    ) -> dict:
        landmark: dict = {
            "position": [1.0, 2.0, 0.0],
            "size": list(size),
        }
        if appearance:
            landmark["appearance"] = appearance
        if prefab_id:
            landmark["prefabId"] = prefab_id
        return {
            "id": "demo",
            "locations": {
                "room": {
                    "spatial": {
                        "sizeMeters": [8, 10, 4],
                        "landmarks": {"bench": landmark},
                    }
                }
            },
        }

    def _plant(self, root: Path, prefab_id: str, digest: str | None, title: str, origin: str) -> None:
        category, name = prefab_id.split("/", 1)
        directory = root / category / name
        directory.mkdir(parents=True, exist_ok=True)
        write_schema_glb(directory / "model.glb", *box_mesh((1.0, 1.0, 1.0)))
        record = {
            "id": prefab_id,
            "assetHash": digest,
            "sizeMeters": [1, 1, 1],
            "origin": origin,
            "source": origin,
            "sourceId": name,
            "title": title,
            "license": "CC0",
        }
        (directory / "prefab.json").write_text(json.dumps(record), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
