from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from asset_resolver import (  # noqa: E402
    AssetResolver,
    apply_lock_override,
    asset_hash,
    export_credits,
    keyword_score,
    prune_unused_library,
)
from asset_sources import (  # noqa: E402
    Candidate,
    Need,
    SketchfabSource,
    TextTo3DSource,
    materialize_mesh,
    normalize_license,
)
from coords import schema_to_gltf  # noqa: E402
from fetch_asset import candidate_for_url  # noqa: E402
from mesh_io import (  # noqa: E402
    TRIANGLE_BUDGET,
    box_mesh,
    decimate,
    primitive_mesh,
    proportions_match,
    read_schema_mesh,
    write_schema_glb,
)


class CountingSource:
    name = "fake"

    def __init__(self, candidates: list[Candidate]) -> None:
        self.candidates = candidates
        self.calls = 0

    def lookup(self, source_id: str) -> Candidate | None:
        self.calls += 1
        for candidate in self.candidates:
            if candidate.source_id == source_id:
                return candidate
        return None

    def search(self, need: Need, limit: int = 4) -> list[Candidate]:
        raise AssertionError("catalog search is not used")

    def fetch(self, candidate: Candidate, directory: Path) -> Path:
        raise AssertionError("local_path should skip the network fetch")


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

    def test_licenses_drop_nc_nd_and_sa(self) -> None:
        self.assertEqual(normalize_license("CC0"), "CC0")
        self.assertEqual(normalize_license("cc-by"), "CC-BY")
        self.assertIsNone(normalize_license("CC-BY-NC"))
        self.assertIsNone(normalize_license("CC-BY-SA"))
        self.assertIsNone(normalize_license("CC-BY-ND"))
        self.assertIsNone(normalize_license("CC-BY-NC-SA"))
        self.assertEqual(normalize_license("CC Attribution"), "CC-BY")
        self.assertEqual(normalize_license("by"), "CC-BY")

    def test_text_to_3d_stays_unregistered(self) -> None:
        with self.assertRaises(RuntimeError):
            TextTo3DSource().search(Need("chair"))

    def test_keyword_score_prefers_the_matching_title(self) -> None:
        need = Need("stone column", ("castle",))
        matching = self._candidate("stone-column", "stone column", tags=("castle",))
        other = self._candidate("apple", "apple")
        self.assertGreater(keyword_score(need, matching), keyword_score(need, other))

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

    def test_sketchfab_fetch_extracts_the_archive(self) -> None:
        def fake_get_json(url: str, headers: dict | None = None) -> dict:
            del url, headers
            return {"glb": {"url": "https://example.test/model.zip"}}

        def fake_download(url: str, destination: Path) -> None:
            del url
            destination.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(destination, "w") as archive:
                archive.write(self.cube, "model.glb")

        directory = self.library / "raw" / "sketchfab" / "anvil"
        candidate = self._candidate("e63f1154ee0b41f8a797db683526142a", "Blacksmith Anvil")
        candidate = Candidate(
            source="sketchfab",
            source_id=candidate.source_id,
            title=candidate.title,
            author=candidate.author,
            license=candidate.license,
            page_url=candidate.page_url,
            download_url="https://api.sketchfab.com/v3/models/e63f1154ee0b41f8a797db683526142a/download",
        )
        with patch("asset_sources._get_json", fake_get_json), patch("asset_sources._download", fake_download):
            fetched = SketchfabSource("token").fetch(candidate, directory)
        self.assertEqual(fetched.suffix, ".glb")
        _vertices, faces = read_schema_mesh(fetched)
        self.assertGreater(len(faces), 0)

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

    def test_show_prefab_beats_the_library_and_skips_lookup(self) -> None:
        size = (1.0, 1.0, 1.0)
        digest = asset_hash("fake", "wooden-chair", size)
        self._plant(self.shows / "demo" / "assets", "blocks/show-chair", digest, "from show", "show")
        self._plant(self.library / "prefabs", "blocks/library-chair", digest, "from library", "library")
        source = CountingSource([self._candidate("wooden-chair", "wooden chair")])
        resolver = self._resolver(source)
        resolved = resolver.resolve_show(self._show(asset_id="fake:wooden-chair"))
        item = resolved["landmark:room/bench"]
        self.assertEqual(item.origin, "show")
        self.assertEqual(item.title, "from show")
        self.assertEqual(source.calls, 0)

    def test_pinned_prefab_skips_lookup(self) -> None:
        self._plant(self.shows / "demo" / "assets", "blocks/custom-chair", None, "pinned", "show")
        source = CountingSource([self._candidate("wooden-chair", "wooden chair")])
        resolver = self._resolver(source)
        show = self._show(asset_id="fake:wooden-chair", prefab_id="blocks/custom-chair")
        resolved = resolver.resolve_show(show)
        self.assertEqual(resolved["landmark:room/bench"].prefab_id, "blocks/custom-chair")
        self.assertEqual(source.calls, 0)

    def test_download_is_normalized_and_a_second_resolve_does_not_fetch(self) -> None:
        source = CountingSource([self._candidate("stone-chair", "stone chair", license_name="CC-BY")])
        first = self._resolver(source)
        show = self._show(asset_id="fake:stone-chair")
        resolved = first.resolve_show(show)
        item = resolved["landmark:room/bench"]
        self.assertEqual(item.origin, "downloaded")
        self.assertEqual(source.calls, 1)
        vertices, faces = read_schema_mesh(item.glb)
        np.testing.assert_allclose(vertices.min(axis=0), [-0.5, -0.5, 0.0], atol=1e-4)
        np.testing.assert_allclose(vertices.max(axis=0) - vertices.min(axis=0), [1.0, 1.0, 1.0], atol=1e-4)
        self.assertLessEqual(len(faces), TRIANGLE_BUDGET)
        raw_copy = self.library / "raw" / "fake" / "stone-chair" / "cube.glb"
        self.assertTrue(raw_copy.is_file())
        second = self._resolver(source)
        again = second.resolve_show(show)
        self.assertEqual(again["landmark:room/bench"].prefab_id, item.prefab_id)
        self.assertEqual(source.calls, 1)
        credits = self.output / "CREDITS.md"
        export_credits(self.library / "sources.json", credits)
        text = credits.read_text(encoding="utf-8")
        self.assertIn("CC-BY", text)
        self.assertIn("Sketchfab CC-BY", text)

    def test_same_asset_id_shares_one_prefab(self) -> None:
        source = CountingSource([self._candidate("column", "stone column")])
        show = self._show(asset_id="fake:column")
        show["locations"]["room"]["spatial"]["landmarks"]["column_r"] = dict(
            show["locations"]["room"]["spatial"]["landmarks"]["bench"]
        )
        resolved = self._resolver(source).resolve_show(show)
        left = resolved["landmark:room/bench"].prefab_id
        right = resolved["landmark:room/column_r"].prefab_id
        self.assertEqual(left, right)
        self.assertEqual(source.calls, 1)

    def test_bad_license_fails_and_a_named_mesh_is_scaled(self) -> None:
        forbidden = CountingSource([self._candidate("collar", "iron collar", license_name="CC-BY-NC")])
        forbidden_resolver = self._resolver(forbidden)
        with self.assertRaises(RuntimeError):
            forbidden_resolver.resolve_show(self._show(asset_id="fake:collar"))
        self.assertTrue(any("license" in warning for warning in forbidden_resolver.warnings))

        long_request = self._show(asset_id="fake:spear", size=(8.0, 0.2, 0.2))
        spear = CountingSource([self._candidate("spear", "spear")])
        spear_resolver = self._resolver(spear)
        spear_resolved = spear_resolver.resolve_show(long_request)
        item = spear_resolved["landmark:room/bench"]
        self.assertEqual(item.origin, "downloaded")
        vertices, _faces = read_schema_mesh(item.glb)
        np.testing.assert_allclose(vertices.max(axis=0) - vertices.min(axis=0), [8.0, 0.2, 0.2], atol=1e-3)

    def test_offline_never_looks_up_and_a_missing_model_fails(self) -> None:
        offline_source = CountingSource([self._candidate("chair", "wooden chair")])
        show = self._show(asset_id="fake:chair")
        with self.assertRaises(RuntimeError):
            self._resolver(offline_source, offline=True).resolve_show(show)
        self.assertEqual(offline_source.calls, 0)

        empty = CountingSource([])
        with self.assertRaises(RuntimeError):
            self._resolver(empty).resolve_show(self._show(asset_id="fake:missing-prop"))
        with self.assertRaises(RuntimeError):
            self._resolver(empty).resolve_show(self._show(asset_id="fake:missing-prop"))
        self.assertEqual(empty.calls, 2)

    def test_set_is_gltf_y_up_and_has_no_characters(self) -> None:
        show = self._show(asset_id="fake:bench")
        show["characters"] = {"ada": {"name": "Ada"}}
        show["props"] = {"cup": {"assetId": "fake:cup", "sizeMeters": [0.1, 0.1, 0.12]}}
        source = CountingSource(
            [self._candidate("bench", "bench"), self._candidate("cup", "cup")]
        )
        self._resolver(source).resolve_show(show)
        document = json.loads((self.output / "demo" / "sets" / "room" / "set.json").read_text(encoding="utf-8"))
        self.assertEqual(document["space"], "gltf-y-up")
        self.assertEqual([item["id"] for item in document["instances"]], ["bench"])
        self.assertEqual(document["instances"][0]["position"], list(schema_to_gltf((1.0, 2.0, 0.0))))
        self.assertEqual(document["instances"][0]["schemaPosition"], [1.0, 2.0, 0.0])
        self.assertNotIn("ada", json.dumps(document["instances"]))
        self.assertTrue((self.output / "demo" / "sets" / "room" / "set.glb").is_file())
        self.assertIn("prop:cup", json.loads((self.output / "demo" / "assets" / "resolved.json").read_text())["prefabs"])

    def test_unused_library_prefabs_are_removed(self) -> None:
        source = CountingSource([self._candidate("stone-chair", "stone chair")])
        show = self._show(asset_id="fake:stone-chair")
        resolved = self._resolver(source).resolve_show(show)
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
        source = CountingSource([self._candidate("stone-chair", "stone chair")])
        self._resolver(source).resolve_show(self._show(asset_id="fake:stone-chair"))
        lock_path = self.library / "lock.json"
        digest = next(iter(json.loads(lock_path.read_text(encoding="utf-8"))["needs"]))
        apply_lock_override(lock_path, digest, "blocks/picked")
        updated = json.loads(lock_path.read_text(encoding="utf-8"))["needs"][digest]
        self.assertEqual(updated["prefabId"], "blocks/picked")
        self.assertEqual(updated["origin"], "override")
        self.assertFalse(updated["provisional"])

    def test_fetch_asset_refuses_unknown_hosts(self) -> None:
        with self.assertRaises(SystemExit):
            candidate_for_url("https://www.mixamo.com/characters")
        with self.assertRaises(SystemExit):
            candidate_for_url("https://example.com/model.glb")
        with self.assertRaises(SystemExit):
            candidate_for_url("https://polyhaven.com/a/wooden_crate")

    def _resolver(self, source: CountingSource, **kwargs) -> AssetResolver:
        return AssetResolver(
            "demo",
            library_dir=self.library,
            shows_dir=self.shows,
            output_dir=self.output,
            sources=[source],
            **kwargs,
        )

    def _candidate(
        self,
        source_id: str,
        title: str,
        license_name: str = "CC0",
        tags: tuple[str, ...] = (),
    ) -> Candidate:
        return Candidate(
            source="fake",
            source_id=source_id,
            title=title,
            author="Tester",
            license=license_name,
            page_url=f"https://example.test/{source_id}",
            tags=tags,
            local_path=str(self.cube),
        )

    def _show(
        self,
        asset_id: str | None = None,
        size: tuple[float, float, float] = (1.0, 1.0, 1.0),
        prefab_id: str | None = None,
    ) -> dict:
        landmark = {
            "position": [1.0, 2.0, 0.0],
            "size": list(size),
        }
        if asset_id:
            landmark["assetId"] = asset_id
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
