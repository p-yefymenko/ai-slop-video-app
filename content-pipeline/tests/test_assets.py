from __future__ import annotations

import importlib.util
import json
import os
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
    export_credits,
    keyword_score,
    need_hash,
)
from asset_sources import (  # noqa: E402
    Candidate,
    LocalPackSource,
    Need,
    ObjaverseSource,
    PolyHavenSource,
    TextTo3DSource,
    materialize_mesh,
    normalize_license,
    objaverse_license_allowed,
)
from build_assets import load_clip_score  # noqa: E402
from coords import schema_to_gltf  # noqa: E402
from fetch_asset import candidate_for_url  # noqa: E402
from mesh_io import (  # noqa: E402
    box_mesh,
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

    def search(self, need: Need, limit: int = 4) -> list[Candidate]:
        self.calls += 1
        return list(self.candidates)[:limit]

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
        self.assertTrue(objaverse_license_allowed("CC0"))
        self.assertTrue(objaverse_license_allowed("CC-BY"))
        self.assertFalse(objaverse_license_allowed("CC-BY-NC"))

    def test_text_to_3d_and_objaverse_stay_idle(self) -> None:
        with self.assertRaises(RuntimeError):
            TextTo3DSource().search(Need("chair"))
        os.environ.pop("OBJAVERSE_ENABLE", None)
        self.assertEqual(ObjaverseSource().search(Need("chair")), [])

    def test_local_pack_indexes_files_already_on_disk(self) -> None:
        pack = self.library / "raw" / "kenney"
        pack.mkdir(parents=True)
        (pack / "wooden-chair.glb").write_bytes(self.cube.read_bytes())
        found = LocalPackSource("kenney", pack, "Kenney").search(Need("wooden chair"))
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].license, "CC0")
        self.assertTrue(found[0].local_path)

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

    def test_polyhaven_fetch_creates_texture_directories(self) -> None:
        payload = {
            "gltf": {
                "1k": {
                    "gltf": {
                        "url": "https://example.test/gothic_coffee_table_1k.gltf",
                        "include": {
                            "textures/gothic_coffee_table_nor_gl_1k.jpg": {
                                "url": "https://example.test/nor.jpg"
                            },
                            "gothic_coffee_table.bin": {"url": "https://example.test/model.bin"},
                        },
                    }
                }
            }
        }

        class Body:
            def __init__(self, data: bytes) -> None:
                self.data = data

            def read(self) -> bytes:
                return self.data

            def __enter__(self):
                return self

            def __exit__(self, *args) -> bool:
                return False

        def urlopen(request, timeout=30):
            url = getattr(request, "full_url", str(request))
            if url.endswith("/files/gothic_coffee_table"):
                return Body(json.dumps(payload).encode("utf-8"))
            return Body(b"file")

        directory = self.library / "raw" / "polyhaven" / "gothic-coffee-table"
        with patch("asset_sources.urllib.request.urlopen", urlopen):
            fetched = PolyHavenSource().fetch(
                self._candidate("gothic_coffee_table", "Gothic Coffee Table"),
                directory,
            )
        self.assertEqual(fetched.name, "gothic_coffee_table_1k.gltf")
        self.assertTrue((directory / "textures" / "gothic_coffee_table_nor_gl_1k.jpg").is_file())
        self.assertTrue((directory / "gothic_coffee_table.bin").is_file())

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

    def test_show_prefab_beats_the_library_and_skips_search(self) -> None:
        need = Need("wooden chair")
        size = (1.0, 1.0, 1.0)
        digest = need_hash(need, size)
        self._plant(self.shows / "demo" / "assets", "blocks/show-chair", digest, "from show", "show")
        self._plant(self.library / "prefabs", "blocks/library-chair", digest, "from library", "library")
        source = CountingSource([self._candidate("remote", "wooden chair")])
        resolver = self._resolver(source)
        resolved = resolver.resolve_show(self._show(need="wooden chair"))
        item = resolved["landmark:room/bench"]
        self.assertEqual(item.origin, "show")
        self.assertEqual(item.title, "from show")
        self.assertEqual(source.calls, 0)

    def test_pinned_prefab_skips_search(self) -> None:
        self._plant(self.shows / "demo" / "assets", "blocks/custom-chair", None, "pinned", "show")
        source = CountingSource([self._candidate("remote", "wooden chair")])
        resolver = self._resolver(source)
        show = self._show(need="wooden chair", prefab_id="blocks/custom-chair")
        resolved = resolver.resolve_show(show)
        self.assertEqual(resolved["landmark:room/bench"].prefab_id, "blocks/custom-chair")
        self.assertEqual(source.calls, 0)

    def test_download_is_normalized_and_a_second_resolve_does_not_search(self) -> None:
        source = CountingSource([self._candidate("stone-chair", "stone chair", license_name="CC-BY")])
        first = self._resolver(source)
        show = self._show(need="stone chair")
        resolved = first.resolve_show(show)
        item = resolved["landmark:room/bench"]
        self.assertEqual(item.origin, "downloaded")
        self.assertEqual(source.calls, 1)
        vertices, _faces = read_schema_mesh(item.glb)
        np.testing.assert_allclose(vertices.min(axis=0), [ -0.5, -0.5, 0.0], atol=1e-4)
        np.testing.assert_allclose(vertices.max(axis=0) - vertices.min(axis=0), [1.0, 1.0, 1.0], atol=1e-4)
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
        self.assertIn("Powered by Poly Haven", text)

    def test_same_need_shares_one_prefab(self) -> None:
        source = CountingSource([self._candidate("column", "stone column")])
        show = self._show(need="stone column")
        show["locations"]["room"]["spatial"]["landmarks"]["column_r"] = dict(
            show["locations"]["room"]["spatial"]["landmarks"]["bench"]
        )
        resolved = self._resolver(source).resolve_show(show)
        left = resolved["landmark:room/bench"].prefab_id
        right = resolved["landmark:room/column_r"].prefab_id
        self.assertEqual(left, right)
        self.assertEqual(source.calls, 1)

    def test_bad_license_and_bad_proportions_fall_back(self) -> None:
        forbidden = CountingSource([self._candidate("collar", "iron collar", license_name="CC-BY-NC")])
        forbidden_resolver = self._resolver(forbidden)
        forbidden_resolved = forbidden_resolver.resolve_show(self._show(need="iron collar"))
        self.assertEqual(forbidden_resolved["landmark:room/bench"].origin, "fallback")
        self.assertTrue(any("primitive" in warning for warning in forbidden_resolver.warnings))

        long_request = self._show(need="spear", size=(8.0, 0.2, 0.2))
        spear = CountingSource([self._candidate("spear", "spear")])
        spear_resolver = self._resolver(spear)
        spear_resolved = spear_resolver.resolve_show(long_request)
        self.assertEqual(spear_resolved["landmark:room/bench"].origin, "fallback")
        self.assertTrue(any("proportions" in warning for warning in spear_resolver.warnings))

    def test_offline_never_searches_and_online_retries_a_provisional_lock(self) -> None:
        offline_source = CountingSource([self._candidate("chair", "wooden chair")])
        offline = self._resolver(offline_source, offline=True)
        show = self._show(need="wooden chair")
        offline.resolve_show(show)
        self._resolver(offline_source, offline=True).resolve_show(show)
        self.assertEqual(offline_source.calls, 0)
        lock = json.loads((self.library / "lock.json").read_text(encoding="utf-8"))
        digest = next(iter(lock["needs"]))
        self.assertTrue(lock["needs"][digest]["provisional"])

        empty = CountingSource([])
        self._resolver(empty).resolve_show(self._show(need="missing prop"))
        self._resolver(empty).resolve_show(self._show(need="missing prop"))
        self.assertEqual(empty.calls, 2)

    def test_clip_can_override_keyword_rank(self) -> None:
        if importlib.util.find_spec("open_clip") is None:
            with self.assertRaises(SystemExit):
                load_clip_score()
        worse = self._candidate("worse-chair", "stone column")
        better = self._candidate("better-chair", "apple")
        source = CountingSource([worse, better])

        def clip_fn(query: str, thumb: Path) -> float:
            return 5.0 if "better-chair" in thumb.as_posix() else 0.1

        resolved = self._resolver(source, clip_fn=clip_fn).resolve_show(self._show(need="stone column"))
        self.assertEqual(resolved["landmark:room/bench"].source_id, "better-chair")

    def test_set_is_gltf_y_up_and_has_no_characters(self) -> None:
        show = self._show(need="bench")
        show["characters"] = {"ada": {"name": "Ada"}}
        show["props"] = {"cup": {"need": {"query": "cup"}, "sizeMeters": [0.1, 0.1, 0.12]}}
        self._resolver(CountingSource([]), offline=True).resolve_show(show)
        document = json.loads((self.output / "demo" / "sets" / "room" / "set.json").read_text(encoding="utf-8"))
        self.assertEqual(document["space"], "gltf-y-up")
        self.assertEqual([item["id"] for item in document["instances"]], ["bench"])
        self.assertEqual(document["instances"][0]["position"], list(schema_to_gltf((1.0, 2.0, 0.0))))
        self.assertEqual(document["instances"][0]["schemaPosition"], [1.0, 2.0, 0.0])
        self.assertNotIn("ada", json.dumps(document["instances"]))
        self.assertTrue((self.output / "demo" / "sets" / "room" / "set.glb").is_file())
        self.assertIn("prop:cup", json.loads((self.output / "demo" / "assets" / "resolved.json").read_text())["prefabs"])

    def test_lock_override_pins_a_known_need(self) -> None:
        source = CountingSource([self._candidate("stone-chair", "stone chair")])
        self._resolver(source).resolve_show(self._show(need="stone chair"))
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
        candidate, _fetch = candidate_for_url("https://polyhaven.com/a/wooden_crate")
        self.assertEqual(candidate.source, "polyhaven")
        self.assertEqual(candidate.license, "CC0")

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
        need: str,
        size: tuple[float, float, float] = (1.0, 1.0, 1.0),
        prefab_id: str | None = None,
    ) -> dict:
        landmark = {
            "kind": "box",
            "position": [1.0, 2.0, 0.0],
            "size": list(size),
            "need": {"query": need},
        }
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
            "needHash": digest,
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
