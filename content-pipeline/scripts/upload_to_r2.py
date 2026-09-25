#!/usr/bin/env python3
"""Upload generated MP4s + thumbnails to R2 and register episodes on the Worker."""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

from ffmpeg_tools import extract_thumbnail

from pipeline_paths import OUTPUT_DIR, episode_video_path, thumbnail_path
API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8787")
ADMIN_SECRET = os.environ.get("ADMIN_SECRET", "dev-admin-secret")
R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET = os.environ.get("R2_BUCKET", "reelshort-videos")


def admin_request(path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{API_BASE_URL}{path}",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {ADMIN_SECRET}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as res:
        return json.loads(res.read().decode("utf-8"))


def upload_via_s3(local_path: Path, key: str, content_type: str) -> None:
    import boto3

    client = boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
    )
    client.upload_file(
        str(local_path),
        R2_BUCKET,
        key,
        ExtraArgs={"ContentType": content_type},
    )


def upload_via_worker(local_path: Path, key: str) -> None:
    boundary = "----ReelShortBoundary"
    filename = local_path.name
    file_bytes = local_path.read_bytes()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="key"\r\n\r\n{key}\r\n'
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")
    req = urllib.request.Request(
        f"{API_BASE_URL}/admin/upload",
        data=body,
        headers={
            "Authorization": f"Bearer {ADMIN_SECRET}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as res:
        json.loads(res.read().decode("utf-8"))


def pretty_title(slug: str) -> str:
    return slug.replace("-", " ").title()


def ensure_series(slug: str, cover_image_url: str | None) -> str:
    series_id = f"series-{slug}"
    payload = {
        "id": series_id,
        "title": pretty_title(slug),
        "slug": slug,
        "isPublished": True,
    }
    if cover_image_url:
        payload["coverImageUrl"] = cover_image_url
    admin_request("/admin/series", payload)
    return series_id


def upload_object(local_path: Path, key: str, content_type: str) -> None:
    if R2_ACCOUNT_ID and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY:
        upload_via_s3(local_path, key, content_type)
        return
    upload_via_worker(local_path, key)


def locate_episode_video(episode_directory: Path) -> Path | None:
    """Prefer the stage layout, then the flat layout from before the path refactor."""
    show_id = episode_directory.parent.name
    episode_number = int(episode_directory.name)
    current = episode_video_path(show_id, episode_number)
    if current.is_file():
        return current
    legacy = episode_directory / "episode.mp4"
    if legacy.is_file():
        return legacy
    clips = sorted((episode_directory / "03_postvis" / "clips").glob("scene_*.mp4"))
    if clips:
        return clips[0]
    legacy_scenes = sorted(
        path for path in episode_directory.glob("scene_*.mp4") if path.is_file()
    )
    return legacy_scenes[0] if legacy_scenes else None


def locate_thumbnail(episode_directory: Path, video: Path) -> Path:
    if video.parent.name == "04_edit":
        return thumbnail_path(episode_directory.parent.name, int(episode_directory.name))
    return episode_directory / "thumbnail.jpg"


def main() -> None:
    manifests = sorted(OUTPUT_DIR.glob("generate/*/*/manifest.json"))
    if not manifests:
        raise SystemExit(
            f"No generated episodes in {OUTPUT_DIR}. Run `pnpm run content:generate` first."
        )
    for manifest_path in manifests:
        script = json.loads(manifest_path.read_text(encoding="utf-8"))
        episode_dir = manifest_path.parent
        mp4 = locate_episode_video(episode_dir)
        if mp4 is None:
            print(f"Skipping {episode_dir}: no mp4 output")
            continue
        slug = script["series"]
        title = script.get("title") or f"Episode {script['episodeNumber']}"
        video_key = f"{slug}/{script['episodeNumber']}/episode.mp4"
        upload_object(mp4, video_key, "video/mp4")
        thumb = locate_thumbnail(episode_dir, mp4)
        thumb_key = None
        if extract_thumbnail(mp4, thumb):
            thumb_key = f"{slug}/{script['episodeNumber']}/thumbnail.jpg"
            upload_object(thumb, thumb_key, "image/jpeg")
        series_id = ensure_series(slug, thumb_key)
        admin_request(
            "/admin/episodes",
            {
                "seriesId": series_id,
                "order": script["episodeNumber"],
                "title": title,
                "videoUrl": video_key,
                "thumbnailUrl": thumb_key,
                "coinCost": script.get("coinCost", 0),
                "isFree": script.get("isFree", False),
            },
        )
        print(f"Uploaded and registered {slug} episode {script['episodeNumber']}")


if __name__ == "__main__":
    main()
