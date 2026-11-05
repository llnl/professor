"""Download the pinned Bootstrap CSS used by the Dash GUI.

Run this at build time before packaging the dashboard. The stylesheet is installed
in the Dash application's normal ``assets`` directory and served by Dash at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

BOOTSTRAP_VERSION = "5.3.8"
ARCHIVE_NAME = f"bootstrap-{BOOTSTRAP_VERSION}-dist.zip"
ARCHIVE_URL = f"https://github.com/twbs/bootstrap/releases/download/v{BOOTSTRAP_VERSION}/{ARCHIVE_NAME}"
ARCHIVE_SHA256 = "3258c873cbcb1e2d81f4374afea2ea6437d9eee9077041073fd81dd579c5ba6b"
CSS_ARCHIVE_PATH = f"bootstrap-{BOOTSTRAP_VERSION}-dist/css/bootstrap.min.css"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".build-assets"
DEFAULT_ASSETS_DIR = PROJECT_ROOT / "src" / "professor" / "gui" / "assets"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as archive_file:
        for chunk in iter(lambda: archive_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(destination: Path) -> None:
    """Download the release archive to a temporary file, then verify and cache it."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        ARCHIVE_URL,
        headers={"User-Agent": "professor-bootstrap-assets-builder"},
    )

    temporary_path: Path | None = None
    try:
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            tempfile.NamedTemporaryFile(
                mode="wb", dir=destination.parent, prefix=f"{ARCHIVE_NAME}.", suffix=".part", delete=False
            ) as temporary_file,
        ):
            temporary_path = Path(temporary_file.name)
            shutil.copyfileobj(response, temporary_file)

        actual_sha256 = sha256_file(temporary_path)
        if actual_sha256 != ARCHIVE_SHA256:
            raise ValueError(f"Downloaded archive checksum mismatch: expected {ARCHIVE_SHA256}, got {actual_sha256}")

        os.replace(temporary_path, destination)
    except (OSError, urllib.error.URLError, ValueError):
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def verify_archive(archive_path: Path) -> None:
    if not archive_path.is_file():
        raise FileNotFoundError(f"Bootstrap archive does not exist: {archive_path}")

    actual_sha256 = sha256_file(archive_path)
    if actual_sha256 != ARCHIVE_SHA256:
        raise ValueError(
            f"Bootstrap archive checksum mismatch for {archive_path}: expected {ARCHIVE_SHA256}, got {actual_sha256}"
        )


def extract_stylesheet(archive_path: Path, assets_dir: Path) -> Path:
    """Extract only bootstrap.min.css, replacing the destination atomically."""
    destination = assets_dir / "bootstrap.min.css"
    assets_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(archive_path) as archive:
            stylesheet = archive.read(CSS_ARCHIVE_PATH)
    except (KeyError, zipfile.BadZipFile) as exc:
        raise ValueError(f"{archive_path} does not contain {CSS_ARCHIVE_PATH}") from exc

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=assets_dir, prefix="bootstrap.min.css.", suffix=".part", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(stylesheet)

        os.replace(temporary_path, destination)
    except OSError:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        help="use an existing Bootstrap release ZIP instead of downloading it",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"directory for the downloaded ZIP (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=DEFAULT_ASSETS_DIR,
        help=f"Dash assets directory (default: {DEFAULT_ASSETS_DIR})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    archive_path = args.archive

    try:
        if archive_path is None:
            archive_path = args.cache_dir / ARCHIVE_NAME
            if not archive_path.exists():
                print(f"Downloading Bootstrap {BOOTSTRAP_VERSION} from {ARCHIVE_URL}")
                download_archive(archive_path)
        elif not archive_path.is_absolute():
            archive_path = Path.cwd() / archive_path

        verify_archive(archive_path)
        stylesheet_path = extract_stylesheet(archive_path, args.assets_dir)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Installed Bootstrap {BOOTSTRAP_VERSION} stylesheet: {stylesheet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
