"""Collect TorchVision's native extensions for PyInstaller bundles."""

# ruff: noqa: I001, N999

from pathlib import Path

from PyInstaller.utils.hooks import get_package_paths


package_dir = Path(get_package_paths("torchvision")[1])

# TorchVision keeps its native extensions (for example, ``_C.so``) directly in
# the package directory.  PyInstaller's generic dynamic-library collection can
# miss extensions at that level, while torchvision's Python modules still get
# collected.  Include every platform-native extension so the package can load
# its registered operators at runtime.
binaries = [
    (str(path), "torchvision") for path in package_dir.iterdir() if path.suffix in {".so", ".dylib", ".pyd", ".dll"}
]
