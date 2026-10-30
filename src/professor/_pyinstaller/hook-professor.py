"""PyInstaller hook for data and dynamically imported Professor model classes."""

# ruff: noqa: I001, N999

from PyInstaller.utils.hooks import collect_data_files, copy_metadata


datas = collect_data_files("professor", excludes=["_pyinstaller/**"])
datas += copy_metadata("professor")

# Config files resolve these modules at runtime through importlib. PyInstaller cannot
# discover them by following imports from professor.vela.model.
hiddenimports = [
    "professor.layers",
    "professor.torch_models",
]
