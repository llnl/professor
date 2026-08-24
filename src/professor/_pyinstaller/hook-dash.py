"""Bundle the JavaScript, favicon, and component data distributed with Dash."""

# ruff: noqa: I001, N999

from PyInstaller.utils.hooks import collect_data_files, copy_metadata


datas = collect_data_files("dash")
datas += copy_metadata("dash")
