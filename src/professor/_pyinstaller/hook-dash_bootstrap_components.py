"""Bundle Dash Bootstrap Components' generated component assets."""

# ruff: noqa: I001, N999

from PyInstaller.utils.hooks import collect_data_files, copy_metadata


datas = collect_data_files("dash_bootstrap_components")
datas += copy_metadata("dash-bootstrap-components")
