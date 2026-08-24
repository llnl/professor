# -*- mode: python ; coding: utf-8 -*-
"""Portable onedir specification for the Professor Dash GUI.

Run this specification with PyInstaller on each target operating system. PyInstaller
packages native libraries and therefore does not cross-compile.
"""

# ruff: noqa: F821, I001, UP009

from __future__ import annotations

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules


spec_dir = Path(SPECPATH).resolve()
project_root = spec_dir.parents[2]
entry_point = spec_dir / "run_prof_dash_gui.py"
logo = spec_dir / "logo_tiny.jpg"

# User configurations may name application-specific Python modules that cannot be
# inferred at build time. Supply them as a comma-separated list when needed.
extra_hidden_imports = [
    name.strip() for name in os.environ.get("PROFESSOR_HIDDEN_IMPORTS", "").split(",") if name.strip()
]

hidden_imports = [
    "professor.layers",
    "professor.torch_models",
]
hidden_imports += collect_submodules("torchlayers")
hidden_imports += extra_hidden_imports

analysis = Analysis(
    [str(entry_point)],
    pathex=[str(project_root / "src")],
    binaries=[],
    datas=[],
    hiddenimports=hidden_imports,
    hookspath=[str(spec_dir)],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "napari",
        "PyQt5",
        "PyQt6",
        "PySide2",
        "PySide6",
    ],
    noarchive=False,
)

pyz = PYZ(analysis.pure)

icon = None
if sys.platform == "darwin" or sys.platform.startswith("win"):
    icon = str(logo)

executable = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="prof-dash-gui",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon,
)

distribution = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="prof-dash-gui",
)

if sys.platform == "darwin":
    app = BUNDLE(
        distribution,
        name="Professor Dash GUI.app",
        icon=str(logo),
        bundle_identifier="gov.llnl.professor.dash-gui",
        info_plist={
            "CFBundleDisplayName": "Professor Dash GUI",
            "NSHighResolutionCapable": True,
        },
    )
