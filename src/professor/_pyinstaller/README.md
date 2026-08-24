# Building `prof-dash-gui` with PyInstaller

PyInstaller bundles native Python extensions, so build once on each target operating
system rather than attempting to cross-compile. The same specification supports Linux,
macOS, and Windows and creates an onedir distribution under `dist/`.

Install Professor with its Dash dependencies and PyInstaller into the project virtual
environment, then run from the repository root:

```console
python -m pip install -e '.[dash]' pyinstaller
python -m PyInstaller --clean --noconfirm \
    src/professor/_pyinstaller/prof-dash-gui.spec
```

On Windows, use the same command on one line or replace the shell continuation with
PowerShell's backtick. The output executable is:

- Linux: `dist/prof-dash-gui/prof-dash-gui`
- Windows: `dist/prof-dash-gui/prof-dash-gui.exe`
- macOS CLI: `dist/prof-dash-gui/prof-dash-gui`
- macOS app bundle: `dist/Professor Dash GUI.app`

Pass arguments exactly as with the installed command, for example:

```console
dist/prof-dash-gui/prof-dash-gui configs/pchip.yaml --port 8888
```

The application prints the local URL to the console and does not open a browser.

## Configuration-provided imports

Professor configuration files can load model or helper modules dynamically. The spec
already includes the built-in `professor.layers` and `professor.torch_models` modules.
For additional installed modules, provide a comma-separated list at build time:

```console
PROFESSOR_HIDDEN_IMPORTS=my_package.models,my_package.helpers \
    python -m PyInstaller --clean --noconfirm \
    src/professor/_pyinstaller/prof-dash-gui.spec
```

The listed packages must be installed in the build environment. PyInstaller cannot
embed arbitrary Python source files that a configuration refers to only at runtime.

## Platform notes

- Build on the oldest Linux distribution that the binary must support; glibc is
  generally backward-incompatible.
- Build separately for each intended macOS architecture. Code signing and notarization
  are deployment steps and are intentionally not configured in the spec.
- The Windows build should be produced on Windows. Microsoft Visual C++ and hardware
  accelerator runtimes required by the selected PyTorch wheel remain platform runtime
  considerations.
