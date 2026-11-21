# Professor Dash GUI Electron app

This directory wraps the frozen `prof-dash-gui` PyInstaller distribution in a
small Electron desktop window. Electron does not replace Python: it starts the
PyInstaller executable, waits for its local Dash server, and displays that server
in the application window.

## Requirements

Build on the operating system where the application will run. PyInstaller and
Electron do not provide a portable cross-platform build.

1. Install Python and create/use the repository virtual environment:

   ```console
   source .venv/bin/activate       # macOS/Linux
   # Windows PowerShell: .venv\Scripts\Activate.ps1
   ```

2. Build the PyInstaller distribution from the repository root. Follow
   [`src/professor/_pyinstaller/README.md`](../src/professor/_pyinstaller/README.md).
   Electron expects the result at `dist/prof-dash-gui/`.

3. Install Node.js (the current LTS release) from <https://nodejs.org/>. This
   provides both `node` and `npm`; no JavaScript knowledge is required to run the
   commands below.

## Run while developing

From this directory, install the JavaScript dependencies once:

```console
npm install
```

Start the desktop window and pass the same arguments accepted by `prof-dash-gui`:

```console
npm start -- ../configs/pchip.yaml
```

For a different port, use `npm start -- ../configs/pchip.yaml --port 8899`.
The wrapper waits for the Dash server before opening the window. Close the window
to stop the bundled Python process.

## Build a distributable app

After `npm install` and a successful PyInstaller build, run:

```console
npm run package
```

The unpacked application is written to `electron/release/`. Launch the generated
application for your operating system from there. The build includes the complete
`dist/prof-dash-gui/` directory, including its `_internal/` directory, so do not
copy only the executable by itself.

Build separately on Linux, macOS, and Windows. To request a specific Electron
Builder target, append its option, for example `npm run package -- --linux dir`.

### Automated Apple Silicon build

On a native arm64 Mac, install Xcode Command Line Tools and the Homebrew build
libraries, then run the build script from this directory:

```console
brew install openssl@3 readline sqlite xz tcl-tk pkgconf
ELECTRON_BUILD_ROOT="$PWD/build" ./build_scripts/build_electron_app_macos.sh
```

The script builds the pinned Python version and packages an arm64 `.app`. It
prints the resulting app path when the build completes.

### Linux and HPC builds

`build_scripts/build_electron_app.sh` builds on Linux x86-64. Load the compiler
toolchain you want to use, then provide an absolute build root and a compiler
label:

```console
ELECTRON_BUILD_ROOT="$PWD/build" \
COMPILER_ID=gcc-13.3.1-magic \
./build_scripts/build_electron_app.sh
```

`COMPILER_ID` labels the installed Python and virtual environment; set `CC` and
`CXX` or load a compiler module to select the actual compiler. The script uses
the CPU PyTorch wheel index by default. It prints the release directory inside
the fresh Professor checkout when the build completes.

The scheduler wrappers load the site compiler and accelerator modules and set
their own build roots and Python package versions:

| Script | Submit command | Build configuration |
| --- | --- | --- |
| `build_scripts/build_electron_app_dane.sbatch` | `sbatch build_scripts/build_electron_app_dane.sbatch` | Slurm CPU build with GCC 13.3.1. |
| `build_scripts/build_electron_app_tuolumne.flux` | `flux batch build_scripts/build_electron_app_tuolumne.flux` | Flux build with ROCm 7.2.1 and its matching PyTorch wheel index. |

Both wrappers use `$USER` to select the per-user Lustre root and shared build
script path. They still assume the site mounts and the `MADA/professor` checkout
layout. Slurm writes `electron_build_dane.out` and `electron_build_dane.err`;
Flux writes `electron_build_tuolumne.out` and `electron_build_tuolumne.err`.

## Passing user configuration files

The packaged application accepts the same command-line arguments as the Python
program. A user can launch it with a configuration path, for example:

```console
Professor\ Dash\ GUI ../configs/pchip.yaml --port 8899
```

Paths are interpreted relative to the directory from which the application is
launched. The Electron wrapper keeps the Python console output available to help
diagnose startup failures when running from a terminal.
