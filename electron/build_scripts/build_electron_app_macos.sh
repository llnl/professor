#!/bin/bash

set -Eeuo pipefail
trap 'status=$?; printf "Build failed with status %s at line %s: %s\n" "$status" "$LINENO" "$BASH_COMMAND" >&2' ERR
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

: "${ELECTRON_BUILD_ROOT:?Set ELECTRON_BUILD_ROOT to a writable build/install directory}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  die "This script builds a macOS app and must run on macOS."
fi
if [[ "$(uname -m)" != "arm64" ]]; then
  die "This script targets Apple Silicon (arm64); detected $(uname -m)."
fi
if [[ "$(/usr/sbin/sysctl -in sysctl.proc_translated 2>/dev/null || printf '0')" == "1" ]]; then
  die "Run this script in a native arm64 shell, not under Rosetta."
fi

for tool in clang clang++ make curl git tar pkg-config; do
  command -v "$tool" >/dev/null 2>&1 || die "Required tool '$tool' is missing. Install Xcode Command Line Tools and Homebrew prerequisites."
done
xcode-select -p >/dev/null 2>&1 || die "Xcode Command Line Tools are required. Run: xcode-select --install"

command -v brew >/dev/null 2>&1 || die "Homebrew is required to provide Python build libraries. Install Homebrew, then install: openssl@3 readline sqlite xz tcl-tk"
MISSING_FORMULAE=()
for formula in openssl@3 readline sqlite xz tcl-tk; do
  if ! brew list --versions "$formula" >/dev/null 2>&1; then
    MISSING_FORMULAE+=("$formula")
  fi
done
if (( ${#MISSING_FORMULAE[@]} > 0 )); then
  die "Install the missing Homebrew formulae before building: ${MISSING_FORMULAE[*]} (brew install ${MISSING_FORMULAE[*]})"
fi

OPENSSL_PREFIX=$(brew --prefix openssl@3)
READLINE_PREFIX=$(brew --prefix readline)
SQLITE_PREFIX=$(brew --prefix sqlite)
XZ_PREFIX=$(brew --prefix xz)
TCLTK_PREFIX=$(brew --prefix tcl-tk)
export PKG_CONFIG_PATH="${PKG_CONFIG_PATH:+${PKG_CONFIG_PATH}:}${OPENSSL_PREFIX}/lib/pkgconfig:${READLINE_PREFIX}/lib/pkgconfig:${SQLITE_PREFIX}/lib/pkgconfig:${XZ_PREFIX}/lib/pkgconfig:${TCLTK_PREFIX}/lib/pkgconfig"

TCLTK_FOUND=0
for tcltk_query in 'tcl >= 8.5.12 tk >= 8.5.12' 'tcl8.6 tk8.6' 'tcl86 tk86' 'tcl8.5 >= 8.5.12 tk8.5 >= 8.5.12' 'tcl85 >= 8.5.12 tk85 >= 8.5.12'; do
  if pkg-config --exists "$tcltk_query"; then
    TCLTK_FOUND=1
    break
  fi
done
(( TCLTK_FOUND == 1 )) || die "Homebrew Tcl/Tk pkg-config files were not found. Reinstall tcl-tk with Homebrew."

PYTHON_VERSION=${PYTHON_VERSION:-3.14.7}
NODE_VERSION=${NODE_VERSION:-24.21.0}
TORCH_VERSION=${TORCH_VERSION:-2.13.0}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.28.0}
TRIANGLE_VERSION=${TRIANGLE_VERSION:-v20250106}
PROFESSOR_REF=${PROFESSOR_REF:-feature/sherman/electronApp}
BUILD_JOBS=${BUILD_JOBS:-$(sysctl -n hw.ncpu)}
COMPILER_ID=apple-clang
TORCH_VARIANT=mac-arm64
export CC=clang
export CXX=clang++

[[ "$BUILD_JOBS" =~ ^[1-9][0-9]*$ ]] || die "BUILD_JOBS must be a positive integer."

mkdir -p "$ELECTRON_BUILD_ROOT"
ELECTRON_BUILD_ROOT=$(cd "$ELECTRON_BUILD_ROOT" && pwd -P)
export ELECTRON_BUILD_ROOT

PYTHON_PREFIX="${ELECTRON_BUILD_ROOT}/python-${PYTHON_VERSION}-${COMPILER_ID}-arm64"
PYTHON_EXE="${PYTHON_PREFIX}/bin/python3"
TORCH_KEY=$(printf '%s\n%s' "$TORCH_VERSION" "$TORCHVISION_VERSION" | shasum -a 256 | cut -c1-12)
VENV_DIR="${ELECTRON_BUILD_ROOT}/venv-python-${PYTHON_VERSION}-${COMPILER_ID}-${TORCH_VARIANT}-${TORCH_KEY}"
VENV_PYTHON="${VENV_DIR}/bin/python"
NODE_DIR="${ELECTRON_BUILD_ROOT}/node-v${NODE_VERSION}-darwin-arm64"

BUILD_DIR=$(mktemp -d "${ELECTRON_BUILD_ROOT}/build.${COMPILER_ID}.${TORCH_VARIANT}.XXXXXXXX")
printf 'Build workspace: %s\n' "$BUILD_DIR"
printf 'Install root: %s\n' "$ELECTRON_BUILD_ROOT"
printf 'Target: macOS arm64; compiler: %s; Torch: %s\n' "$COMPILER_ID" "$TORCH_VARIANT"

if [[ ! -x "$NODE_DIR/bin/node" ]]; then
  echo "Setting up Node.js ${NODE_VERSION} for macOS arm64"
  NODE_ARCHIVE="${BUILD_DIR}/node-v${NODE_VERSION}-darwin-arm64.tar.gz"
  curl --fail --location --retry 3 --output "$NODE_ARCHIVE" \
    "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-darwin-arm64.tar.gz"
  tar -xzf "$NODE_ARCHIVE" -C "$ELECTRON_BUILD_ROOT"
fi
export PATH="${NODE_DIR}/bin:${PATH}"

if [[ ! -x "$PYTHON_EXE" ]]; then
  echo "Building Python ${PYTHON_VERSION} with Apple Clang"
  PYTHON_ARCHIVE="${BUILD_DIR}/Python-${PYTHON_VERSION}.tgz"
  curl --fail --location --retry 3 --output "$PYTHON_ARCHIVE" \
    "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz"
  tar -zxf "$PYTHON_ARCHIVE" -C "$BUILD_DIR"
  cd "${BUILD_DIR}/Python-${PYTHON_VERSION}"

  HOMEBREW_CPPFLAGS="-I${OPENSSL_PREFIX}/include -I${READLINE_PREFIX}/include -I${SQLITE_PREFIX}/include -I${XZ_PREFIX}/include -I${TCLTK_PREFIX}/include"
  HOMEBREW_LDFLAGS="-L${OPENSSL_PREFIX}/lib -L${READLINE_PREFIX}/lib -L${SQLITE_PREFIX}/lib -L${XZ_PREFIX}/lib -L${TCLTK_PREFIX}/lib"
  ./configure \
    --prefix="$PYTHON_PREFIX" \
    --enable-shared \
    --with-lto \
    --enable-optimizations \
    --with-openssl="$OPENSSL_PREFIX" \
    --with-openssl-rpath=auto \
    CPPFLAGS="${CPPFLAGS:+${CPPFLAGS} }${HOMEBREW_CPPFLAGS}" \
    LDFLAGS="${LDFLAGS:+${LDFLAGS} }${HOMEBREW_LDFLAGS} -Wl,-rpath,${PYTHON_PREFIX}/lib"
  make -j"$BUILD_JOBS"
  make install

  "$PYTHON_EXE" -c 'import _tkinter, hashlib, sqlite3, ssl; print("Python build has Tk, SQLite, and SSL support")'
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Creating Python environment for macOS arm64"
  "$PYTHON_EXE" -m venv "$VENV_DIR"
fi

echo "Installing pinned macOS arm64 Python packages from PyPI"
"$VENV_PYTHON" -m pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}"
"$VENV_PYTHON" -m pip install pyinstaller
"$VENV_PYTHON" -m pip install "git+https://github.com/drufat/triangle.git@${TRIANGLE_VERSION}"

echo "Building professor executable from ${PROFESSOR_REF}"
PROFESSOR_DIR="${BUILD_DIR}/professor"
git init "$PROFESSOR_DIR"
git -C "$PROFESSOR_DIR" remote add origin https://github.com/llnl/professor.git
git -C "$PROFESSOR_DIR" fetch --depth 1 origin "$PROFESSOR_REF"
git -C "$PROFESSOR_DIR" checkout --detach FETCH_HEAD
cd "$PROFESSOR_DIR"
git rev-parse HEAD
"$VENV_PYTHON" -m pip install '.[all,dash]'

echo "Fetching Bootstrap stylesheet for PyInstaller"
"$VENV_PYTHON" src/professor/_pyinstaller/fetch_bootstrap_styles.py
"$VENV_PYTHON" -m PyInstaller --clean --noconfirm src/professor/_pyinstaller/prof-dash-gui.spec

echo "Applying Tk scaling patch to PyInstaller bundle"
"$VENV_PYTHON" "$PROJECT_ROOT/src/professor/_pyinstaller/patch_tk_scaling.py" dist/prof-dash-gui

echo "Building macOS arm64 Electron app"
cd "${PROFESSOR_DIR}/electron"
npm ci
npm run package -- --mac dir --arm64

APP_PATH=$(find "${PROFESSOR_DIR}/electron/release" -type d -name 'Professor Dash GUI.app' -print -quit)
[[ -n "$APP_PATH" ]] || die "Electron Builder finished without creating Professor Dash GUI.app under ${PROFESSOR_DIR}/electron/release"
printf 'Done! Electron app: %s\n' "$APP_PATH"
