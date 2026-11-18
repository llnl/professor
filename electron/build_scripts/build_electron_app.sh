#!/bin/bash

set -Eeuo pipefail
trap 'status=$?; printf "Build failed with status %s at line %s: %s\n" "$status" "$LINENO" "$BASH_COMMAND" >&2' ERR

# Check build configuration
echo "Checking build configuration"

: "${ELECTRON_BUILD_ROOT:?Set ELECTRON_BUILD_ROOT to a writable build/install directory}"
: "${COMPILER_ID:?Set COMPILER_ID to identify the compiler used to build Python}"

PYTHON_VERSION=${PYTHON_VERSION:-3.14.7}
NODE_VERSION=${NODE_VERSION:-24.21.0}
TORCH_VERSION=${TORCH_VERSION:-2.13.0}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.28.0}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}
TORCH_VARIANT=${TORCH_VARIANT:-cpu}
TRIANGLE_VERSION=${TRIANGLE_VERSION:-v20250106}
PROFESSOR_REF=${PROFESSOR_REF:-feature/sherman/electronApp}
BUILD_JOBS=${BUILD_JOBS:-32}
PYTHON_PREFIX="${ELECTRON_BUILD_ROOT}/python-${PYTHON_VERSION}-${COMPILER_ID}"
PYTHON_EXE="${PYTHON_PREFIX}/bin/python3"
TORCH_KEY=$(printf '%s\n%s\n%s' "$TORCH_INDEX_URL" "$TORCH_VERSION" "$TORCHVISION_VERSION" | sha256sum | cut -c1-12)
VENV_DIR="${ELECTRON_BUILD_ROOT}/venv-python-${PYTHON_VERSION}-${COMPILER_ID}-${TORCH_VARIANT}-${TORCH_KEY}"
VENV_PYTHON="${VENV_DIR}/bin/python"
NODE_DIR="${ELECTRON_BUILD_ROOT}/node-v${NODE_VERSION}-linux-x64"

# Keep source and intermediate files isolated so another run cannot collide
# with an existing clone or partially completed build.
mkdir -p "$ELECTRON_BUILD_ROOT"
BUILD_DIR=$(mktemp -d "${ELECTRON_BUILD_ROOT}/build.${COMPILER_ID}.${TORCH_VARIANT}.XXXXXXXX")
printf 'Build workspace: %s\n' "$BUILD_DIR"
printf 'Install root: %s\n' "$ELECTRON_BUILD_ROOT"
printf 'Compiler variant: %s; Torch variant: %s\n' "$COMPILER_ID" "$TORCH_VARIANT"

# Download Node.js once and reuse the versioned installation on later runs.
if [[ ! -x "$NODE_DIR/bin/node" ]]; then
  echo "Setting up Node.js ${NODE_VERSION}"
  NODE_ARCHIVE="${BUILD_DIR}/node-v${NODE_VERSION}-linux-x64.tar.xz"
  wget -O "$NODE_ARCHIVE" "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz"
  tar -xf "$NODE_ARCHIVE" -C "$ELECTRON_BUILD_ROOT"
fi
export PATH="${NODE_DIR}/bin:${PATH}"

# Compile Python once per compiler and share that base interpreter only between
# Torch environments built with the same compiler.
if [[ ! -x "$PYTHON_EXE" ]]; then
  echo "Setting up Python ${PYTHON_VERSION} with ${COMPILER_ID}"
  PYTHON_ARCHIVE="${BUILD_DIR}/Python-${PYTHON_VERSION}.tgz"
  wget -O "$PYTHON_ARCHIVE" "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz"
  tar -zxf "$PYTHON_ARCHIVE" -C "$BUILD_DIR"
  cd "${BUILD_DIR}/Python-${PYTHON_VERSION}"

  ./configure \
    --prefix="$PYTHON_PREFIX" \
    --enable-shared \
    --with-lto \
    --enable-optimizations \
    LDFLAGS="-Wl,-rpath=${PYTHON_PREFIX}/lib"
  make -j"$BUILD_JOBS"
  make install
fi

# Keep installed packages separate for each compiler/Torch combination.
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Creating Python environment for ${COMPILER_ID}/${TORCH_VARIANT}"
  "$PYTHON_EXE" -m venv "$VENV_DIR"
fi

echo "Installing Python prerequisites from ${TORCH_INDEX_URL}"
"$VENV_PYTHON" -m pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  --index-url "$TORCH_INDEX_URL"
"$VENV_PYTHON" -m pip install pyinstaller
"$VENV_PYTHON" -m pip install "git+https://github.com/drufat/triangle.git@${TRIANGLE_VERSION}"

# Use a fresh shallow checkout for each build. The ref can be a branch, tag, or
# commit and can be overridden when submitting the job.
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
"$VENV_PYTHON" src/professor/_pyinstaller/patch_tk_scaling.py dist/prof-dash-gui

# Build the Electron app from the locked JavaScript dependency set.
echo "Building Electron app"
cd "${PROFESSOR_DIR}/electron"
npm ci
npm run package

echo "Done! Electron package: ${PROFESSOR_DIR}/electron/release"
