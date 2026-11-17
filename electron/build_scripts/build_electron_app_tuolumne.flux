#!/bin/bash
#flux: --job-name=build_electron
#flux: --output='electron_build_log.out'
#flux: --error='electron_build_log.err'
#flux: -N 1
#flux: -n 32
#flux: -t 45m
#flux: -B geophys
#flux: -q pdebug

set -Eeuo pipefail
trap 'status=$?; printf "Build failed with status %s at line %s: %s\n" "$status" "$LINENO" "$BASH_COMMAND" >&2' ERR

module load rocm/7.2.1
module load gcc/13.3.1-magic

PYTHON_VERSION=3.14.7
NODE_VERSION=24.21.0
TRIANGLE_VERSION=v20250106
PROFESSOR_REF=${PROFESSOR_REF:-feature/sherman/electronApp}
USER_NAME=${USER:-$(id -un)}
TARGET_DIR="/p/lustre5/${USER_NAME}/professor_electron"
PYTHON_PREFIX="$TARGET_DIR/python-${PYTHON_VERSION}"
PYTHON_EXE="$PYTHON_PREFIX/bin/python3"
NODE_DIR="$TARGET_DIR/node-v${NODE_VERSION}-linux-x64"

# Keep source and intermediate files isolated so another run cannot collide
# with an existing clone or partially completed build.
mkdir -p "$TARGET_DIR"
BUILD_DIR=$(mktemp -d "$TARGET_DIR/build.XXXXXXXX")
printf 'Build workspace: %s\n' "$BUILD_DIR"

# Download Node.js once and reuse the versioned installation on later runs.
if [[ ! -x "$NODE_DIR/bin/node" ]]; then
  echo "Setting up Node.js ${NODE_VERSION}"
  NODE_ARCHIVE="$BUILD_DIR/node-v${NODE_VERSION}-linux-x64.tar.xz"
  wget -O "$NODE_ARCHIVE" "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz"
  tar -xf "$NODE_ARCHIVE" -C "$TARGET_DIR"
fi
export PATH="$NODE_DIR/bin:$PATH"

# Compile Python once and reuse the versioned installation on later runs.
if [[ ! -x "$PYTHON_EXE" ]]; then
  echo "Setting up Python ${PYTHON_VERSION}"
  PYTHON_ARCHIVE="$BUILD_DIR/Python-${PYTHON_VERSION}.tgz"
  wget -O "$PYTHON_ARCHIVE" "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz"
  tar -zxf "$PYTHON_ARCHIVE" -C "$BUILD_DIR"
  cd "$BUILD_DIR/Python-${PYTHON_VERSION}"

  ./configure \
    --prefix="$PYTHON_PREFIX" \
    --enable-shared \
    --with-lto \
    --enable-optimizations \
    LDFLAGS="-Wl,-rpath=${PYTHON_PREFIX}/lib"
  make -j32
  make install
fi

echo "Installing Python prerequisites"
"$PYTHON_EXE" -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/rocm7.2
"$PYTHON_EXE" -m pip install pyinstaller
"$PYTHON_EXE" -m pip install "https://github.com/drufat/triangle/archive/refs/tags/${TRIANGLE_VERSION}.tar.gz"

# Use a fresh shallow checkout for each build. The ref can be a branch, tag, or
# commit and can be overridden when submitting the job.
echo "Building professor executable from ${PROFESSOR_REF}"
PROFESSOR_DIR="$BUILD_DIR/professor"
git init "$PROFESSOR_DIR"
git -C "$PROFESSOR_DIR" remote add origin https://github.com/llnl/professor.git
git -C "$PROFESSOR_DIR" fetch --depth 1 origin "$PROFESSOR_REF"
git -C "$PROFESSOR_DIR" checkout --detach FETCH_HEAD
cd "$PROFESSOR_DIR"
git rev-parse HEAD
"$PYTHON_EXE" -m pip install '.[all,dash]'
"$PYTHON_EXE" -m PyInstaller --clean --noconfirm src/professor/_pyinstaller/prof-dash-gui.spec

# Build the Electron app from the locked JavaScript dependency set.
echo "Building Electron app"
cd "$PROFESSOR_DIR/electron"
npm ci
npm run package

echo "Done! Electron package: $PROFESSOR_DIR/electron/release"
