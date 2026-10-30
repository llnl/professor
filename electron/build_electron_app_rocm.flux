#!/bin/bash
#flux: --job-name=build_electron
#flux: --output='electron_build_log.out'
#flux: --error='electron_build_log.err'
#flux: -N 1
#flux: -n 32
#flux: -t 45m
#flux: -B geophys
#flux: -q pdebug

module load rocm/7.2.1
module load gcc/13.3.1-magic
PYTHON_VERSION=3.14.7
TARGET_DIR=/p/lustre5/${USER}/professor_electon/

# Setup
mkdir -p $TARGET_DIR
cd $TARGET_DIR

# Download node.js and make sure that it is available on the path
echo "Setting up node.js"
wget https://nodejs.org/dist/v24.21.0/node-v24.21.0-linux-x64.tar.xz
tar -xf node-v24.21.0-linux-x64.tar.xz
export PATH=$TARGET_DIR/node-v24.21.0-linux-x64/bin:$PATH

# Download and compile python
echo "Setting up python"
wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz
tar -zxf Python-${PYTHON_VERSION}.tgz
cd Python-${PYTHON_VERSION}

./configure \
  --prefix=$TARGET_DIR/python-${PYTHON_VERSION} \
  --enable-shared \
  --with-lto \
  --enable-optimizations \
  LDFLAGS="-Wl,-rpath=$TARGET_DIR/python-${PYTHON_VERSION}/lib"
make -j32
make install

# Install key pre-requisites
# Note: one of prof's pre-requisites relies on the triangle package
# The pypi version of this package is broken, so install it from git
echo "Installing python pre-requisites"
PYTHON_EXE=$TARGET_DIR/python-${PYTHON_VERSION}/bin/python3
$PYTHON_EXE -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/rocm7.2
$PYTHON_EXE -m pip install pyinstaller
$PYTHON_EXE -m pip install git+https://github.com/drufat/triangle.git

# Download and install professor
echo "Building professor executable"
cd $TARGET_DIR
git clone --depth 1 --branch feature/sherman/electronApp https://github.com/llnl/professor.git
cd professor
$PYTHON_EXE -m pip install .[all,dash]
$PYTHON_EXE -m PyInstaller src/professor/_pyinstaller/prof-dash-gui.spec

# Build the electron app
echo "Building electron app"
cd electron
npm install
npm run package

echo "Done!"
