#!/bin/bash
# Build the webifc pybind11 extension module.
#
# Usage:  ./build_webifc.sh
# Output: webifc*.so in the project root (~/Documents/FreeIFC/ or repo root)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# Detect pybind11 cmake dir
PYBIND11_DIR="$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())" 2>/dev/null)" || true
if [ -z "$PYBIND11_DIR" ]; then
    # Fallback: common pip install location
    PYBIND11_DIR="$(python3 -c "import site; print(site.getusersitepackages())")/pybind11/share/cmake/pybind11"
fi

echo "pybind11 cmake dir: ${PYBIND11_DIR}"
echo "Python executable:  $(which python3)"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake "${SCRIPT_DIR}/webifc_bind" \
    -DCMAKE_BUILD_TYPE=Release \
    -Dpybind11_DIR="${PYBIND11_DIR}" \
    -DPYTHON_EXECUTABLE="$(which python3)"

NPROC=$(nproc --ignore=1 2>/dev/null || echo 4)
make -j"${NPROC}"

echo ""
echo "Built: $(ls "${SCRIPT_DIR}"/webifc*.so 2>/dev/null || echo 'webifc*.so not found — check build output')"
