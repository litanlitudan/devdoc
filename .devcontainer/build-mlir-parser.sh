#!/bin/bash
#
# Build the C++ MLIR parser in the devcontainer
# This script can be run manually or as part of postCreateCommand
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MLIR_SRC="$PROJECT_ROOT/src/mlir"
BUILD_DIR="$MLIR_SRC/build"

echo "========================================="
echo "Building C++ MLIR Parser"
echo "========================================="
echo ""

# Check if LLVM is available
if ! command -v llvm-config &> /dev/null; then
    echo "❌ Error: llvm-config not found in PATH"
    echo "Make sure LLVM/MLIR is installed in the container"
    exit 1
fi

echo "✓ LLVM version: $(llvm-config --version)"
echo "✓ LLVM directory: $LLVM_DIR"
echo "✓ CMake prefix path: $CMAKE_PREFIX_PATH"
echo ""

# Create build directory
echo "Creating build directory: $BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Navigate to build directory
cd "$BUILD_DIR"

# Run CMake
echo ""
echo "Running CMake configuration..."
cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=clang++ \
    ..

# Build
echo ""
echo "Building MLIR parser..."
ninja

# Check if build succeeded
if [ -f "$BUILD_DIR/mlir_parser" ]; then
    echo ""
    echo "========================================="
    echo "✓ Build successful!"
    echo "========================================="
    echo ""
    echo "MLIR parser binary: $BUILD_DIR/mlir_parser"
    echo ""
    echo "You can test it with:"
    echo "  cd src/mlir/build"
    echo "  ./mlir_parser ../../tests/sample.mlir"
    echo ""
else
    echo ""
    echo "❌ Build failed - mlir_parser binary not found"
    exit 1
fi
