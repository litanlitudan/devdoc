#!/usr/bin/env bash
# Build ai-edge-model-explorer-adapter from source
# This script compiles the native C++ MLIR parser to avoid GLIBC compatibility issues

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building Model Explorer Adapter from source...${NC}"

# Check for Bazel
if ! command -v bazel &> /dev/null; then
    echo -e "${RED}Error: Bazel is not installed.${NC}"
    echo "Please install Bazel: https://bazel.build/install"
    echo ""
    echo "On macOS: brew install bazelisk"
    echo "On Ubuntu: sudo apt install bazel"
    exit 1
fi

# Check for conda environment and use its Python
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    # Conda environment is active
    PYTHON_CMD="python"
    echo -e "${GREEN}Detected conda environment: ${CONDA_DEFAULT_ENV}${NC}"
else
    # Use system python3
    PYTHON_CMD="python3"
fi

if ! command -v $PYTHON_CMD &> /dev/null; then
    echo -e "${RED}Error: Python is not installed.${NC}"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
PYTHON_PATH=$(which $PYTHON_CMD)
echo -e "${GREEN}Using Python ${PYTHON_VERSION} at ${PYTHON_PATH}${NC}"

# Navigate to the adapter source directory
ADAPTER_DIR="third_party/model-explorer/src/builtin-adapter"
if [ ! -d "$ADAPTER_DIR" ]; then
    echo -e "${RED}Error: Adapter source directory not found at ${ADAPTER_DIR}${NC}"
    exit 1
fi

cd "$ADAPTER_DIR"

# Determine package version (use latest if not specified)
PACKAGE_VERSION="${1:-0.1.16}"
echo -e "${YELLOW}Building version ${PACKAGE_VERSION}...${NC}"

# Run the build script
BUILD_SCRIPT="python/pip_package/build_pip_package.sh"
if [ ! -f "$BUILD_SCRIPT" ]; then
    echo -e "${RED}Error: Build script not found at ${BUILD_SCRIPT}${NC}"
    exit 1
fi

echo -e "${GREEN}Running Bazel build (this may take several minutes)...${NC}"
bash "$BUILD_SCRIPT" "$PACKAGE_VERSION"

# Find the generated wheel
WHEEL_FILE=$(find gen/adapter_pip/dist -name "*.whl" | head -n 1)
if [ -z "$WHEEL_FILE" ]; then
    echo -e "${RED}Error: Build failed - no wheel file generated${NC}"
    exit 1
fi

echo -e "${GREEN}Build successful! Wheel file: ${WHEEL_FILE}${NC}"

# Install the wheel
echo -e "${YELLOW}Installing the built wheel...${NC}"
$PYTHON_CMD -m pip install --force-reinstall "$WHEEL_FILE"

echo -e "${GREEN}✓ Model Explorer Adapter built and installed successfully!${NC}"
echo ""
echo "The adapter was built natively for your system, avoiding GLIBC compatibility issues."
