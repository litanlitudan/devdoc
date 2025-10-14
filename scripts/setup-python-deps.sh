#!/usr/bin/env bash
# Setup Python dependencies for ONNX and MLIR support
# This script replaces the need for requirements.txt in the root directory

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🐍 Checking Python dependencies..."

# Find Python executable
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}❌ Python not found${NC}"
    echo "Please install Python 3.9+ to use ONNX and MLIR features"
    exit 1
fi

echo "  ✓ Found Python: $($PYTHON_CMD --version)"

# Check if pip is available
if ! $PYTHON_CMD -m pip --version &> /dev/null; then
    echo -e "${RED}❌ pip not found${NC}"
    echo "Please install pip to manage Python dependencies"
    exit 1
fi

# Function to check if a package is installed
check_package() {
    local package=$1
    if $PYTHON_CMD -c "import $2" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to install a package
install_package() {
    local package=$1
    local import_name=$2

    echo -e "${YELLOW}  📦 Installing $package...${NC}"

    # Check if we're in a conda environment
    if [ -n "$CONDA_DEFAULT_ENV" ]; then
        echo "  ℹ️  Conda environment detected: $CONDA_DEFAULT_ENV"
        echo "  Using pip within conda environment"
    fi

    if $PYTHON_CMD -m pip install "$package" --quiet; then
        echo -e "${GREEN}  ✓ Successfully installed $package${NC}"
    else
        echo -e "${RED}  ❌ Failed to install $package${NC}"
        echo "  Try manually: $PYTHON_CMD -m pip install $package"
        exit 1
    fi
}

# Define packages to check/install
# Format: "package-spec|import-name"
PACKAGES=(
    "onnx>=1.12.0|onnx"
    "ai-edge-model-explorer-adapter>=0.1.13|ai_edge_model_explorer_adapter"
)

# Check and install packages
INSTALLED_COUNT=0
TOTAL_COUNT=${#PACKAGES[@]}

for pkg_info in "${PACKAGES[@]}"; do
    IFS='|' read -r package import_name <<< "$pkg_info"

    if check_package "$package" "$import_name"; then
        echo "  ✓ $import_name is already installed"
        INSTALLED_COUNT=$((INSTALLED_COUNT + 1))
    else
        echo "  ⚠️  $import_name not found"
        install_package "$package" "$import_name"
        INSTALLED_COUNT=$((INSTALLED_COUNT + 1))
    fi
done

echo ""
echo -e "${GREEN}✅ All Python dependencies ready ($INSTALLED_COUNT/$TOTAL_COUNT)${NC}"
echo ""
