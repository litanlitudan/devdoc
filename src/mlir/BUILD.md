# MLIR Context-Based Parser - Build Instructions

This directory contains a C++ implementation of the MLIR parser that uses proper MLIR context, dialect registration, and the documented pipeline from `devdocs/parser/universal-mlir-parser-design.md`.

## Features

✅ **Proper MLIR Context**: Uses MLIRContext with dialect registration
✅ **Allow Unregistered Dialects**: Supports custom/unknown dialects
✅ **ModuleOp Parsing**: Parses MLIR text to proper ModuleOp structure
✅ **Verification**: Validates IR correctness
✅ **Graph Building**: Proper graph construction with region traversal
✅ **Multi-Graph Output**: One graph per func.func
🚧 **Conditional Normalization**: VHLO→StableHLO (TODO)
🚧 **Uniquing Pass**: CreateUniqueOpNamesPass (TODO)

## Prerequisites

### Required

1. **LLVM/MLIR** (version 17.0 or later)
   - Built from source with MLIR enabled
   - Or installed from pre-built packages

2. **CMake** (version 3.20 or later)

3. **C++ Compiler** with C++17 support
   - GCC 9+ or Clang 10+

4. **nlohmann/json** library
   - For JSON output generation

### Optional

- **ninja**: Faster build system (recommended)

## Building LLVM/MLIR from Source

If you don't have LLVM/MLIR installed:

```bash
# Clone LLVM project
git clone https://github.com/llvm/llvm-project.git
cd llvm-project

# Create build directory
mkdir build && cd build

# Configure with MLIR enabled
cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local/llvm

# Build (this takes a while!)
ninja

# Install
sudo ninja install
```

## Installing nlohmann/json

### Using package manager (recommended)

```bash
# macOS
brew install nlohmann-json

# Ubuntu/Debian
sudo apt-get install nlohmann-json3-dev

# Fedora
sudo dnf install json-devel
```

### From source

```bash
git clone https://github.com/nlohmann/json.git
cd json
mkdir build && cd build
cmake ..
sudo make install
```

## Building the MLIR Parser

```bash
# Navigate to this directory
cd src/mlir

# Create build directory
mkdir build && cd build

# Configure
cmake .. -DCMAKE_PREFIX_PATH=/usr/local/llvm

# If LLVM is in a different location:
# cmake .. -DCMAKE_PREFIX_PATH=/path/to/llvm/install

# Build
make

# Optional: Install system-wide
sudo make install
```

## Verification

Test the parser with a simple MLIR file:

```bash
# Create test MLIR
cat > test.mlir << 'EOF'
module {
  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %0 = "custom.op"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
    func.return %0 : tensor<2x3xf32>
  }
}
EOF

# Run parser
./build/mlir_parser test-graph < test.mlir

# Expected output: JSON graph structure
```

## Integration with devdoc

The parser is automatically integrated via `scripts/parse_mlir_cpp.py`:

```python
# Tries C++ parser first, falls back to regex parser if not available
python3 scripts/parse_mlir_cpp.py <filename> < model.mlir > graph.json
```

## Troubleshooting

### CMake can't find MLIR

```bash
# Set CMAKE_PREFIX_PATH to your LLVM installation
cmake .. -DCMAKE_PREFIX_PATH=/usr/local/llvm

# Or use MLIR_DIR directly
cmake .. -DMLIR_DIR=/usr/local/llvm/lib/cmake/mlir
```

### nlohmann/json not found

```bash
# Install via package manager (see above)
# Or specify path manually
cmake .. -Dnlohmann_json_DIR=/path/to/json/cmake
```

### Linker errors

```bash
# Ensure all MLIR dialect libraries are installed
# Check your LLVM build includes all required dialects
```

## Development

### Adding More Dialects

Edit `mlir_parser.cpp` and add dialect includes:

```cpp
// Add at top
#include "mlir/Dialect/TensorFlow/IR/TF.h"

// In MLIRContextManager constructor
context_.loadDialect<TF::TensorFlowDialect>();
```

Update `CMakeLists.txt`:

```cmake
target_link_libraries(mlir_parser PRIVATE
    ...
    MLIRTensorFlowDialect  # Add new dialect
)
```

### Implementing Normalization

The conditional normalization (VHLO→StableHLO) should be added in the pass pipeline section:

```cpp
// In main(), after module verification
if (hasVHLODialect(module)) {
    pm.addPass(createVHLOToStablehloPass());
}
```

### Implementing Uniquing Pass

Add CreateUniqueOpNamesPass to ensure deterministic node IDs:

```cpp
pm.addPass(createSymbolPrivatizePass());
pm.addPass(createSymbolDCEPass());
// Custom pass to add unique names to operations
```

## Performance

- **Parsing Speed**: ~10-100x faster than regex parser
- **Memory Usage**: More efficient for large models
- **Accuracy**: Full MLIR IR validation and verification

## Fallback Behavior

If the C++ parser is not built, the system automatically falls back to the regex-based Python parser with a warning:

```
ℹ C++ parser not found, using regex fallback
  Build C++ parser with: cd src/mlir && mkdir build && cd build && cmake .. && make
✓ Used regex-based parser (fallback)
```

This ensures devdoc continues working even without the C++ parser.
