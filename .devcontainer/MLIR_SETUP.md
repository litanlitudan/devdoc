# MLIR/LLVM DevContainer Setup Guide

This document describes the MLIR/LLVM development environment in the devcontainer.

## What's Included

### LLVM 18 Toolchain

- **LLVM Core**: `llvm-18`, `llvm-18-dev`, `llvm-18-tools`
- **MLIR Libraries**: `libmlir-18-dev`, `mlir-18-tools`
- **Compilers**: `clang-18`, `lld-18`
- **Build Tools**: CMake, Ninja, pkg-config
- **JSON Library**: nlohmann-json3-dev (required for MLIR parser)

### Pre-configured Environment

```bash
LLVM_DIR=/usr/lib/llvm-18
CMAKE_PREFIX_PATH=/usr/lib/llvm-18
PATH includes /usr/lib/llvm-18/bin
```

### Tool Alternatives

The container configures `update-alternatives` for:

- `clang` → `clang-18`
- `clang++` → `clang++-18`
- `llvm-config` → `llvm-config-18`

This means you can use `clang` directly without specifying the version.

## Quick Start

### 1. Open in DevContainer

```bash
# Open VS Code in project directory
code .

# Press F1, then select:
Dev Containers: Reopen in Container
```

### 2. Verify Installation

```bash
# Check LLVM version
llvm-config --version

# Check MLIR tools
which mlir-opt
mlir-opt --version

# Check Clang
clang --version
```

### 3. Build MLIR Parser

```bash
# Use the build script (recommended)
.devcontainer/build-mlir-parser.sh

# Or build manually
cd src/mlir
mkdir -p build && cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..
ninja
```

## Directory Structure

```
devdoc/
├── .devcontainer/
│   ├── devcontainer.json       # VS Code devcontainer config
│   ├── Dockerfile              # Container image definition
│   ├── build-mlir-parser.sh    # MLIR parser build script
│   ├── README.md               # General devcontainer docs
│   └── MLIR_SETUP.md          # This file
├── src/mlir/                   # MLIR parser C++ source
│   ├── mlir_parser.cpp         # Main parser implementation
│   ├── CMakeLists.txt          # CMake configuration
│   ├── BUILD.md                # Detailed build instructions
│   └── build/                  # Build output (created by cmake)
└── lib/
    └── mlir-regex-parser.ts    # TypeScript fallback parser
```

## Build Configuration

### CMake Options

The environment automatically provides:

```cmake
CMAKE_PREFIX_PATH=/usr/lib/llvm-18  # Finds LLVM/MLIR
CMAKE_CXX_COMPILER=clang++          # Uses Clang 18
CMAKE_BUILD_TYPE=Release            # Optimized build
```

### Build with Different Options

```bash
cd src/mlir/build

# Debug build
cmake -DCMAKE_BUILD_TYPE=Debug ..
ninja

# Verbose build
cmake -DCMAKE_VERBOSE_MAKEFILE=ON ..
ninja

# Clean build
ninja clean
cmake ..
ninja
```

## Testing the Parser

### Basic Test

```bash
cd src/mlir/build
./mlir_parser ../../tests/sample.mlir
```

### Integration Test

```bash
# Run from project root
npm test
```

The test suite automatically detects the C++ parser if available.

## Troubleshooting

### LLVM Not Found

If CMake can't find LLVM:

```bash
# Check environment
echo $CMAKE_PREFIX_PATH
echo $LLVM_DIR

# Verify LLVM installation
llvm-config --version
llvm-config --cmakedir

# Set explicitly in CMake
cmake -DCMAKE_PREFIX_PATH=/usr/lib/llvm-18 ..
```

### Build Errors

Common issues:

1. **Missing nlohmann-json**

   ```bash
   sudo apt-get update
   sudo apt-get install nlohmann-json3-dev
   ```

2. **Wrong compiler**

   ```bash
   cmake -DCMAKE_CXX_COMPILER=clang++-18 ..
   ```

3. **Clean build**
   ```bash
   cd src/mlir
   rm -rf build
   mkdir build && cd build
   cmake -G Ninja ..
   ninja
   ```

### Parser Not Detected

The Node.js code looks for the parser at `src/mlir/build/mlir_parser`.

Verify it exists:

```bash
ls -la src/mlir/build/mlir_parser
```

If not found, the TypeScript fallback parser will be used automatically.

## Performance Comparison

### C++ MLIR Parser (with LLVM context)

- ✅ Full MLIR IR verification
- ✅ Proper dialect registration
- ✅ 10-100x faster than regex parser
- ✅ Accurate AST traversal
- ⚡ Recommended for production use

### TypeScript Regex Parser (fallback)

- ✅ No build required
- ✅ Works without LLVM
- ✅ Handles all MLIR dialects
- ⚠️ Best-effort text parsing
- 📊 Good for development/testing

## Additional Resources

- **Build Instructions**: `src/mlir/BUILD.md`
- **DevContainer README**: `.devcontainer/README.md`
- **MLIR Documentation**: https://mlir.llvm.org/
- **LLVM Documentation**: https://llvm.org/docs/

## Upgrading LLVM Version

To use a different LLVM version (e.g., LLVM 19):

1. Edit `.devcontainer/Dockerfile`:

   ```dockerfile
   # Change all instances of llvm-18 to llvm-19
   llvm-19 \
   llvm-19-dev \
   # ... etc
   ```

2. Update environment:

   ```dockerfile
   ENV LLVM_DIR=/usr/lib/llvm-19
   ENV CMAKE_PREFIX_PATH=/usr/lib/llvm-19
   ```

3. Update update-alternatives:

   ```dockerfile
   RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-19 100 \
       # ... etc
   ```

4. Update `.devcontainer/devcontainer.json`:

   ```json
   "remoteEnv": {
     "LLVM_DIR": "/usr/lib/llvm-19",
     "CMAKE_PREFIX_PATH": "/usr/lib/llvm-19"
   }
   ```

5. Rebuild container: `Dev Containers: Rebuild Container`

## Contributing

When making changes to the MLIR parser:

1. Build and test locally in the devcontainer
2. Verify both C++ and TypeScript parsers work
3. Update tests if adding new features
4. Document any new build requirements

## License

Same as the main devdoc project - see LICENSE file.
