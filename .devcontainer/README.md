# Devdoc DevContainer Configuration

This devcontainer provides a complete development environment for the devdoc project with all necessary dependencies pre-installed.

## Features

### Included by Default

- **Ubuntu 24.04 LTS** - Stable base operating system
- **Node.js 20 LTS** - JavaScript/TypeScript runtime (from NodeSource)
- **Python 3** with ONNX - For ONNX shape inference support
- **LLVM 18 & MLIR** - Full LLVM/MLIR development toolchain
  - `llvm-18`, `llvm-18-dev`, `llvm-18-tools`
  - `libmlir-18-dev`, `mlir-18-tools`
  - `clang-18`, `lld-18`
- **Build Tools** - GCC, CMake, Ninja, pkg-config
- **JSON Library** - nlohmann-json3-dev for C++ JSON support
- **Development Tools** - Git, curl, wget, vim, jq, sudo
- **npm Global Packages** - TypeScript, ts-node, nodemon, npm-check-updates
- **Non-root User** - Development runs as `dev` with sudo access
- **Pre-configured Environment**:
  - `LLVM_DIR=/usr/lib/llvm-18`
  - `CMAKE_PREFIX_PATH=/usr/lib/llvm-18`
  - LLVM tools in PATH with update-alternatives

## Getting Started

### Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop)
- [VS Code](https://code.visualstudio.com/)
- [Dev Containers Extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

### Opening in DevContainer

1. Open VS Code in the devdoc project directory
2. Press `F1` or `Cmd+Shift+P` (Mac) / `Ctrl+Shift+P` (Windows/Linux)
3. Select "Dev Containers: Reopen in Container"
4. Wait for the container to build and start

### First Time Setup

After the container starts, dependencies will be automatically installed via `npm install`.

### Verifying LLVM/MLIR Installation

After the container is ready, verify the LLVM/MLIR setup:

```bash
# Check LLVM version
llvm-config --version
# Should output: 18.x.x

# Check MLIR tools
which mlir-opt
# Should output: /usr/lib/llvm-18/bin/mlir-opt

# Verify environment variables
echo $LLVM_DIR
# Should output: /usr/lib/llvm-18

echo $CMAKE_PREFIX_PATH
# Should output: /usr/lib/llvm-18
```

### Development Workflow

```bash
# Start development server with MCP integration
make dev

# Run tests
npm test

# Run tests with coverage
npm run cover

# Build the project
npm run build

# Lint code
npx xo
```

## Port Forwarding

The following ports are automatically forwarded:

- **8642** - Devdoc App Server
- **3684** - dev3000 MCP Server
- **35729** - LiveReload Server

## Building the C++ MLIR Parser

The devcontainer includes LLVM 18 and MLIR, so you can build the high-performance C++ MLIR parser immediately:

### Quick Build (Recommended)

Use the provided build script:

```bash
.devcontainer/build-mlir-parser.sh
```

This script will:

- Verify LLVM/MLIR installation
- Configure CMake with optimal settings
- Build using Ninja for fast compilation
- Report build status and location

### Manual Build

Or build manually:

```bash
cd src/mlir
mkdir -p build && cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++ ..
ninja
```

The environment is pre-configured with:

- `CMAKE_PREFIX_PATH=/usr/lib/llvm-18` (automatically finds LLVM/MLIR)
- All required dependencies (nlohmann-json, MLIR headers, etc.)
- Clang 18 as the default C++ compiler

### Automatic Build on Container Start

To automatically build the MLIR parser when the container is created, edit `.devcontainer/devcontainer.json`:

```json
"postCreateCommand": "npm install && .devcontainer/build-mlir-parser.sh"
```

### Testing the Parser

```bash
# After building, test with a sample MLIR file
./mlir_parser ../../tests/sample.mlir
```

### Fallback Parser

If you skip the C++ build, the TypeScript regex parser in `lib/mlir-regex-parser.ts` is automatically used as a fallback.

See `src/mlir/BUILD.md` for detailed build instructions and troubleshooting.

## Customization

### Upgrading LLVM/MLIR Version

The container uses LLVM 18. To upgrade to a newer version:

1. Edit `.devcontainer/Dockerfile`
2. Replace all `llvm-18`, `mlir-18`, `clang-18` with the new version
3. Update `ENV LLVM_DIR=/usr/lib/llvm-XX` in Dockerfile
4. Update `CMAKE_PREFIX_PATH` in devcontainer.json
5. Rebuild the container

### Adding VS Code Extensions

Edit `.devcontainer/devcontainer.json` and add extension IDs to the `extensions` array.

### Modifying System Dependencies

Edit `.devcontainer/Dockerfile` and add packages to the `apt-get install` command.

## Troubleshooting

**For detailed troubleshooting, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md)**

### Container Build Issues

If the container fails to build:

1. **Check Docker is running**

   ```bash
   docker ps
   ```

2. **User creation errors (UID/GID conflicts)**

   If you see errors like "exit code: 4" during build, this is typically a UID/GID conflict. The Dockerfile is configured to handle existing users gracefully. Try:

   ```bash
   # Rebuild without cache
   Dev Containers: Rebuild Container Without Cache
   ```

3. **Check Docker disk space**

   ```bash
   docker system df
   ```

4. **Prune unused images**
   ```bash
   docker system prune -a
   ```

### Permission Issues

The container runs as the `dev` user by default. If you need root access:

1. Open a terminal in VS Code
2. Run: `sudo -i` (passwordless sudo is configured)
3. Or run individual commands with `sudo`

### Port Conflicts

If ports are already in use:

1. Stop conflicting services on your host machine
2. Or modify port mappings in `.devcontainer/devcontainer.json`

## References

- [Dev Containers Documentation](https://code.visualstudio.com/docs/devcontainers/containers)
- [devcontainer.json Reference](https://containers.dev/implementors/json_reference/)
- [Devdoc Documentation](../README.md)
