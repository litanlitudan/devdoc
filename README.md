# Markserv Installation Guide

## Prerequisites

### Required
- Node.js (v18 or higher)
- npm (comes with Node.js)
- Git

### Optional (for ONNX & MLIR support)
- Python 3.9+ (for processing ONNX and MLIR files)
- Python packages (see [Python Dependencies](#python-dependencies))

## Installation Steps

```shell
# Clone the repository
$ git clone https://github.com/litanlitudan/markserv.git

# Navigate to the markserv directory
$ cd markserv

# Install dependencies
$ npm install

# Build the project
$ npm run build

# Link the package globally to use 'markserv' command
$ npm link
```

### Python Dependencies

To enable ONNX and MLIR file visualization, install the required Python packages:

```shell
# Install Python dependencies
$ pip install -r requirements.txt

# Or install individually:
$ pip install onnx>=1.12.0  # For ONNX shape inference
$ pip install ai-edge-model-explorer-adapter>=0.1.13  # For MLIR parsing
```

**Note**: ONNX and MLIR files will still be displayed without Python, but without enhanced graph visualization features.

## Running the Server

```shell
# Basic usage - serve current directory
$ markserv

# Serve specific directory or file
$ markserv path/to/docs

# Advanced usage with custom port and network access
$ markserv path/to/docs -p 8642 -a 0.0.0.0
```

### Command Options
- `-p 8642` - Sets the server port to 8642 (default: 8008)
- `-a 0.0.0.0` - Makes the server accessible from external networks (default: localhost)
- `-s` - Silent mode (minimal output)
- `-v` - Verbose mode (detailed output)
- `-w` - Enable file watching for live reload

## Features

### Core Features
- 📝 Renders Markdown files as HTML with GitHub-style CSS
- 📁 Beautiful directory listing with file icons
- 🎨 Syntax highlighting for code blocks
- 📊 Support for Mermaid diagrams
- 🔢 MathJax support for mathematical expressions
- 🌐 Network accessible for sharing documentation

### AI Model Visualization (with Python)
- 🤖 **ONNX Model Visualization**: View neural network graphs with shape inference
- ⚡ **MLIR Graph Visualization**: View compiler IR with tensor shape information
  - Supports StableHLO, TensorFlow Lite, and other MLIR dialects
  - Displays input/output tensor shapes in node labels
  - Uses Google's Model Explorer adapter for accurate parsing

## Common Use Cases

### Local Documentation Server
```shell
# Serve your project's documentation
$ markserv ./docs -p 3000
```

### Remote Access for Team Sharing
```shell
# Make documentation available to your network
$ markserv ./docs -p 8642 -a 0.0.0.0
```

### Keep Server Running in Background
```shell
# Run server as a background process (Unix/Linux/macOS)
$ nohup markserv ./docs -p 8642 -a 0.0.0.0 &
```

### Visualize AI Models
```shell
# Serve directory containing ONNX and MLIR files
$ markserv ./models

# Then open in browser:
# - http://localhost:8008/model.onnx - Interactive ONNX graph
# - http://localhost:8008/model.mlir - Interactive MLIR graph with shapes
```

## Troubleshooting

### Port Already in Use
If you see an error about the port being in use, try a different port:
```shell
$ markserv -p 3001
```

### External Access Not Working
Ensure your firewall allows connections on the specified port and use `-a 0.0.0.0`:
```shell
$ markserv -p 8642 -a 0.0.0.0
```

### Python Dependencies Not Found

If you see warnings about missing Python packages:

```shell
# For ONNX support:
$ pip install onnx>=1.12.0

# For MLIR support:
$ pip install ai-edge-model-explorer-adapter>=0.1.13
```

**Note**: Make sure Python 3.9+ is installed and accessible via `python3` command (or `python` in conda environments).

### MLIR Parsing Errors

If you encounter MLIR parsing errors:

1. **Check MLIR syntax**: Ensure your `.mlir` file uses valid StableHLO syntax
2. **Update Python package**: `pip install --upgrade ai-edge-model-explorer-adapter`
3. **Verify Python version**: Model Explorer adapter requires Python 3.9+
4. **Check dialect support**: Some custom MLIR dialects may not be fully supported

Example error:
```
INVALID_ARGUMENT: Failed to parse MLIR module: 'stablehlo.broadcast_in_dim'
op attribute 'broadcast_dimensions' failed to satisfy constraint
```

This indicates a syntax issue in your MLIR file. Check that attributes use the correct format (e.g., `array<i64: 1>` instead of `dense<1> : tensor<1xi64>`).

## Supported File Types

### Text & Documentation
- Markdown (`.md`, `.markdown`)
- Plain text (`.txt`)
- HTML (`.html`, `.htm`)
- Code files (with syntax highlighting)

### AI Models & IR
- **ONNX Models** (`.onnx`) - Neural network graph visualization with shape inference
- **MLIR Files** (`.mlir`) - Compiler intermediate representation with tensor shapes
  - StableHLO dialect
  - TensorFlow Lite dialect
  - Other standard MLIR dialects

### Images & Media
- Common image formats (PNG, JPEG, GIF, SVG)
- Icons and favicons

## License

Apache License 2.0

Copyright 2024-2025 Tan Li

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
