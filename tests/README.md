# Test Directory Structure

This directory contains all tests and test fixtures for the markserv project, organized for clarity and maintainability.

## Directory Layout

```
tests/
├── unit/                      # Unit test files (*.test.js)
│   ├── api.test.js
│   ├── copy-button.test.js
│   ├── diff.test.js
│   ├── download.test.js
│   ├── log.test.js
│   ├── mlir-tensor-shapes.test.js
│   └── server.test.modern.js
│
└── fixtures/                  # Test fixtures and sample files
    ├── markdown/              # Markdown test files
    │   ├── example.md
    │   ├── tables.md
    │   ├── toc.md
    │   ├── links.md
    │   ├── emojis.md
    │   ├── mathjax.md
    │   ├── mlir-test.md
    │   ├── test-mlir.md
    │   └── ...
    │
    ├── mlir/                  # MLIR test files
    │   ├── test.mlir
    │   ├── example.mlir
    │   ├── stablehlo_sin.mlir
    │   ├── stablehlo_matmul.mlir
    │   └── stablehlo_complex.mlir
    │
    ├── onnx/                  # ONNX model files
    │   └── bigbird_Opset17.onnx
    │
    ├── expected/              # Expected HTML output files
    │   ├── *.expected.html
    │   └── *.render-fixture.html
    │
    ├── templates/             # Template test files
    │   ├── header.md
    │   ├── footer.md
    │   ├── article.md
    │   ├── head.html
    │   ├── index.html
    │   └── subdir/
    │       └── sub-article.md
    │
    ├── directories/           # Directory structure fixtures
    │   ├── testdir/
    │   ├── markserv-cli-readme/
    │   ├── 中文测试/
    │   └── subdir/
    │
    └── [other fixtures]       # Other test data files
        ├── example.js
        ├── sample.json
        ├── sample-data.json
        ├── test.diff
        ├── test.log
        ├── test-log-filter.log
        ├── Dockerfile*
        ├── Makefile
        └── *.implanted-fixture.*
```

## Running Tests

```bash
# Run all tests
npm test

# Run tests with coverage
npm run cover

# Run a specific test file
npx ava tests/unit/api.test.js

# Run tests matching a pattern
npx ava tests/unit/*.test.js --match "*markdown*"
```

## Test Categories

### Unit Tests (`unit/`)

All unit test files use the AVA framework and follow the naming convention `*.test.js`. Each test file focuses on a specific feature or module:

- **api.test.js** - Tests for the API route functionality
- **copy-button.test.js** - Tests for the copy button feature
- **diff.test.js** - Tests for diff file rendering
- **download.test.js** - Tests for download functionality
- **log.test.js** - Tests for log file rendering
- **mlir-tensor-shapes.test.js** - Tests for MLIR tensor shape inference
- **server.test.modern.js** - Modernized server tests

### Test Fixtures (`fixtures/`)

#### Markdown Fixtures (`fixtures/markdown/`)
Sample markdown files used to test various rendering features:
- Basic markdown rendering
- Tables, TOC, links, emojis
- MathJax equations
- MLIR code blocks
- Internationalization (中文测试)

#### MLIR Fixtures (`fixtures/mlir/`)
MLIR (Multi-Level Intermediate Representation) test files:
- StableHLO dialect examples
- Various operation types (sin, matmul, complex operations)
- Used for testing MLIR graph visualization

#### ONNX Fixtures (`fixtures/onnx/`)
ONNX model files for testing model visualization:
- bigbird_Opset17.onnx - Sample ONNX model

#### Expected Output (`fixtures/expected/`)
HTML files containing expected rendering output for comparison:
- `*.expected.html` - Expected HTML output for specific features
- `*.render-fixture.html` - Rendering test fixtures

#### Templates (`fixtures/templates/`)
Template files for testing the implant system:
- Header, footer, and article templates
- Nested template structures
- HTML and markdown templates

#### Directories (`fixtures/directories/`)
Directory structures for testing directory indexing and navigation:
- Simple test directories
- README file discovery
- International character handling

#### Other Fixtures
Miscellaneous test data files:
- JSON data samples
- Diff files
- Log files for testing log rendering
- Docker configuration files
- Implant fixtures for testing template inclusion

## Writing New Tests

### Adding a New Unit Test

1. Create a new file in `tests/unit/` with the `.test.js` extension
2. Import necessary dependencies and the server module:
   ```javascript
   import {describe, it, expect, beforeAll, afterAll} from 'vitest'
   import serverModule from '../../dist/server.js'
   import path from 'node:path'
   import {fileURLToPath} from 'node:url'

   const {init} = serverModule
   const __filename = fileURLToPath(import.meta.url)
   const __dirname = path.dirname(__filename)
   ```
3. Use `path.join(__dirname, '..', '..')` to reference the project root
4. Use `path.join(__dirname, '..', 'fixtures', ...)` to reference test fixtures

### Adding New Fixtures

1. **Markdown files**: Place in `fixtures/markdown/`
2. **MLIR files**: Place in `fixtures/mlir/`
3. **Expected HTML**: Place in `fixtures/expected/`
4. **ONNX models**: Place in `fixtures/onnx/`
5. **Templates**: Place in `fixtures/templates/`
6. **Directory structures**: Place in `fixtures/directories/`
7. **Other data**: Place directly in `fixtures/`

## Path Conventions

### In Test Files

- **Project root**: `path.join(__dirname, '..', '..')`
- **Tests directory**: `path.join(__dirname, '..')`
- **Fixtures**: `path.join(__dirname, '..', 'fixtures')`
- **Specific fixture type**: `path.join(__dirname, '..', 'fixtures', 'markdown')`

### Examples

```javascript
// Reference a markdown fixture
const mdPath = path.join(__dirname, '..', 'fixtures', 'markdown', 'example.md')

// Reference an MLIR fixture
const mlirPath = path.join(__dirname, '..', 'fixtures', 'mlir', 'test.mlir')

// Reference expected HTML
const expectedPath = path.join(__dirname, '..', 'fixtures', 'expected', 'example.expected.html')

// Reference a directory fixture
const dirPath = path.join(__dirname, '..', 'fixtures', 'directories', 'testdir')
```

## Notes

- All test files should be self-contained and not depend on test execution order
- Use `getPort()` to get an available port for server tests
- Clean up resources (close servers, delete temp files) in `afterAll` hooks
- Keep fixtures minimal and focused on specific test scenarios
- Document any complex test setups or unusual fixture requirements
