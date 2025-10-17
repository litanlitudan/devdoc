# Global Usage Instructions for Devdoc

After setting up devdoc globally with `npm link`, you can use it from anywhere on your system.

## Installation for Global Use

1. **From the devdoc directory**, run:
   ```bash
   npm link
   ```

2. This creates global symlinks:
   - `devdoc` command → points to `lib/cli.js`
   - `readme` command → points to `lib/readme.js`

## Global Commands

### `devdoc` Command
Use devdoc from any directory to serve markdown and static files:

```bash
# Serve current directory
devdoc

# Serve specific directory
devdoc /path/to/directory

# Serve specific file
devdoc document.md

# With options
devdoc --port 3000 --silent
devdoc -p 3000 -s

# Serve on specific address
devdoc --address 0.0.0.0 --port 8080
```

### `readme` Command
Quickly serve the nearest README.md file:

```bash
# Find and serve the nearest README.md
readme

# With custom port
readme --port 3000
```

## Command Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--port` | `-p` | HTTP port | 8642 |
| `--livereloadport` | `-b` | LiveReload port | 35729 |
| `--address` | `-a` | Bind address | localhost |
| `--silent` | `-s` | Silent mode | false |
| `--verbose` | `-v` | Verbose output | false |
| `--version` | | Show version | |
| `--help` | | Show help | |

## Examples

### Serve a Project Documentation
```bash
cd ~/my-project
devdoc docs/
```

### Serve on All Network Interfaces
```bash
devdoc --address 0.0.0.0 --port 8080
```

### Silent Mode (No Console Output)
```bash
devdoc --silent
```

### Quick README Preview
```bash
cd ~/my-project
readme
```

## MLIR Support
Devdoc now supports `.mlir` files with syntax highlighting:

```bash
# Serve directory with MLIR files
devdoc /path/to/mlir/files

# MLIR files will be rendered with proper syntax highlighting
```

## Uninstalling Global Link

To remove the global link:
```bash
npm unlink -g devdoc
```

## Troubleshooting

If `devdoc` command is not found:
1. Check npm global bin directory: `npm bin -g`
2. Ensure it's in your PATH
3. Re-run `npm link` from the devdoc directory

## Features
- 🎨 GitHub-flavored markdown rendering
- 📁 Directory listing with icons
- 🔄 LiveReload on file changes
- 🎯 MLIR file support with syntax highlighting
- 🔢 Math rendering (MathJax)
- 📊 Mermaid diagram support
- 🎨 Syntax highlighting for code blocks
- 📑 Table of contents generation
- ✅ Task list support
- 😀 Emoji support