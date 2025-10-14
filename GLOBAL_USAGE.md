# Global Usage Instructions for Markserv

After setting up markserv globally with `npm link`, you can use it from anywhere on your system.

## Installation for Global Use

1. **From the markserv directory**, run:
   ```bash
   npm link
   ```

2. This creates global symlinks:
   - `markserv` command → points to `lib/cli.js`
   - `readme` command → points to `lib/readme.js`

## Global Commands

### `markserv` Command
Use markserv from any directory to serve markdown and static files:

```bash
# Serve current directory
markserv

# Serve specific directory
markserv /path/to/directory

# Serve specific file
markserv document.md

# With options
markserv --port 3000 --silent
markserv -p 3000 -s

# Serve on specific address
markserv --address 0.0.0.0 --port 8080
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
markserv docs/
```

### Serve on All Network Interfaces
```bash
markserv --address 0.0.0.0 --port 8080
```

### Silent Mode (No Console Output)
```bash
markserv --silent
```

### Quick README Preview
```bash
cd ~/my-project
readme
```

## MLIR Support
Markserv now supports `.mlir` files with syntax highlighting:

```bash
# Serve directory with MLIR files
markserv /path/to/mlir/files

# MLIR files will be rendered with proper syntax highlighting
```

## Uninstalling Global Link

To remove the global link:
```bash
npm unlink -g markserv
```

## Troubleshooting

If `markserv` command is not found:
1. Check npm global bin directory: `npm bin -g`
2. Ensure it's in your PATH
3. Re-run `npm link` from the markserv directory

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