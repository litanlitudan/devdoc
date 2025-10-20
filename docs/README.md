# Devdoc Documentation

Welcome to the devdoc documentation directory! This contains detailed guides and references for using devdoc.

## Quick Links

### Getting Started

- 📖 [Main README](../README.md) - Project overview and basic usage
- 🤖 [CLAUDE.md](../CLAUDE.md) - Guidelines for Claude Code when working with this codebase
- 📋 [Makefile](../Makefile) - Build automation and development commands

### Development & Integration

- 🚀 [dev3000 MCP Integration Guide](dev3000-mcp-integration.md) - **NEW!** Comprehensive guide to dev3000 integration
  - Quick start instructions
  - Development modes (standard, debug, servers-only, tail)
  - MCP server features and endpoints
  - Browser automation and coordination
  - Troubleshooting and configuration
  - Claude Code integration examples

- 📝 [dev3000 Integration Changelog](CHANGELOG-dev3000.md) - What changed in the dev3000 integration
  - New make targets
  - Technical details
  - Migration guide
  - Benefits for developers and AI assistance

## Development Workflow

### Standard Development

```bash
# Start with full MCP integration
make dev

# Access the app
open http://localhost:8642

# Access MCP logs
open http://localhost:3684/logs?project=devdoc
```

### Debug Mode

```bash
# Start with debug logging
make dev-debug
```

### Headless Mode

```bash
# Run servers without browser (for CI/CD)
make dev-no-browser
```

### Log Monitoring

```bash
# Start with log tailing
make dev-tail
```

## Key Features

### Browser Monitoring (via dev3000)

- ✅ Real-time error detection
- ✅ Performance metrics and CLS tracking
- ✅ Console log capture
- ✅ Visual diff analysis
- ✅ Network monitoring

### MCP Server Integration

- ✅ Automated error analysis with `fix_my_app`
- ✅ Performance debugging with `fix_my_jank`
- ✅ Browser automation with chrome-devtools coordination
- ✅ Smart routing for optimal tool selection

### Development Features

- ✅ Live reload with file watching
- ✅ Markdown rendering with GitHub styling
- ✅ MLIR visualization with Model Explorer
- ✅ Directory indexing with Material Design icons
- ✅ Multiple development modes

## Architecture

```
devdoc/
├── src/                      # TypeScript source code
│   ├── cli/                  # CLI commands and entry points
│   ├── server/               # Express server and middleware
│   └── mlir/                 # MLIR parsing (C++ and TypeScript)
├── lib/                      # Legacy JavaScript files
├── docs/                     # Documentation (you are here!)
│   ├── dev3000-mcp-integration.md
│   └── CHANGELOG-dev3000.md
├── tests/                    # Test files
├── Makefile                  # Build automation
└── package.json              # Dependencies and scripts
```

## Contributing

When working on this project:

1. **Read the guides**: Start with the relevant documentation
2. **Use make targets**: Leverage the Makefile for consistent workflows
3. **Enable dev3000**: Use `make dev` for the best development experience
4. **Check MCP logs**: Monitor `http://localhost:3684/logs` for errors
5. **Follow conventions**: See [CLAUDE.md](../CLAUDE.md) for coding guidelines

## Troubleshooting

### Port Conflicts

```bash
make port-check  # Check what's using port 8642
make port-kill   # Kill processes on port 8642
```

### MCP Server Issues

```bash
npx dev3000 --kill-mcp  # Kill MCP server on port 3684
make dev                # Restart with clean state
```

### Build Issues

```bash
make clean       # Remove dist/ directory
make build       # Rebuild project
```

### Test Failures

```bash
make test        # Run all tests
make cover       # Run tests with coverage
```

## Additional Resources

- [dev3000 GitHub](https://github.com/cline/dev3000)
- [Model Context Protocol](https://modelcontextprotocol.io)
- [Chrome DevTools Protocol](https://chromedevtools.github.io/devtools-protocol/)
- [Model Explorer](https://github.com/google-ai-edge/model-explorer)

## Recent Updates

### 2025-10-20: Enhanced dev3000 Integration

- ✨ New `make dev` with explicit script passing
- ✨ Three new development modes (debug, servers-only, tail)
- ✨ Comprehensive integration guide
- ✨ Architecture and flow diagrams
- ✨ Troubleshooting documentation

See [CHANGELOG-dev3000.md](CHANGELOG-dev3000.md) for details.

---

**Need help?** Check the troubleshooting sections in each guide, or review the main [CLAUDE.md](../CLAUDE.md) for comprehensive development instructions.
