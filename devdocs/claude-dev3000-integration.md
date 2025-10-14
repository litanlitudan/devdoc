# Using dev3000 with markserv (Optional)

This document explains how to optionally integrate **dev3000** with markserv for enhanced AI-assisted debugging via Claude Code.

## Overview

**dev3000** is a development monitoring tool by Vercel that captures:

- Server logs and errors
- Browser console messages
- Network traffic
- Visual state changes and screenshots
- Exposes a Model Context Protocol (MCP) server for AI agents

**When is dev3000 useful for markserv?**

- Debugging rendering issues with markdown/HTML templates
- Investigating browser-side JavaScript errors
- Analyzing network requests for static assets
- Troubleshooting live-reload functionality

**Note:** dev3000 is most valuable for complex frontend applications. For basic markserv usage, the built-in live-reload and standard debugging tools may be sufficient.

## Quick Start

```bash
# Install dev3000 globally (one-time setup)
make install-dev3000

# Start markserv with dev3000 monitoring
make dev
```

This single command will:
- Start markserv on port 8642
- Launch your browser automatically
- Set up the MCP server for Claude Code integration
- Monitor server logs, browser console, and network traffic

**Note:** If you modify TypeScript files, you'll need to rebuild with `npm run build` before restarting `make dev`.

## Registering the MCP Server with Claude Code

Claude Code can discover the dev3000 MCP server in several ways:

### Option 1: Auto-Discovery (Easiest)

Claude Code automatically detects MCP servers running on `localhost:3684` (dev3000's default MCP port).

**Steps:**

1. Start dev3000:
   ```bash
   make dev
   ```

2. Launch Claude Code in your project:
   ```bash
   claude
   ```

3. Verify by asking Claude to check logs or browser console

### Option 2: Using Claude CLI (Recommended for Persistence)

Register the MCP server using Claude's built-in CLI command:

```bash
# Add dev3000 MCP server (HTTP transport required)
claude mcp add --transport http dev3000 http://localhost:3684

# Verify it was added
claude mcp list

# Remove if needed
claude mcp remove dev3000
```

This registers the MCP server in Claude's global configuration, so it will be available in any project.

### Option 3: Manual Configuration (Project-Specific)

Create a `.mcp.json` file in your project root:

```json
{
  "mcpServers": {
    "dev3000": {
      "url": "http://localhost:3684",
      "description": "Development monitoring and debugging"
    }
  }
}
```

**Note:** Add `.mcp.json` to `.gitignore` if you don't want to commit it.

### Verify the Connection

Once configured, test the MCP connection by asking Claude:

```text
"What's in the server logs?"
"Show me recent browser console errors"
"What network requests failed?"
```

If Claude responds with actual log data, the MCP connection is working!

### Troubleshooting

**MCP server not responding:**

```bash
# Check if MCP server is running
curl http://localhost:3684

# Check dev3000 MCP logs
tail -f ~/.d3k/logs/mcp.log

# Verify port isn't blocked
lsof -i :3684
```

**Claude can't see the MCP server:**

- Restart Claude Code after adding the MCP server
- Check that dev3000 is running with `make dev`
- Verify the MCP port (3684) isn't being used by another service

## Example Workflow: Debugging with dev3000 and Claude Code

**Scenario:** You're working on a custom Handlebars template and notice rendering issues.

**Setup:**

```bash
# Start markserv with dev3000 monitoring
make dev
```

This will:
- Start markserv on http://localhost:8642
- Launch your browser automatically
- Set up MCP server on http://localhost:3684

**Register MCP Server (One-time setup):**

```bash
claude mcp add --transport http dev3000 http://localhost:3684
```

**Debugging Workflow:**

1. **Navigate** in your browser to the problematic markdown file

2. **Launch Claude Code** in the markserv directory:
   ```bash
   claude
   ```

3. **Ask Claude to investigate:**
   ```text
   "I'm seeing a rendering issue with the Handlebars template.
   Check the browser console and server logs for errors."
   ```

4. **Claude uses dev3000's MCP** to:
   - Inspect browser console for JavaScript errors
   - Check server logs for template compilation errors
   - Analyze network requests for failed asset loads
   - Review the rendered HTML output

5. **Claude suggests fixes** based on the captured context:
   - Template syntax corrections
   - Missing variable definitions
   - Asset path issues
   - Handlebars helper problems

6. **Review and approve** Claude's proposed changes

## Best Practices for markserv + dev3000

### When to Use dev3000

**Good use cases:**

- ✅ Debugging complex template rendering issues
- ✅ Investigating browser-side JavaScript errors in custom templates
- ✅ Analyzing network performance for large markdown files with many assets
- ✅ Troubleshooting live-reload websocket connections
- ✅ Examining MLIR/Model Explorer visualization rendering

**When standard tools are sufficient:**

- ⚠️ Basic markdown rendering errors (use console logs)
- ⚠️ File serving issues (use browser DevTools)
- ⚠️ Simple syntax errors (use linting)
- ⚠️ TypeScript compilation errors (use `npm run build`)

### Tips for Effective Usage

1. **Register the MCP server once** - Use `claude mcp add --transport http dev3000 http://localhost:3684` for permanent setup across all projects
2. **Rebuild when needed** - Run `npm run build` before restarting `make dev` if you've modified TypeScript files
3. **Use the Makefile** - `make dev` is the simplest way to start everything with proper monitoring
4. **Leverage `CLAUDE.md`** - Guide Claude about markserv-specific patterns and conventions
5. **Ask specific questions** - "What's causing the 404 for this asset?" works better than "fix my app"
6. **Use dev3000 for tricky bugs** - Especially useful for hard-to-reproduce browser issues or timing-related problems

### Common dev3000 Commands

**Using Makefile (Recommended):**

```bash
# Install dev3000 globally (one-time)
make install-dev3000

# Start markserv with dev3000 monitoring
make dev
```

**Direct npx commands:**

```bash
# Start with standard TUI interface
npx dev3000 --port 8642

# With debug logging for detailed output
npx dev3000 --port 8642 --debug

# Kill existing MCP server if needed
npx dev3000 --kill-mcp
```

### Viewing Logs and Errors

**dev3000 TUI (Terminal Interface):**

When you run `make dev`, dev3000 displays a TUI showing:
- Server logs in real-time
- Browser console messages
- Network requests
- Error summaries

**dev3000 Web UI:**

Open in your browser for detailed view:
```text
http://localhost:3684/logs?project=markserv
```

Shows comprehensive logs including:
- Server-side console output
- Browser console errors and warnings
- Network requests and responses
- Screenshots and visual state

**Debug Mode:**

For maximum detail, use the `--debug` flag:
```bash
npx dev3000 --port 8642 --debug
```

This shows dev3000 internals + your app's output.

**Log Files:**

Check timestamped log files:
```bash
tail -f ~/.d3k/logs/markserv-d3k.log
```

## Additional Resources

- [dev3000 GitHub Repository](https://github.com/vercel-labs/dev3000)
- [Claude Code Best Practices](https://www.anthropic.com/engineering/claude-code-best-practices)
- [Model Context Protocol Documentation](https://modelcontextprotocol.io/)
- [markserv CLAUDE.md](../CLAUDE.md) - Project-specific Claude Code guidance
