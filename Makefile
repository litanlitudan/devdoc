# Makefile for devdoc - Build automation and task orchestration
#
# This Makefile provides a unified interface for all development tasks,
# enabling consistent workflows across local development and CI/CD.

.PHONY: help install build test lint clean dev dev-debug dev-no-browser dev-tail watch ci port-check port-kill mcp-kill typecheck format
.PHONY: install-dev3000 setup-python build-adapter clean-all test-watch test-ui start cover

# Default target - show help
.DEFAULT_GOAL := help

# ============================================================================
# Help
# ============================================================================

help:
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  devdoc - Build Automation Targets"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "  📦 Installation & Setup:"
	@echo "    make install          Install npm and Python dependencies"
	@echo "    make setup-python     Setup Python dependencies only"
	@echo "    make install-dev3000  Install dev3000 globally"
	@echo ""
	@echo "  🏗️  Build:"
	@echo "    make build            Compile TypeScript to dist/"
	@echo "    make watch            Build in watch mode for development"
	@echo ""
	@echo "  🧪 Testing:"
	@echo "    make test             Run all tests"
	@echo "    make test-watch       Run tests in watch mode"
	@echo "    make test-ui          Run tests with UI"
	@echo "    make cover            Run tests with coverage report"
	@echo ""
	@echo "  🔍 Code Quality:"
	@echo "    make lint             Run linter (with auto-fix)"
	@echo "    make format           Format code with prettier"
	@echo "    make typecheck        Type check without emitting"
	@echo ""
	@echo "  🚀 Development (dev3000 MCP Integration):"
	@echo "    make dev              Full integration: MCP server + browser monitoring"
	@echo "                          📡 MCP: localhost:3684 | 🌐 App: localhost:8642"
	@echo "    make dev-debug        Debug mode with detailed logging (TUI disabled)"
	@echo "    make dev-no-browser   Servers only mode (manual browser control)"
	@echo "    make dev-tail         Tail mode with consolidated log output"
	@echo "    make start            Start server without dev3000 (requires build)"
	@echo ""
	@echo "  🧹 Cleanup:"
	@echo "    make clean            Remove dist/ directory"
	@echo "    make clean-all        Remove all build artifacts and node_modules"
	@echo ""
	@echo "  🔧 Utilities:"
	@echo "    make port-check       Check if port 8642 is in use"
	@echo "    make port-kill        Kill process on port 8642"
	@echo "    make mcp-kill         Stop dev3000 MCP server (port 3684)"
	@echo "    make build-adapter    Build Model Explorer C++ adapter"
	@echo ""
	@echo "  🤖 CI/CD:"
	@echo "    make ci               Full CI pipeline (install→lint→typecheck→test→build)"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ============================================================================
# Installation & Setup
# ============================================================================

install:
	@echo "📦 Installing dependencies..."
	npm install
	@$(MAKE) setup-python

setup-python:
	@echo "🐍 Setting up Python dependencies..."
	npm run setup:python

install-dev3000:
	@echo "🔧 Installing dev3000 globally..."
	npm install -g dev3000

# ============================================================================
# Build
# ============================================================================

build:
	@echo "🏗️  Building project..."
	npm run build

watch:
	@echo "👀 Starting build in watch mode..."
	npm run build:watch

build-adapter:
	@echo "🔨 Building C++ adapter from source..."
	npm run build:adapter

# ============================================================================
# Testing
# ============================================================================

test: build
	@echo "🧪 Running tests..."
	npm test

test-watch:
	@echo "👀 Running tests in watch mode..."
	npm run test:watch

test-ui:
	@echo "🖥️  Running tests with UI..."
	npm run test:ui

cover: build
	@echo "📊 Running tests with coverage..."
	npm run cover

# ============================================================================
# Code Quality
# ============================================================================

lint:
	@echo "🔍 Running linter..."
	npm run lint

format:
	@echo "✨ Formatting code..."
	npm run format

typecheck:
	@echo "🔎 Type checking..."
	npm run typecheck

# ============================================================================
# Development
# ============================================================================

dev: port-kill build
	@echo "🚀 Starting development server with dev3000 MCP integration..."
	@echo ""
	@echo "📊 Endpoints:"
	@echo "  🌐 App Server:    http://localhost:8642"
	@echo "  📡 MCP Server:    http://localhost:3684"
	@echo "  📜 Logs UI:       http://localhost:3684/logs?project=devdoc"
	@echo "  💚 Health Check:  http://localhost:3684/health"
	@echo ""
	@echo "🔧 Features enabled:"
	@echo "  ✅ Browser monitoring and error detection"
	@echo "  ✅ Performance metrics and CLS tracking"
	@echo "  ✅ Chrome DevTools MCP coordination"
	@echo "  ✅ Live reload for markdown/MLIR files"
	@echo ""
	@echo "💡 Tip: Use Ctrl+C to stop all servers"
	@echo ""
	@npx dev3000 --kill-mcp 2>/dev/null || true
	npx dev3000 --port 8642 --script start

dev-debug: port-kill build
	@echo "🐛 Starting development server with debug logging..."
	@echo "📊 Debug mode: Detailed log output enabled (TUI disabled)"
	@echo ""
	@npx dev3000 --kill-mcp 2>/dev/null || true
	npx dev3000 --port 8642 --script start --debug

dev-no-browser: port-kill build
	@echo "🖥️  Starting dev3000 servers only (no browser launch)..."
	@echo "📊 MCP & App servers will start without browser"
	@echo "💡 Navigate to http://localhost:8642 manually"
	@echo ""
	@npx dev3000 --kill-mcp 2>/dev/null || true
	npx dev3000 --port 8642 --script start --servers-only

dev-tail: port-kill build
	@echo "📜 Starting development server with log tailing..."
	@echo "📊 Consolidated log output enabled (like tail -f)"
	@echo ""
	@npx dev3000 --kill-mcp 2>/dev/null || true
	npx dev3000 --port 8642 --script start --tail

start: build
	@echo "🚀 Starting server..."
	npm run start

# ============================================================================
# Cleanup
# ============================================================================

clean:
	@echo "🧹 Cleaning build artifacts..."
	npm run clean

clean-all:
	@echo "🧹 Cleaning all build artifacts and dependencies..."
	npm run clean:artifacts
	rm -rf node_modules

# ============================================================================
# Port Management
# ============================================================================

port-check:
	@echo "🔍 Checking port 8642..."
	@npm run port:who || true

port-kill:
	@echo "🔧 Freeing port 8642..."
	@npm run port:kill || true

mcp-kill:
	@echo "🔧 Stopping MCP server on port 3684..."
	@npx dev3000 --kill-mcp || true

# ============================================================================
# CI/CD
# ============================================================================

ci: install lint typecheck test build
	@echo "✅ CI pipeline completed successfully"
