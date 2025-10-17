# Makefile for devdoc - Build automation and task orchestration
#
# This Makefile provides a unified interface for all development tasks,
# enabling consistent workflows across local development and CI/CD.

.PHONY: help install build test lint clean dev watch ci port-check port-kill typecheck format
.PHONY: install-dev3000 setup-python build-adapter clean-all test-watch test-ui start

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
	@echo "  🚀 Development:"
	@echo "    make dev              Kill ports, build, start dev3000 server"
	@echo "    make start            Start server (requires build)"
	@echo ""
	@echo "  🧹 Cleanup:"
	@echo "    make clean            Remove dist/ directory"
	@echo "    make clean-all        Remove all build artifacts and node_modules"
	@echo ""
	@echo "  🔧 Utilities:"
	@echo "    make port-check       Check if port 8642 is in use"
	@echo "    make port-kill        Kill process on port 8642"
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
	@echo "🚀 Starting development server with dev3000..."
	npx dev3000 --port 8642

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

# ============================================================================
# CI/CD
# ============================================================================

ci: install lint typecheck test build
	@echo "✅ CI pipeline completed successfully"
