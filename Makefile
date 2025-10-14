.PHONY: dev install-dev3000 test cover lint setup-python build-adapter clean help

# Default target
.DEFAULT_GOAL := help

# Development server with dev3000
dev:
	npm run port:kill && npm run build && npx dev3000 --port 8642

# Install dev3000 globally
install-dev3000:
	npm install -g dev3000

# Run tests
test:
	npm test

# Run tests with coverage
cover:
	npm run cover

# Run linter
lint:
	npx xo

# Setup Python dependencies for ONNX/MLIR support
setup-python:
	npm run setup:python

# Build Model Explorer adapter from source
build-adapter:
	npm run build:adapter

# Clean node_modules and reinstall
clean:
	rm -rf node_modules
	npm install

# Show available commands
help:
	@echo "Available commands:"
	@echo "  make dev            - Kill ports, build, and start dev3000 server on port 8642"
	@echo "  make install-dev3000 - Install dev3000 globally with npm"
	@echo "  make test           - Run tests"
	@echo "  make cover          - Run tests with coverage"
	@echo "  make lint           - Run XO linter"
	@echo "  make setup-python   - Install Python dependencies for ONNX/MLIR"
	@echo "  make build-adapter  - Build Model Explorer adapter from source"
	@echo "  make clean          - Remove node_modules and reinstall"
	@echo "  make help           - Show this help message"
