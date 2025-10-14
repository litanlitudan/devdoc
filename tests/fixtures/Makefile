# Sample Makefile for testing markserv rendering
# This demonstrates Makefile syntax highlighting

.PHONY: all clean test install

# Variables
CC = gcc
CFLAGS = -Wall -Wextra -O2
TARGET = myapp
SOURCES = main.c utils.c
OBJECTS = $(SOURCES:.c=.o)

# Default target
all: $(TARGET)

# Build the target
$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^
	@echo "Build complete!"

# Compile source files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Run tests
test: $(TARGET)
	@echo "Running tests..."
	./$(TARGET) --test

# Install the application
install: $(TARGET)
	install -m 755 $(TARGET) /usr/local/bin/

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET)
	@echo "Cleaned build artifacts"

# Help target
help:
	@echo "Available targets:"
	@echo "  all      - Build the application (default)"
	@echo "  test     - Run tests"
	@echo "  install  - Install to /usr/local/bin"
	@echo "  clean    - Remove build artifacts"
	@echo "  help     - Show this help message"
