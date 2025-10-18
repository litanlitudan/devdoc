#!/usr/bin/env python3
"""
MLIR Parser - C++ Context Implementation Wrapper

This script wraps the C++ MLIR parser that uses MLIR context and proper dialect registration.
Falls back to regex-based parser if C++ binary is not available.

The C++ implementation provides:
- Proper MLIR context with dialect registration
- Parsing to ModuleOp with allowUnregisteredDialects(true)
- Conditional normalization (VHLO→StableHLO)
- CreateUniqueOpNamesPass for stable node IDs
- Full region and block traversal

Build instructions:
    cd src/mlir
    mkdir build && cd build
    cmake -DCMAKE_PREFIX_PATH=/path/to/llvm/install ..
    make
    sudo make install

Usage:
    python3 parse_mlir_cpp.py <filename> < model.mlir > graph.json
"""

import sys
import subprocess
import json
from pathlib import Path

def find_cpp_parser():
    """
    Find the C++ MLIR parser binary.
    Searches in order:
    1. src/mlir/build/mlir_parser
    2. src/mlir/mlir_parser
    3. System PATH
    """
    script_dir = Path(__file__).parent.parent

    # Check build directory
    build_path = script_dir / "src" / "mlir" / "build" / "mlir_parser"
    if build_path.exists():
        return str(build_path)

    # Check source directory
    src_path = script_dir / "src" / "mlir" / "mlir_parser"
    if src_path.exists():
        return str(src_path)

    # Check system PATH
    try:
        result = subprocess.run(["which", "mlir_parser"],
                              capture_output=True, text=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass

    return None

def use_cpp_parser(mlir_content: str, filename: str) -> str:
    """
    Use C++ MLIR parser if available.

    Args:
        mlir_content: MLIR text content
        filename: Filename to use as graph ID

    Returns:
        JSON string from parser

    Raises:
        FileNotFoundError: If C++ parser not found
        subprocess.CalledProcessError: If parser fails
    """
    parser_path = find_cpp_parser()
    if not parser_path:
        raise FileNotFoundError("C++ MLIR parser not found. Build instructions in src/mlir/BUILD.md")

    # Run C++ parser
    result = subprocess.run(
        [parser_path, filename],
        input=mlir_content,
        capture_output=True,
        text=True,
        timeout=30
    )

    if result.returncode != 0:
        # Parser error - try to parse JSON error message
        try:
            error_json = json.loads(result.stdout)
            if "error" in error_json:
                # Return the error JSON
                return result.stdout
        except json.JSONDecodeError:
            pass

        # Fallback to stderr if stdout isn't valid JSON
        raise subprocess.CalledProcessError(
            result.returncode,
            parser_path,
            output=result.stdout,
            stderr=result.stderr
        )

    return result.stdout

def use_regex_parser(mlir_content: str, filename: str) -> str:
    """
    Fallback to regex-based Python parser.

    Args:
        mlir_content: MLIR text content
        filename: Filename to use as graph ID

    Returns:
        JSON string from parser
    """
    script_dir = Path(__file__).parent
    regex_parser = script_dir / "parse_mlir_regex.py"

    result = subprocess.run(
        ["python3", str(regex_parser), filename],
        input=mlir_content,
        capture_output=True,
        text=True,
        timeout=30
    )

    return result.stdout

def main():
    if len(sys.argv) < 2:
        print(json.dumps({
            "error": "Invalid arguments",
            "message": "Usage: parse_mlir_cpp.py <filename>"
        }), file=sys.stderr)
        return 1

    filename = sys.argv[1]

    # Read MLIR content from stdin
    mlir_content = sys.stdin.read()

    if not mlir_content.strip():
        print(json.dumps({
            "error": "Empty input",
            "message": "No MLIR content provided"
        }))
        return 1

    try:
        # Try C++ parser first
        output = use_cpp_parser(mlir_content, filename)

        # Check if C++ parser returned an error (parsing failed)
        try:
            result_data = json.loads(output)
            if "error" in result_data:
                # C++ parser failed to parse, fall back to regex parser
                print(f"⚠ C++ parser failed to parse MLIR: {result_data.get('message', 'Unknown error')}", file=sys.stderr)
                print("  Falling back to regex parser", file=sys.stderr)

                output = use_regex_parser(mlir_content, filename)
                print(output)
                print("✓ Used regex-based parser (fallback)", file=sys.stderr)
                return 0
        except json.JSONDecodeError:
            # Output is not valid JSON, fall back to regex parser
            print("⚠ C++ parser returned invalid JSON", file=sys.stderr)
            print("  Falling back to regex parser", file=sys.stderr)

            output = use_regex_parser(mlir_content, filename)
            print(output)
            print("✓ Used regex-based parser (fallback)", file=sys.stderr)
            return 0

        # C++ parser succeeded
        print(output)
        print("✓ Used C++ MLIR context parser", file=sys.stderr)
        return 0

    except FileNotFoundError:
        # C++ parser not available, fall back to regex parser
        print("ℹ C++ parser not found, using regex fallback", file=sys.stderr)
        print("  Build C++ parser with: cd src/mlir && mkdir build && cd build && cmake .. && make", file=sys.stderr)

        output = use_regex_parser(mlir_content, filename)
        print(output)
        print("✓ Used regex-based parser (fallback)", file=sys.stderr)
        return 0

    except subprocess.CalledProcessError as e:
        # C++ parser failed, fall back to regex parser
        print(f"⚠ C++ parser failed: {e.stderr}", file=sys.stderr)
        print("  Falling back to regex parser", file=sys.stderr)

        output = use_regex_parser(mlir_content, filename)
        print(output)
        print("✓ Used regex-based parser (fallback)", file=sys.stderr)
        return 0

    except Exception as e:
        print(json.dumps({
            "error": "Parser execution failed",
            "message": str(e)
        }), file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
