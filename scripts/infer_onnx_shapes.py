#!/usr/bin/env python3
"""
ONNX Shape Inference Script

Reads an ONNX model from stdin, performs shape inference using the official
ONNX library, and writes the enriched model to stdout.

Usage:
    python3 infer_onnx_shapes.py < model.onnx > model_with_shapes.onnx
"""

import sys
import onnx
from onnx import shape_inference

def main():
    try:
        # Read model from stdin (binary)
        model_bytes = sys.stdin.buffer.read()

        # Parse the model
        model = onnx.ModelProto()
        model.ParseFromString(model_bytes)

        # Perform shape inference
        inferred_model = shape_inference.infer_shapes(model)

        # Write enriched model to stdout (binary)
        sys.stdout.buffer.write(inferred_model.SerializeToString())

        return 0

    except Exception as e:
        # Write error to stderr
        sys.stderr.write(f"Shape inference error: {str(e)}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
