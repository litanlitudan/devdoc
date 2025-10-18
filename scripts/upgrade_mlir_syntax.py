#!/usr/bin/env python3
"""
MLIR Syntax Upgrader - Convert old MLIR syntax to MLIR 21.x compatible syntax

Handles dialect version changes:
- tensor.expand_shape: Add output_shape attribute from result type
- tensor.collapse_shape: Similar transformations
"""

import sys
import re


def upgrade_tensor_expand_shape(mlir_content: str) -> tuple[str, int]:
    """
    Upgrade tensor.expand_shape from old syntax to MLIR 21.x syntax.

    Old: %result = tensor.expand_shape %input [[0], [1, 2]] : tensor<12xf32> into tensor<1x12xf32>
    New: %result = tensor.expand_shape %input [[0], [1, 2]] output_shape [1, 12] : tensor<12xf32> into tensor<1x12xf32>

    Returns: (upgraded_content, count_of_upgrades)
    """
    upgrade_count = 0

    def extract_output_shape(result_type: str) -> str:
        """Extract shape dimensions from tensor<...> type."""
        # Match tensor<1x12x128xf32> -> [1, 12, 128]
        # Match the dimension pattern: number followed by 'x', up until we hit a letter (element type)
        match = re.search(r'tensor<((?:\d+x)+)\w', result_type)
        if match:
            # Split by 'x' and remove empty strings
            dims = [d for d in match.group(1).split('x') if d]
            return f"[{', '.join(dims)}]"
        return ""

    def upgrade_expand_shape(match):
        nonlocal upgrade_count
        prefix = match.group(1)  # %result = tensor.expand_shape %input
        reassociation = match.group(2)  # [[0], [1, 2, 3]]
        input_type = match.group(3)  # tensor<1x12xf32>
        result_type = match.group(4)  # tensor<1x1x1x12xf32>

        # Extract output shape from result type
        output_shape = extract_output_shape(result_type)

        if not output_shape:
            # Can't parse shape, return original
            return match.group(0)

        upgrade_count += 1

        # Build upgraded syntax
        return f"{prefix}{reassociation} output_shape {output_shape} : {input_type} into {result_type}"

    # Pattern for tensor.expand_shape without output_shape attribute
    # Matches: %result = tensor.expand_shape %input [[...]] : type into type
    # Does NOT match if 'output_shape' already present
    expand_pattern = re.compile(
        r'(%[\w]+\s*=\s*tensor\.expand_shape\s+%[\w]+\s+)'  # prefix with result assignment
        r'(\[\[[\d,\s\[\]]+\]\])\s+'  # reassociation indices
        r'(?!output_shape)'  # negative lookahead - don't match if output_shape already present
        r':\s*'
        r'(tensor<[^>]+>)\s+'  # input type
        r'into\s+'
        r'(tensor<[^>]+>)',  # result type
        re.MULTILINE
    )

    upgraded_content = expand_pattern.sub(upgrade_expand_shape, mlir_content)

    return upgraded_content, upgrade_count


def main():
    # Read MLIR content from stdin
    mlir_content = sys.stdin.read()

    if not mlir_content.strip():
        print("Error: No MLIR content provided", file=sys.stderr)
        return 1

    # Apply upgrades
    upgraded_content, expand_upgrades = upgrade_tensor_expand_shape(mlir_content)

    # Print upgraded content to stdout
    print(upgraded_content)

    # Print upgrade summary to stderr
    if expand_upgrades > 0:
        print(f"✓ Upgraded {expand_upgrades} tensor.expand_shape operation(s)", file=sys.stderr)
    else:
        print("ℹ No syntax upgrades needed", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
