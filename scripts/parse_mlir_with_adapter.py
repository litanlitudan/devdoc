#!/usr/bin/env python3
"""
MLIR Graph Parser using Model Explorer's Pre-Built Adapter

Uses the ai-edge-model-explorer-adapter package which contains pre-compiled
MLIR parsing code from Google's Model Explorer project.

Removes all dense constant values to prevent segfaults, preserving only tensor shape information.

Usage:
    python3 parse_mlir_with_adapter.py <filename> < model.mlir > graph.json
"""

import sys
import json
import re
from typing import Dict, List, Any, Tuple, Optional

# Import Model Explorer's pre-built adapter
try:
    from ai_edge_model_explorer_adapter import _pywrap_convert_wrapper as convert_wrapper
    ADAPTER_AVAILABLE = True
except ImportError:
    ADAPTER_AVAILABLE = False


def remove_dense_constant_values(mlir_content: str) -> Tuple[str, int]:
    """
    Remove all dense constant values from MLIR, preserving only tensor shape information.

    The C++ adapter can crash when parsing MLIR with large constant tensors.
    For visualization purposes, we only need tensor shapes, not the actual values.

    Transforms:
        - Single line: %cst = arith.constant dense<[1.0, 2.0, ...]> : tensor<1000xf32>
          → %cst = arith.constant dense<0.0> : tensor<1000xf32>  // VALUES_REMOVED

        - Multi-line: %weights = "tf.Const"() {value = dense<[[...], [...]]> : tensor<1000x1000xf32>}
          → %weights = "tf.Const"() {value = dense<0.0> : tensor<1000x1000xf32>}  // VALUES_REMOVED

    Returns:
        Tuple of (preprocessed_mlir, count_of_replaced_constants)
    """
    replaced_count = 0

    def replace_dense_values(match):
        nonlocal replaced_count
        prefix = match.group(1)
        dense_content = match.group(2)
        suffix = match.group(3)

        # Always replace with minimal placeholder
        replaced_count += 1

        # Extract tensor type for informational comment
        type_match = re.search(r'tensor<([^>]+)>', suffix)
        tensor_type = type_match.group(1) if type_match else "unknown"

        # Calculate original size for debugging
        size_kb = len(dense_content) / 1024
        size_info = f"{size_kb:.1f}KB" if size_kb < 1024 else f"{size_kb/1024:.1f}MB"

        # Replace with minimal placeholder preserving type information
        return f'{prefix}dense<0.0>{suffix}  // VALUES_REMOVED ({size_info}, shape: {tensor_type})'

    # Pattern to match dense constants (handles multi-line with non-greedy matching)
    # Captures: (prefix)(dense<...content...>)(: tensor<...> suffix)
    dense_pattern = re.compile(
        r'(.*?dense)<([^>]+(?:>[^:]*?<[^>]+)*)>(.*?tensor<[^>]+>.*?)(?=\s*(?:\n|$|//|%|\}))',
        re.DOTALL
    )

    processed_content = dense_pattern.sub(replace_dense_values, mlir_content)

    return processed_content, replaced_count


def parse_with_adapter(mlir_content: str, filename: str) -> Dict[str, Any]:
    """
    Parse MLIR using Model Explorer's pre-built C++ adapter.
    This is the fastest and most accurate method.

    Removes all dense constant values to prevent segfaults in the C++ adapter.
    Only tensor shapes are preserved for graph visualization.
    """
    import tempfile
    import os

    # Remove all dense constant values (we only need shapes for visualization)
    processed_mlir, replaced_count = remove_dense_constant_values(mlir_content)

    # Write preprocessed MLIR to temporary file (adapter expects file path)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
        f.write(processed_mlir)
        temp_path = f.name

    try:
        # Use Model Explorer's C++ MLIR parser
        config = convert_wrapper.VisualizeConfig()
        json_str = convert_wrapper.ConvertMlirToJson(config, temp_path)

        # Parse the JSON response
        result = json.loads(json_str)

        # Model Explorer returns a list of graph collections
        # Each collection has subgraphs with nodes
        if isinstance(result, list) and len(result) > 0:
            collection = result[0]  # First graph collection

            # Look for subgraphs (Model Explorer uses subgraphs instead of top-level nodes)
            if 'subgraphs' in collection and len(collection['subgraphs']) > 0:
                subgraph = collection['subgraphs'][0]  # Usually 'main' function
                nodes = subgraph.get('nodes', [])

                # Build node lookup for following edges
                node_map = {node['id']: node for node in nodes}

                # Enhance labels with tensor shape information
                for node in nodes:
                    enhance_node_label_with_shapes(node, node_map)

                result_graph = {
                    'id': filename,
                    'nodes': nodes
                }

                # Add info about removed constants
                if replaced_count > 0:
                    result_graph['metadata'] = {
                        'constants_removed': replaced_count,
                        'message': f'Removed {replaced_count} dense constant value(s). Shape information preserved.'
                    }

                return result_graph

        result_graph = {
            'id': filename,
            'nodes': []
        }

        # Add info about removed constants
        if replaced_count > 0:
            result_graph['metadata'] = {
                'constants_removed': replaced_count,
                'message': f'Removed {replaced_count} dense constant value(s). Shape information preserved.'
            }

        return result_graph
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def enhance_node_label_with_shapes(node: Dict[str, Any], node_map: Optional[Dict[str, Any]] = None) -> None:
    """
    Enhance node label with input and output tensor shape information.
    Modifies the node in place.

    Args:
        node: The node to enhance
        node_map: Optional map of node IDs to nodes for following edges
    """
    base_label = node.get('label', '')

    # Extract input shapes from metadata or by following incoming edges
    input_shapes = []

    # First try direct inputsMetadata
    if 'inputsMetadata' in node and node['inputsMetadata']:
        for input_meta in node['inputsMetadata']:
            if 'attrs' in input_meta:
                for attr in input_meta['attrs']:
                    if attr.get('key') == 'tensor_shape':
                        input_shapes.append(attr.get('value', ''))

    # If no inputsMetadata, infer from incoming edges
    elif node_map and 'incomingEdges' in node and node['incomingEdges']:
        for edge in node['incomingEdges']:
            source_id = edge.get('sourceNodeId')
            source_output_id = edge.get('sourceNodeOutputId', '0')

            if source_id in node_map:
                source_node = node_map[source_id]
                # Get the output shape from the source node
                if 'outputsMetadata' in source_node and source_node['outputsMetadata']:
                    for output_meta in source_node['outputsMetadata']:
                        if output_meta.get('id') == source_output_id:
                            for attr in output_meta.get('attrs', []):
                                if attr.get('key') == 'tensor_shape':
                                    input_shapes.append(attr.get('value', ''))
                                    break

    # Extract output shapes
    output_shapes = []
    if 'outputsMetadata' in node and node['outputsMetadata']:
        for output_meta in node['outputsMetadata']:
            if 'attrs' in output_meta:
                for attr in output_meta['attrs']:
                    if attr.get('key') == 'tensor_shape':
                        output_shapes.append(attr.get('value', ''))

    # Build enhanced label
    enhanced_label = base_label

    if input_shapes:
        # Format input shapes
        input_str = ', '.join(input_shapes)
        enhanced_label += f"\nin: {input_str}"

    if output_shapes:
        # Format output shapes
        output_str = ', '.join(output_shapes)
        enhanced_label += f"\nout: {output_str}"

    # Limit label length for display
    if len(enhanced_label) > 150:
        enhanced_label = enhanced_label[:147] + "..."

    node['label'] = enhanced_label


def main():
    try:
        # Get filename from command line argument
        filename = sys.argv[1] if len(sys.argv) > 1 else "mlir_model"

        # Read MLIR content from stdin
        mlir_content = sys.stdin.read()

        if not mlir_content.strip():
            print(json.dumps({
                "error": "Empty input",
                "message": "No MLIR content provided"
            }), file=sys.stderr)
            return 1

        # Warn about very large files (>100MB total)
        content_size_mb = len(mlir_content) / (1024 * 1024)
        if content_size_mb > 100:
            print(json.dumps({
                "error": "File too large",
                "message": f"MLIR file is {content_size_mb:.1f}MB. Files over 100MB may cause "
                          f"memory issues or timeouts. Consider splitting the model or using "
                          f"a dedicated MLIR viewer."
            }), file=sys.stderr)
            return 1

        # Check if adapter is available
        if not ADAPTER_AVAILABLE:
            print(json.dumps({
                "error": "Model Explorer adapter not available",
                "message": "Install with: pip install ai-edge-model-explorer-adapter"
            }))
            return 0  # Return 0 so TypeScript can parse the error JSON

        # Parse MLIR using Model Explorer adapter
        try:
            graph = parse_with_adapter(mlir_content, filename)
            print(json.dumps(graph, indent=2))
            return 0
        except Exception as e:
            # Return error JSON to stdout so TypeScript can show detailed error
            print(json.dumps({
                "error": "MLIR parsing failed",
                "message": str(e)
            }))
            return 0  # Return 0 so TypeScript can parse the error JSON

    except Exception as e:
        print(json.dumps({
            "error": "Script execution failed",
            "message": str(e)
        }), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
