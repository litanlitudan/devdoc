#!/usr/bin/env python3
"""
MLIR Graph Parser using Model Explorer's Pre-Built Adapter

Uses the ai-edge-model-explorer-adapter package which contains pre-compiled
MLIR parsing code from Google's Model Explorer project.

Usage:
    python3 parse_mlir_with_adapter.py <filename> < model.mlir > graph.json
"""

import sys
import json
from typing import Dict, List, Any

# Import Model Explorer's pre-built adapter
try:
    from ai_edge_model_explorer_adapter import _pywrap_convert_wrapper as convert_wrapper
    ADAPTER_AVAILABLE = True
except ImportError:
    ADAPTER_AVAILABLE = False


def parse_with_adapter(mlir_content: str, filename: str) -> Dict[str, Any]:
    """
    Parse MLIR using Model Explorer's pre-built C++ adapter.
    This is the fastest and most accurate method.
    """
    import tempfile
    import os

    # Write MLIR to temporary file (adapter expects file path)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
        f.write(mlir_content)
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

                return {
                    'id': filename,
                    'nodes': nodes
                }

        return {
            'id': filename,
            'nodes': []
        }
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def enhance_node_label_with_shapes(node: Dict[str, Any], node_map: Dict[str, Any] = None) -> None:
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
