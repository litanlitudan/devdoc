#!/usr/bin/env python3
"""
Direct MLIR Graph Parser - Supports Arbitrary MLIR Dialects

Parses MLIR text format and constructs Model Explorer graph structure directly in Python.
Supports ALL MLIR dialects by treating operations as generic graph nodes.

Features:
- Supports arbitrary MLIR dialects (no C++ adapter limitations)
- Constructs Model Explorer graph format directly
- Preserves tensor shape information for visualization
- Optional dense constant removal to reduce file size

Supported Approach:
- Parses MLIR text format using Python regex
- Creates Model Explorer GraphNode structures
- Treats all dialects uniformly as generic operations
- No dependency on pre-compiled adapter with hardcoded dialect lists

Usage:
    python3 parse_mlir.py <filename> < model.mlir > graph.json
"""

import sys
import json
import re
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field, asdict


@dataclass
class KeyValue:
    """Key-value pair for attributes."""
    key: str
    value: str


@dataclass
class MetadataItem:
    """Metadata for inputs/outputs."""
    id: str
    attrs: List[KeyValue] = field(default_factory=list)


@dataclass
class IncomingEdge:
    """Edge connecting nodes."""
    sourceNodeId: str
    sourceNodeOutputId: str
    targetNodeInputId: str


@dataclass
class GraphNode:
    """Node in the graph."""
    id: str
    label: str
    namespace: str = ""
    attrs: List[KeyValue] = field(default_factory=list)
    incomingEdges: List[IncomingEdge] = field(default_factory=list)
    inputsMetadata: List[MetadataItem] = field(default_factory=list)
    outputsMetadata: List[MetadataItem] = field(default_factory=list)
    subgraphIds: List[str] = field(default_factory=list)


@dataclass
class MLIROperation:
    """Parsed MLIR operation."""
    outputs: List[str]  # Result values like %0, %1
    op_type: str  # Operation type like "arith.addf" or "custom.transform"
    inputs: List[str]  # Input values like %arg0, %1
    attributes: Dict[str, str]  # Attributes like {mode = "normalize"}
    result_types: List[str]  # Result types like "tensor<2x3xf32>"


def parse_mlir_operations(mlir_content: str) -> List[MLIROperation]:
    """
    Parse MLIR text format to extract operations.

    This is a simplified parser that handles common MLIR patterns:
    - %result = "dialect.operation"(%inputs) {attrs} : (input_types) -> result_types
    - %result = dialect.operation %inputs {attrs} : result_types

    IMPORTANT: Operations are returned in their original order to preserve SSA dependencies.
    """
    # Pattern for quoted operations: %result = "dialect.op"(%inputs) {attrs} : type
    quoted_pattern = re.compile(
        r'(%[\w]+(?:,\s*%[\w]+)*)\s*=\s*'  # outputs
        r'"([^"]+)"\s*'  # operation
        r'\(([^)]*)\)\s*'  # inputs
        r'(?:\{([^}]*)\})?\s*'  # attributes (optional)
        r':\s*(?:\([^)]+\)\s*->\s*)?(.+?)(?:\s|$)',  # result type
        re.MULTILINE
    )

    # Pattern for unquoted operations: %result = dialect.op %inputs {attrs} : type
    unquoted_pattern = re.compile(
        r'(%[\w]+(?:,\s*%[\w]+)*)\s*=\s*'  # outputs
        r'([\w]+\.[\w]+)\s+'  # operation (dialect.op)
        r'([^{:]+?)\s*'  # inputs
        r'(?:\{([^}]*)\})?\s*'  # attributes (optional)
        r':\s*(.+?)(?:\s|$)',  # result type
        re.MULTILINE
    )

    # Collect all matches with their positions to preserve order
    all_matches = []

    # Find all quoted operations
    for match in quoted_pattern.finditer(mlir_content):
        all_matches.append(('quoted', match.start(), match))

    # Find all unquoted operations
    for match in unquoted_pattern.finditer(mlir_content):
        all_matches.append(('unquoted', match.start(), match))

    # Sort by position to preserve original order
    all_matches.sort(key=lambda x: x[1])

    # Process matches in order
    operations = []
    for match_type, _, match in all_matches:
        if match_type == 'quoted':
            outputs = [s.strip() for s in match.group(1).split(',')]
            op_type = match.group(2)
            inputs = [s.strip() for s in match.group(3).split(',') if s.strip() and s.strip() != '']
            attrs_str = match.group(4) or ""
            result_types_str = match.group(5)
        else:  # unquoted
            outputs = [s.strip() for s in match.group(1).split(',')]
            op_type = match.group(2)
            inputs_str = match.group(3)
            # Parse inputs - handle comma-separated values
            inputs = [s.strip() for s in inputs_str.split(',') if s.strip() and s.strip().startswith('%')]
            if not inputs and inputs_str.strip().startswith('%'):
                inputs = [inputs_str.strip()]
            attrs_str = match.group(4) or ""
            result_types_str = match.group(5)

        # Parse attributes
        attributes = {}
        if attrs_str:
            attr_pattern = re.compile(r'(\w+)\s*=\s*([^,}]+)')
            for attr_match in attr_pattern.finditer(attrs_str):
                key = attr_match.group(1)
                value = attr_match.group(2).strip()
                attributes[key] = value

        # Parse result types
        result_types = [t.strip() for t in result_types_str.split('->') if t.strip()]
        if result_types:
            result_types = [result_types[-1]]  # Take the final result type

        operations.append(MLIROperation(
            outputs=outputs,
            op_type=op_type,
            inputs=inputs,
            attributes=attributes,
            result_types=result_types
        ))

    return operations


def parse_function_inputs(mlir_content: str) -> List[Tuple[str, str]]:
    """
    Parse function inputs from MLIR.
    Returns list of (name, type) tuples.
    """
    # Pattern: func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>)
    func_pattern = re.compile(
        r'func\.func\s+@\w+\s*\(([^)]+)\)',
        re.MULTILINE
    )

    match = func_pattern.search(mlir_content)
    if not match:
        return []

    inputs_str = match.group(1)
    inputs = []

    # Parse each input: %arg0: tensor<2x3xf32>
    input_pattern = re.compile(r'(%\w+)\s*:\s*([^,)]+)')
    for input_match in input_pattern.finditer(inputs_str):
        name = input_match.group(1)
        type_str = input_match.group(2).strip()
        inputs.append((name, type_str))

    return inputs


def parse_function_outputs(mlir_content: str) -> List[str]:
    """
    Parse function return values from MLIR.
    Returns list of output value names.
    """
    # Pattern: func.return %result : tensor<2x3xf32>
    return_pattern = re.compile(
        r'func\.return\s+([^:]+)',
        re.MULTILINE
    )

    match = return_pattern.search(mlir_content)
    if not match:
        return []

    outputs_str = match.group(1).strip()
    outputs = [s.strip() for s in outputs_str.split(',')]
    return outputs


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


def create_model_explorer_graph(
    operations: List[MLIROperation],
    inputs: List[Tuple[str, str]],
    outputs: List[str],
    filename: str
) -> Dict[str, Any]:
    """
    Create Model Explorer graph structure from parsed MLIR operations.
    """
    nodes = []
    node_id_counter = 0
    value_to_output_node = {}  # Maps value names to (node_id, output_index)

    # Create input nodes
    for i, (input_name, input_type) in enumerate(inputs):
        node = GraphNode(
            id=f"input_{i}",
            label="Input",
            namespace="Inputs"
        )
        node.attrs.append(KeyValue(key="name", value=input_name))
        node.attrs.append(KeyValue(key="index", value=str(i)))

        # Add output metadata for the input
        output_meta = MetadataItem(id="0")
        output_meta.attrs.append(KeyValue(key="__tensor_tag", value=input_name))
        output_meta.attrs.append(KeyValue(key="tensor_shape", value=input_type))
        node.outputsMetadata.append(output_meta)

        nodes.append(node)
        value_to_output_node[input_name] = (node.id, "0")

    # Create operation nodes
    for op in operations:
        node_id = f"op_{node_id_counter}"
        node_id_counter += 1

        node = GraphNode(
            id=node_id,
            label=op.op_type,
            namespace="main"
        )

        # Add attributes
        for key, value in op.attributes.items():
            node.attrs.append(KeyValue(key=key, value=value))

        # Add incoming edges from inputs
        for input_idx, input_name in enumerate(op.inputs):
            if input_name in value_to_output_node:
                source_node_id, source_output_id = value_to_output_node[input_name]
                edge = IncomingEdge(
                    sourceNodeId=source_node_id,
                    sourceNodeOutputId=source_output_id,
                    targetNodeInputId=str(input_idx)
                )
                node.incomingEdges.append(edge)

                # Add input metadata
                input_meta = MetadataItem(id=str(input_idx))
                input_meta.attrs.append(KeyValue(key="__tensor_tag", value=input_name))
                node.inputsMetadata.append(input_meta)

        # Add output metadata
        for output_idx, output_name in enumerate(op.outputs):
            output_meta = MetadataItem(id=str(output_idx))
            output_meta.attrs.append(KeyValue(key="__tensor_tag", value=output_name))
            if op.result_types and output_idx < len(op.result_types):
                output_meta.attrs.append(KeyValue(key="tensor_shape", value=op.result_types[output_idx]))
            node.outputsMetadata.append(output_meta)

            # Record this node as producing this value
            value_to_output_node[output_name] = (node.id, str(output_idx))

        nodes.append(node)

    # Create output nodes
    for i, output_name in enumerate(outputs):
        node = GraphNode(
            id=f"output_{i}",
            label="Output",
            namespace="Outputs"
        )
        node.attrs.append(KeyValue(key="name", value=output_name))
        node.attrs.append(KeyValue(key="index", value=str(i)))

        # Add incoming edge from the producing operation
        if output_name in value_to_output_node:
            source_node_id, source_output_id = value_to_output_node[output_name]
            edge = IncomingEdge(
                sourceNodeId=source_node_id,
                sourceNodeOutputId=source_output_id,
                targetNodeInputId="0"
            )
            node.incomingEdges.append(edge)

        nodes.append(node)

    # Convert to dict format (Model Explorer expects this structure)
    nodes_dict = [asdict(node) for node in nodes]

    return {
        'id': filename,
        'nodes': nodes_dict
    }


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

        # Parse MLIR using direct Python parser
        try:
            operations = parse_mlir_operations(mlir_content)
            inputs = parse_function_inputs(mlir_content)
            outputs = parse_function_outputs(mlir_content)

            # Create Model Explorer graph
            graph = create_model_explorer_graph(operations, inputs, outputs, filename)

            # Output as JSON
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
