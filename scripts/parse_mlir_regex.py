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
import hashlib
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
class GraphNodeStyle:
    """Style for a graph node."""
    backgroundColor: str = ""
    borderColor: str = ""
    textColor: str = ""
    borderWidth: float = 1.2


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
    style: GraphNodeStyle = None


@dataclass
class EdgeData:
    """Edge in an overlay."""
    sourceNodeId: str
    targetNodeId: str
    label: str = ""


@dataclass
class EdgeOverlay:
    """Single edge overlay group."""
    name: str
    edges: List[EdgeData]
    edgeColor: str
    edgeWidth: int = 2
    edgeLabelFontSize: float = 7.5


@dataclass
class EdgeOverlaysData:
    """Edge overlays task data."""
    type: str = "EDGE_OVERLAYS"  # TaskType.EDGE_OVERLAYS
    name: str = ""
    overlays: List[EdgeOverlay] = field(default_factory=list)


@dataclass
class TasksData:
    """Optional tasks data for edge overlays and additional metadata."""
    edgeOverlaysDataListLeftPane: List[EdgeOverlaysData] = field(default_factory=list)


@dataclass
class Graph:
    """Graph representing a function or region."""
    id: str
    nodes: List[GraphNode] = field(default_factory=list)
    tasksData: TasksData = field(default_factory=TasksData)


@dataclass
class ModelExplorerGraphs:
    """Top-level structure containing multiple graphs."""
    graphs: List[Graph] = field(default_factory=list)


@dataclass
class MLIROperation:
    """Parsed MLIR operation."""
    outputs: List[str]  # Result values like %0, %1
    op_type: str  # Operation type like "arith.addf" or "custom.transform"
    inputs: List[str]  # Input values like %arg0, %1
    attributes: Dict[str, str]  # Attributes like {mode = "normalize"}
    result_types: List[str]  # Result types like "tensor<2x3xf32>"
    regions: List[str] = field(default_factory=list)  # Nested region contents (text)
    namespace: str = ""  # Hierarchical namespace path for Model Explorer layers


def extract_ssa_values(text: str) -> List[str]:
    """
    Extract SSA value names from MLIR text, ignoring array indices and other syntax.

    Examples:
        "%base[1, 1]" -> ["%base"]
        "%arg0, %arg1" -> ["%arg0", "%arg1"]
        "%0, %1[0]" -> ["%0", "%1"]
        "@function_name" -> ["@function_name"]
    """
    # Pattern to match SSA values (%name) or function references (@name)
    # Stops at: brackets, commas, colons, whitespace (unless part of identifier)
    ssa_pattern = re.compile(r'[%@]\w+')
    return ssa_pattern.findall(text)


def extract_regions_from_line(mlir_line: str) -> Tuple[str, List[str]]:
    """
    Extract region contents from an MLIR operation line.

    Regions are delimited by braces { ... }. For operations with multiple brace blocks,
    we need to distinguish between:
    - Attribute blocks: {attr = value, ...}
    - Region blocks: { ^bb0(...): operations... }

    Region blocks contain block arguments (^bb0) or operations.

    Returns:
        Tuple of (line_without_regions, list_of_region_contents)
    """
    regions = []
    result_parts = []
    pos = 0

    while pos < len(mlir_line):
        # Find next opening brace
        brace_start = mlir_line.find('{', pos)

        if brace_start == -1:
            # No more braces, append rest of line
            result_parts.append(mlir_line[pos:])
            break

        # Append content before brace
        result_parts.append(mlir_line[pos:brace_start])

        # Find matching closing brace
        brace_count = 1
        scan_pos = brace_start + 1
        region_start = scan_pos

        while scan_pos < len(mlir_line) and brace_count > 0:
            if mlir_line[scan_pos] == '{':
                brace_count += 1
            elif mlir_line[scan_pos] == '}':
                brace_count -= 1
            scan_pos += 1

        if brace_count == 0:
            # Found complete region
            region_content = mlir_line[region_start:scan_pos-1]

            # Heuristic: Is this a region or attribute block?
            # Regions contain: ^bb, newlines, or operation patterns like %n =
            is_region = (
                '^bb' in region_content or
                '\n' in region_content or
                re.search(r'%\w+\s*=', region_content)
            )

            if is_region:
                regions.append(region_content)
                result_parts.append('{ /* region */ }')
            else:
                # Keep attribute block in the line
                result_parts.append('{' + region_content + '}')

            pos = scan_pos
        else:
            # Unmatched braces, keep as-is
            result_parts.append(mlir_line[brace_start:])
            break

    return ''.join(result_parts), regions


def parse_mlir_operations(mlir_content: str) -> List[MLIROperation]:
    """
    Parse MLIR text format to extract operations.

    Handles various MLIR patterns:
    - With results: %result = "dialect.operation"(%inputs) {attrs} : (input_types) -> result_types
    - With results: %result = dialect.operation %inputs {attrs} : result_types
    - Without results: dialect.operation(%inputs) {attrs} : type
    - Without inputs: %result = dialect.operation {attrs} : type

    IMPORTANT: Operations are returned in their original order to preserve SSA dependencies.
    """
    # Pattern 1: Quoted operations WITH results: %result = "dialect.op"(%inputs) {attrs} : type
    quoted_with_result_pattern = re.compile(
        r'(%[\w]+(?:,\s*%[\w]+)*)\s*=\s*'  # outputs
        r'"([^"]+)"\s*'  # operation
        r'\(([^)]*)\)\s*'  # inputs
        r'(?:\{([^}]*)\})?\s*'  # attributes (optional)
        r':\s*(?:\([^)]+\)\s*->\s*)?(.+?)(?:\s|$)',  # result type
        re.MULTILINE
    )

    # Pattern 2: Unquoted operations WITH results: %result = dialect.op %inputs {attrs} : type
    unquoted_with_result_pattern = re.compile(
        r'(%[\w]+(?:,\s*%[\w]+)*)\s*=\s*'  # outputs
        r'([\w]+\.[\w]+)\s+'  # operation (dialect.op)
        r'([^{:]+?)\s*'  # inputs
        r'(?:\{([^}]*)\})?\s*'  # attributes (optional)
        r':\s*(.+?)(?:\s|$)',  # result type
        re.MULTILINE
    )

    # Pattern 3: Quoted operations WITHOUT results: "dialect.op"(%inputs) {attrs} : type
    quoted_no_result_pattern = re.compile(
        r'^\s*"([\w]+\.[\w]+)"\s*'  # operation
        r'\(([^)]*)\)\s*'  # inputs
        r'(?:\{([^}]*)\})?\s*'  # attributes (optional)
        r'(?::\s*(.+?))?(?:\s|$)',  # optional type
        re.MULTILINE
    )

    # Pattern 4: Unquoted operations WITHOUT results: dialect.op %inputs {attrs} : type
    unquoted_no_result_pattern = re.compile(
        r'^\s*([\w]+\.[\w]+)\s+'  # operation
        r'([^{:\n]+?)\s*'  # inputs
        r'(?:\{([^}]*)\})?\s*'  # attributes (optional)
        r'(?::\s*(.+?))?(?:\s|$)',  # optional type
        re.MULTILINE
    )

    # Pattern 5: Operations without inputs: %result = dialect.op {attrs} : type
    no_input_pattern = re.compile(
        r'(%[\w]+(?:,\s*%[\w]+)*)\s*=\s*'  # outputs
        r'([\w]+\.[\w]+)\s*'  # operation
        r'(?:\{([^}]*)\})?\s*'  # attributes (optional)
        r':\s*(.+?)(?:\s|$)',  # result type
        re.MULTILINE
    )

    # Pattern 6: Call operations WITHOUT results: call @function_name(%args) : type
    call_no_result_pattern = re.compile(
        r'^\s*(call)\s+'  # call keyword
        r'(@\w+)\s*'  # function reference
        r'\(([^)]*)\)\s*'  # arguments
        r':\s*(.+?)(?:\s|$)',  # type signature
        re.MULTILINE
    )

    # Pattern 7: Call operations WITH results: %result = call @function_name(%args) : type
    call_with_result_pattern = re.compile(
        r'(%[\w]+(?:,\s*%[\w]+)*)\s*=\s*'  # outputs
        r'(call)\s+'  # call keyword
        r'(@\w+)\s*'  # function reference
        r'\(([^)]*)\)\s*'  # arguments
        r':\s*(.+?)(?:\s|$)',  # type signature
        re.MULTILINE
    )

    # Pattern 8: Linalg operations with ins/outs: %result = linalg.op {...} ins(...) outs(...) {...}
    linalg_pattern = re.compile(
        r'(%[\w]+(?:,\s*%[\w]+)*)\s*=\s*'  # outputs
        r'(linalg\.[\w]+)\s+'  # operation (linalg.generic, linalg.matmul, etc.)
        r'(?:\{[^}]+\}\s+)?'  # optional attributes block
        r'ins\(([^)]+)\)\s+'  # ins clause
        r'outs\(([^)]+)\)',  # outs clause
        re.MULTILINE
    )

    # Collect all matches with their positions to preserve order
    all_matches = []

    # Pattern 1: Quoted WITH results
    for match in quoted_with_result_pattern.finditer(mlir_content):
        all_matches.append(('quoted_result', match.start(), match))

    # Pattern 2: Unquoted WITH results
    for match in unquoted_with_result_pattern.finditer(mlir_content):
        all_matches.append(('unquoted_result', match.start(), match))

    # Pattern 3: Quoted WITHOUT results
    for match in quoted_no_result_pattern.finditer(mlir_content):
        all_matches.append(('quoted_no_result', match.start(), match))

    # Pattern 4: Unquoted WITHOUT results
    for match in unquoted_no_result_pattern.finditer(mlir_content):
        all_matches.append(('unquoted_no_result', match.start(), match))

    # Pattern 5: No inputs (constants, allocs, etc.)
    for match in no_input_pattern.finditer(mlir_content):
        all_matches.append(('no_input', match.start(), match))

    # Pattern 6: Call operations without results
    for match in call_no_result_pattern.finditer(mlir_content):
        all_matches.append(('call_no_result', match.start(), match))

    # Pattern 7: Call operations with results
    for match in call_with_result_pattern.finditer(mlir_content):
        all_matches.append(('call_with_result', match.start(), match))

    # Pattern 8: Linalg operations
    for match in linalg_pattern.finditer(mlir_content):
        all_matches.append(('linalg', match.start(), match))

    # Sort by position to preserve original order
    all_matches.sort(key=lambda x: x[1])

    # Process matches in order
    operations = []
    for match_type, _, match in all_matches:
        if match_type == 'quoted_result':
            outputs = [s.strip() for s in match.group(1).split(',')]
            op_type = match.group(2)
            inputs = extract_ssa_values(match.group(3))
            attrs_str = match.group(4) or ""
            result_types_str = match.group(5)
        elif match_type == 'unquoted_result':
            outputs = [s.strip() for s in match.group(1).split(',')]
            op_type = match.group(2)
            inputs_str = match.group(3)
            # Extract clean SSA values (removes array indices, etc.)
            inputs = extract_ssa_values(inputs_str)
            attrs_str = match.group(4) or ""
            result_types_str = match.group(5)
        elif match_type == 'quoted_no_result':
            outputs = []  # No outputs
            op_type = match.group(1)
            inputs = extract_ssa_values(match.group(2))
            attrs_str = match.group(3) or ""
            result_types_str = match.group(4) or ""
        elif match_type == 'unquoted_no_result':
            outputs = []  # No outputs
            op_type = match.group(1)
            inputs_str = match.group(2)
            # Extract clean SSA values (removes array indices, etc.)
            inputs = extract_ssa_values(inputs_str)
            attrs_str = match.group(3) or ""
            result_types_str = match.group(4) or ""
        elif match_type == 'no_input':
            outputs = [s.strip() for s in match.group(1).split(',')]
            op_type = match.group(2)
            inputs = []  # No inputs
            attrs_str = match.group(3) or ""
            result_types_str = match.group(4)
        elif match_type == 'call_no_result':
            outputs = []  # Call operations without result assignments
            op_type = match.group(1)  # "call"
            func_ref = match.group(2)  # "@function_name"
            args_str = match.group(3)  # Arguments
            # Extract clean SSA values from arguments
            inputs = [func_ref]  # Add function reference as first input
            if args_str.strip():
                inputs.extend(extract_ssa_values(args_str))
            attrs_str = ""  # Call operations don't have attributes in this syntax
            result_types_str = match.group(4)
        elif match_type == 'call_with_result':
            outputs = [s.strip() for s in match.group(1).split(',')]
            op_type = match.group(2)  # "call"
            func_ref = match.group(3)  # "@function_name"
            args_str = match.group(4)  # Arguments
            # Extract clean SSA values from arguments
            inputs = [func_ref]  # Add function reference as first input
            if args_str.strip():
                inputs.extend(extract_ssa_values(args_str))
            attrs_str = ""  # Call operations don't have attributes in this syntax
            result_types_str = match.group(5)
        elif match_type == 'linalg':
            outputs = [s.strip() for s in match.group(1).split(',')]
            op_type = match.group(2)  # "linalg.generic", "linalg.matmul", etc.
            ins_str = match.group(3)  # ins clause
            outs_str = match.group(4)  # outs clause
            # Extract inputs from both ins() and outs()
            inputs = extract_ssa_values(ins_str) + extract_ssa_values(outs_str)
            attrs_str = ""  # Attributes are complex for linalg, skip for now
            result_types_str = ""  # Will be inferred from outs

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


@dataclass
class MLIRFunction:
    """Parsed MLIR function."""
    name: str
    inputs: List[Tuple[str, str]]  # (name, type)
    outputs: List[str]  # output value names
    operations: List[MLIROperation]
    body_start: int  # Character position where function body starts
    body_end: int  # Character position where function body ends


def assign_namespaces_to_operations(mlir_content: str, operations: List[MLIROperation], base_namespace: str = "") -> None:
    """
    Assign hierarchical namespaces to operations based on brace depth in the MLIR text.

    This function modifies the operations in-place, setting their namespace field
    based on where they appear in the MLIR text and the current region nesting level.

    Strategy:
    1. Build maps from SSA values and operation types to operations
    2. Walk through MLIR text tracking brace depth
    3. For each line, try to match it to an operation (by assignment OR by dialect.op pattern)
    4. Assign current namespace to matched operations
    5. Track when operations open regions and update namespace stack
    """
    # Build mapping from output values to operations
    value_to_op = {}
    for op in operations:
        for output in op.outputs:
            value_to_op[output] = op

    # Build list of operations not yet assigned (for matching by op_type)
    unassigned_ops = list(operations)

    namespace_stack = [base_namespace]
    depth = 0
    operation_counter = {}

    lines = mlir_content.split('\n')
    for line in lines:
        prev_depth = depth

        # Track brace depth changes
        open_braces = line.count('{')
        close_braces = line.count('}')
        depth_change = open_braces - close_braces

        matched_op = None
        op_type = None

        # Try to match by SSA assignment first
        assignment_match = re.match(r'\s*(%[\w]+(?:,\s*%[\w]+)*)\s*=\s*([\w\.]+)', line)
        if assignment_match:
            first_output = assignment_match.group(1).split(',')[0].strip()
            op_type = assignment_match.group(2)
            if first_output in value_to_op:
                matched_op = value_to_op[first_output]
        else:
            # Try to match operations without assignments (like linalg.yield)
            # Pattern: dialect.op (e.g., "linalg.yield", "scf.yield", "cf.br")
            op_match = re.search(r'\b([\w]+\.[\w]+)\b', line)
            if op_match:
                op_type = op_match.group(1)
                # Find first unassigned operation matching this type
                for op in unassigned_ops:
                    if op.op_type == op_type and not op.namespace:
                        matched_op = op
                        break

        # Assign namespace if we matched an operation
        if matched_op:
            current_namespace = namespace_stack[-1]
            matched_op.namespace = current_namespace

            # Remove from unassigned list
            if matched_op in unassigned_ops:
                unassigned_ops.remove(matched_op)

            # Check if this line opens a region (more open braces than close)
            if depth_change > 0 and op_type:
                op_type_clean = op_type.replace('.', '_')
                operation_counter[op_type_clean] = operation_counter.get(op_type_clean, 0) + 1
                layer_id = f"{op_type_clean}_{operation_counter[op_type_clean]}"

                # Push new namespace for the region contents
                new_namespace = f"{current_namespace}/{layer_id}" if current_namespace else layer_id
                namespace_stack.append(new_namespace)

        # Update depth
        depth += depth_change

        # Pop namespace when exiting regions
        if depth < prev_depth and len(namespace_stack) > 1:
            pops_needed = min(abs(prev_depth - depth), len(namespace_stack) - 1)
            for _ in range(pops_needed):
                namespace_stack.pop()

    # Set default namespace for operations without one (shouldn't happen but safety)
    for op in operations:
        if not op.namespace:
            op.namespace = base_namespace


def parse_functions(mlir_content: str) -> List[MLIRFunction]:
    """
    Parse all functions from MLIR content.
    Returns list of MLIRFunction objects.
    """
    functions = []

    # Pattern to find all function definitions from any dialect (*.func)
    # Matches: func.func, gpu.func, spirv.func, llvm.func, async.func, etc.
    func_pattern = re.compile(
        r'(?:\w+\.func)\s+@(\w+)\s*\(([^)]*)\)[^{]*\{',
        re.MULTILINE
    )

    for func_match in func_pattern.finditer(mlir_content):
        func_name = func_match.group(1)
        inputs_str = func_match.group(2)
        func_start = func_match.end()  # Start of function body

        # Find the corresponding closing brace
        brace_count = 1
        pos = func_start
        func_end = len(mlir_content)

        while pos < len(mlir_content) and brace_count > 0:
            if mlir_content[pos] == '{':
                brace_count += 1
            elif mlir_content[pos] == '}':
                brace_count -= 1
                if brace_count == 0:
                    func_end = pos
                    break
            pos += 1

        # Extract function body
        func_body = mlir_content[func_start:func_end]

        # Parse inputs
        inputs = []
        if inputs_str.strip():
            input_pattern = re.compile(r'(%\w+)\s*:\s*([^,)]+)')
            for input_match in input_pattern.finditer(inputs_str):
                name = input_match.group(1)
                type_str = input_match.group(2).strip()
                inputs.append((name, type_str))

        # Parse outputs from func.return
        outputs = parse_function_outputs(func_body)

        # Parse all operations (including nested ones)
        operations = parse_mlir_operations(func_body)

        # Assign hierarchical namespaces based on brace depth and region nesting
        assign_namespaces_to_operations(func_body, operations, func_name)

        functions.append(MLIRFunction(
            name=func_name,
            inputs=inputs,
            outputs=outputs,
            operations=operations,
            body_start=func_start,
            body_end=func_end
        ))

    # If no functions found, try to parse the whole content as implicit main
    if not functions:
        # Parse all operations
        operations = parse_mlir_operations(mlir_content)

        # Assign hierarchical namespaces
        assign_namespaces_to_operations(mlir_content, operations, "main")

        # Parse inputs inline (fallback for legacy MLIR without func.func)
        func_pattern = re.compile(r'func\.func\s+@\w+\s*\(([^)]+)\)', re.MULTILINE)
        match = func_pattern.search(mlir_content)
        inputs = []
        if match:
            inputs_str = match.group(1)
            input_pattern = re.compile(r'(%\w+)\s*:\s*([^,)]+)')
            for input_match in input_pattern.finditer(inputs_str):
                name = input_match.group(1)
                type_str = input_match.group(2).strip()
                inputs.append((name, type_str))

        outputs = parse_function_outputs(mlir_content)

        functions.append(MLIRFunction(
            name="main",
            inputs=inputs,
            outputs=outputs,
            operations=operations,
            body_start=0,
            body_end=len(mlir_content)
        ))

    return functions


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
    outputs = [s.strip() for s in outputs_str.split(',') if s.strip()]  # Filter out empty strings
    return outputs


def extract_location_info(mlir_line: str) -> str:
    """
    Extract location information from MLIR text.
    Supports basic loc() patterns for stable node naming.

    Examples:
        - loc("name") → "name"
        - loc(fused["a", "b"]) → "a_b"
        - No location → ""
    """
    # Pattern 1: loc("name")
    simple_loc = re.search(r'loc\("([^"]+)"\)', mlir_line)
    if simple_loc:
        return simple_loc.group(1)

    # Pattern 2: loc(fused["name1", "name2", ...])
    fused_loc = re.search(r'loc\(fused\[([^\]]+)\]\)', mlir_line)
    if fused_loc:
        names = re.findall(r'"([^"]+)"', fused_loc.group(1))
        return "_".join(names) if names else ""

    return ""


# Extensibility Hooks: Dialect-specific customization points
class DialectHooks:
    """
    Extensibility hooks for dialect-specific customization.
    Override these methods to customize behavior for specific dialects.
    """

    @staticmethod
    def get_node_name(op_type: str, attributes: Dict[str, str], location_info: str = "") -> str:
        """
        Get custom node name for an operation.

        Args:
            op_type: Operation type (e.g., "tf.Const", "stablehlo.add")
            attributes: Operation attributes
            location_info: Extracted location information

        Returns:
            Custom node name or empty string to use default (op_type)
        """
        # Use location info if available for stable naming
        if location_info:
            return location_info

        # Dialect-specific naming (extensible by adding more cases)
        if op_type.startswith("tf."):
            # TensorFlow dialect: use 'name' attribute if available
            return attributes.get("name", "")
        elif op_type.startswith("tfl."):
            # TensorFlow Lite dialect: use 'name' attribute
            return attributes.get("name", "")

        return ""

    @staticmethod
    def format_attribute_value(key: str, value: str, op_type: str) -> str:
        """
        Format attribute value for display.

        Args:
            key: Attribute key
            value: Attribute value
            op_type: Operation type

        Returns:
            Formatted value string
        """
        # Truncate very long values
        if len(value) > 100:
            return value[:97] + "..."
        return value

    @staticmethod
    def detect_region_ops(op_type: str) -> bool:
        """
        Detect if an operation contains regions.

        Args:
            op_type: Operation type

        Returns:
            True if operation likely contains regions
        """
        region_keywords = ['while', 'if', 'cond', 'reduce', 'map', 'scan', 'loop', 'scf.']
        return any(keyword in op_type.lower() for keyword in region_keywords)


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


def detect_function_calls(operations: List[MLIROperation], function_name: str = "") -> Dict[str, List[str]]:
    """
    Detect function calls in operations and return mapping of node_id to called function names.
    Args:
        operations: List of MLIR operations to scan
        function_name: Name of the function containing these operations (for node_id construction)
    Returns:
        Dict mapping node_id to list of called function names
    """
    calls = {}
    for i, op in enumerate(operations):
        if 'call' in op.op_type.lower():
            # Extract function name from inputs or attributes
            called_funcs = []

            # Check for @function_name in inputs
            for input_val in op.inputs:
                if input_val.startswith('@'):
                    called_funcs.append(input_val[1:])  # Remove @ prefix

            if called_funcs:
                # Construct node_id matching the format used in create_graph_for_function
                node_id = f"{function_name}_op_{i}" if function_name else f"op_{i}"
                calls[node_id] = called_funcs

    return calls


UNKNOWN_DIALECT_PALETTE: List[Tuple[str, str]] = [
    ("#E3F2FD", "#0D47A1"),  # Blue
    ("#E8F5E9", "#1B5E20"),  # Green
    ("#FFF3E0", "#E65100"),  # Orange
    ("#F3E5F5", "#4A148C"),  # Purple
    ("#FCE4EC", "#880E4F"),  # Pink
    ("#F1F8E9", "#33691E"),  # Lime
    ("#E0F7FA", "#006064"),  # Teal
    ("#FFF9C4", "#F57F17"),  # Yellow
    ("#ECEFF1", "#37474F"),  # Blue Grey
]

_unknown_dialect_color_cache: Dict[str, Tuple[str, str]] = {}


def get_unknown_dialect_color(dialect: str) -> Tuple[str, str]:
    """
    Deterministically assign colors to unknown/unregistered dialects.
    Uses a stable SHA-256 hash so assignments are consistent across runs.
    """
    if dialect in _unknown_dialect_color_cache:
        return _unknown_dialect_color_cache[dialect]

    digest = hashlib.sha256(dialect.encode("utf-8")).digest()
    palette_index = digest[0] % len(UNKNOWN_DIALECT_PALETTE)
    color_pair = UNKNOWN_DIALECT_PALETTE[palette_index]
    _unknown_dialect_color_cache[dialect] = color_pair
    return color_pair


def get_dialect_color(op_type: str) -> Tuple[str, str]:
    """
    Get background and text color for an operation based on its dialect.

    Returns:
        Tuple of (backgroundColor, textColor) for the operation's dialect
    """
    # Extract dialect from operation type (e.g., "arith.addf" -> "arith")
    dialect = op_type.split('.')[0] if '.' in op_type else op_type

    # Dialect color mapping (using distinct, accessible colors)
    dialect_colors = {
        # Core MLIR dialects
        "arith": ("#E3F2FD", "#0D47A1"),      # Light blue background, dark blue text
        "func": ("#F3E5F5", "#4A148C"),       # Light purple, dark purple
        "tensor": ("#E8F5E9", "#1B5E20"),     # Light green, dark green
        "linalg": ("#FFF3E0", "#E65100"),     # Light orange, dark orange
        "scf": ("#FCE4EC", "#880E4F"),        # Light pink, dark pink
        "memref": ("#E0F2F1", "#004D40"),     # Light teal, dark teal
        "vector": ("#F1F8E9", "#33691E"),     # Light lime, dark lime
        "affine": ("#FFF8E1", "#F57F17"),     # Light yellow, dark yellow
        "cf": ("#EFEBE9", "#3E2723"),         # Light brown, dark brown
        "gpu": ("#E1F5FE", "#01579B"),        # Light cyan, dark cyan

        # ML/DL dialects
        "tosa": ("#F9FBE7", "#827717"),       # Light lime-yellow, dark olive
        "stablehlo": ("#EDE7F6", "#311B92"),  # Light deep purple, dark deep purple
        "mhlo": ("#E8EAF6", "#1A237E"),       # Light indigo, dark indigo
        "tf": ("#FFEBEE", "#B71C1C"),         # Light red, dark red
        "tfl": ("#FFCDD2", "#C62828"),        # Light red-pink, dark red

        # Transformation dialects
        "transform": ("#FBE9E7", "#BF360C"),  # Light deep orange, dark deep orange
        "pdl": ("#E0F7FA", "#006064"),        # Light cyan-teal, dark cyan-teal

        # Other dialects
        "llvm": ("#ECEFF1", "#263238"),       # Light blue-grey, dark blue-grey
        "spirv": ("#F3E5F5", "#6A1B9A"),      # Light purple, dark purple
        "async": ("#E1BEE7", "#6A1B9A"),      # Light purple, dark purple
        "math": ("#C5E1A5", "#558B2F"),       # Light green, dark green
        "call": ("#FFCCBC", "#D84315"),       # Light deep orange, dark deep orange
    }

    # Return dialect-specific colors or deterministic fallback for unknown dialects
    return dialect_colors.get(dialect, get_unknown_dialect_color(dialect))


def generate_edge_overlays_for_graph(graph: Graph, function_name: str) -> EdgeOverlaysData:
    """
    Generate edge overlays showing tensor shapes and data flows.

    Creates multiple overlay groups:
    1. Tensor shapes - shows all edges with their tensor types
    2. Scalar flow - highlights scalar value edges
    3. Tensor flow - highlights tensor value edges

    Args:
        graph: The graph to generate overlays for
        function_name: Name of the function (for the overlay name)

    Returns:
        EdgeOverlaysData with tensor shape overlays
    """
    tensor_edges = []
    scalar_edges = []
    all_edges_with_shapes = []

    # Extract edges with tensor information from nodes
    for node in graph.nodes:
        for incoming_edge in node.incomingEdges:
            source_node_id = incoming_edge.sourceNodeId
            target_node_id = node.id

            # Find the source node to get output metadata
            source_node = next((n for n in graph.nodes if n.id == source_node_id), None)
            if not source_node:
                continue

            # Get the tensor tag and shape from source output metadata
            output_idx = incoming_edge.sourceNodeOutputId
            tensor_tag = ""
            tensor_shape = ""

            for output_meta in source_node.outputsMetadata:
                if output_meta.id == output_idx:
                    for attr in output_meta.attrs:
                        if attr.key == "__tensor_tag":
                            tensor_tag = attr.value
                        elif attr.key == "tensor_shape":
                            tensor_shape = attr.value
                    break

            # Create edge with label
            edge_label = tensor_shape if tensor_shape else tensor_tag

            edge = EdgeData(
                sourceNodeId=source_node_id,
                targetNodeId=target_node_id,
                label=edge_label
            )

            # Categorize by type
            if edge_label:
                all_edges_with_shapes.append(edge)

                # Check if it's a tensor or scalar
                if "tensor<" in edge_label.lower():
                    tensor_edges.append(edge)
                else:
                    scalar_edges.append(edge)

    # Create overlay groups
    overlays = []

    # Overlay 1: All tensor shapes (blue)
    if all_edges_with_shapes:
        overlays.append(EdgeOverlay(
            name="Tensor Shapes",
            edges=all_edges_with_shapes,
            edgeColor="#4285f4",  # Google Blue
            edgeWidth=2,
            edgeLabelFontSize=7.5
        ))

    # Overlay 2: Tensor flow (green)
    if tensor_edges:
        overlays.append(EdgeOverlay(
            name="Tensor Data Flow",
            edges=tensor_edges,
            edgeColor="#34a853",  # Google Green
            edgeWidth=3,
            edgeLabelFontSize=8.0
        ))

    # Overlay 3: Scalar flow (orange)
    if scalar_edges:
        overlays.append(EdgeOverlay(
            name="Scalar Values",
            edges=scalar_edges,
            edgeColor="#fbbc04",  # Google Yellow
            edgeWidth=2,
            edgeLabelFontSize=7.0
        ))

    return EdgeOverlaysData(
        type="EDGE_OVERLAYS",
        name=f"Tensor Flow - {function_name}",
        overlays=overlays
    )


def create_graph_for_function(
    function: MLIRFunction,
    node_id_offset: int = 0,
    available_functions: List[str] = None
) -> Tuple[Graph, int]:
    """
    Create a Model Explorer Graph for a single function.
    Returns (Graph, next_node_id_offset).
    """
    if available_functions is None:
        available_functions = []

    nodes = []
    node_id_counter = node_id_offset
    value_to_output_node = {}  # Maps value names to (node_id, output_index)

    # Create input nodes
    for i, (input_name, input_type) in enumerate(function.inputs):
        node = GraphNode(
            id=f"{function.name}_input_{i}",
            label="Input",
            namespace=f"{function.name}/Inputs"
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

    # Detect function calls to populate subgraphIds
    function_calls = detect_function_calls(function.operations, function.name)

    # Create operation nodes
    for op_idx, op in enumerate(function.operations):
        node_id = f"{function.name}_op_{op_idx}"

        # Use the namespace stored during hierarchical parsing
        # If not set (legacy), fall back to function name
        namespace = op.namespace if op.namespace else function.name

        # Extract location info for stable naming (basic regex-based)
        location_info = ""  # Could be extracted from full MLIR line if available

        # Use dialect hooks to get custom node name
        custom_name = DialectHooks.get_node_name(op.op_type, op.attributes, location_info)
        node_label = custom_name if custom_name else op.op_type

        node = GraphNode(
            id=node_id,
            label=node_label,
            namespace=namespace
        )

        # Add dialect-based color coding
        bg_color, text_color = get_dialect_color(op.op_type)
        node.style = GraphNodeStyle(
            backgroundColor=bg_color,
            textColor=text_color
        )

        # Add attributes (with formatting hook)
        for key, value in op.attributes.items():
            formatted_value = DialectHooks.format_attribute_value(key, value, op.op_type)
            node.attrs.append(KeyValue(key=key, value=formatted_value))

        # Populate subgraphIds for function calls
        if node_id in function_calls:
            for called_func in function_calls[node_id]:
                if called_func in available_functions:
                    node.subgraphIds.append(called_func)

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

        # Add output metadata (only if operation has outputs)
        if op.outputs:
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
    for i, output_name in enumerate(function.outputs):
        node = GraphNode(
            id=f"{function.name}_output_{i}",
            label="Output",
            namespace=f"{function.name}/Outputs"
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

    # Create graph with nodes
    graph = Graph(id=function.name, nodes=nodes)

    # Generate edge overlays for the graph
    edge_overlays = generate_edge_overlays_for_graph(graph, function.name)
    graph.tasksData.edgeOverlaysDataListLeftPane.append(edge_overlays)

    return graph, node_id_counter + len(nodes)


def create_model_explorer_graphs(
    functions: List[MLIRFunction]
) -> ModelExplorerGraphs:
    """
    Create Model Explorer graphs structure from parsed MLIR functions.
    Returns ModelExplorerGraphs with one graph per function.
    """
    graphs = []
    available_functions = [f.name for f in functions]

    node_id_offset = 0
    for function in functions:
        graph, node_id_offset = create_graph_for_function(
            function,
            node_id_offset,
            available_functions
        )
        graphs.append(graph)

    return ModelExplorerGraphs(graphs=graphs)


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
            # Preprocess: Remove dense constant values to prevent memory issues
            processed_mlir, constants_removed = remove_dense_constant_values(mlir_content)
            if constants_removed > 0:
                print(f"✓ Preprocessed {constants_removed} dense constant(s)", file=sys.stderr)

            # Parse all functions and create multi-graph format
            functions = parse_functions(processed_mlir)

            if not functions:
                # Improved diagnostic: No functions found
                print(json.dumps({
                    "error": "No functions found",
                    "message": "MLIR content contains no func.func definitions. Ensure the file contains valid MLIR function definitions.",
                    "diagnostics": {
                        "file_size_bytes": len(mlir_content),
                        "preprocessing": {
                            "constants_removed": constants_removed
                        }
                    }
                }))
                return 0

            # Create Model Explorer graphs
            model_graphs = create_model_explorer_graphs(functions)

            # Convert to dict format for JSON serialization
            output = asdict(model_graphs)

            # Add diagnostic metadata
            output['_metadata'] = {
                "parser": "regex-based-python",
                "functions_parsed": len(functions),
                "preprocessing": {
                    "constants_removed": constants_removed
                }
            }

            # Output as JSON
            print(json.dumps(output, indent=2))

            return 0

        except ValueError as e:
            # Return structured validation error
            print(json.dumps({
                "error": "MLIR validation failed",
                "message": str(e),
                "diagnostics": {
                    "error_type": "ValueError",
                    "parser": "regex-based-python"
                }
            }))
            return 0
        except re.error as e:
            # Return regex parsing error
            print(json.dumps({
                "error": "MLIR parsing failed",
                "message": f"Regex pattern error: {str(e)}",
                "diagnostics": {
                    "error_type": "RegexError",
                    "parser": "regex-based-python"
                }
            }))
            return 0
        except Exception as e:
            # Return generic parsing error with diagnostic info
            print(json.dumps({
                "error": "MLIR parsing failed",
                "message": str(e),
                "diagnostics": {
                    "error_type": type(e).__name__,
                    "parser": "regex-based-python",
                    "file_size_bytes": len(mlir_content)
                }
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
