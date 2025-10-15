# Model Explorer Interactive Compiler Workflow - API Reference

**Date**: 2025-10-15
**Status**: Design Proposal
**Version**: 2.0
**Author**: Claude (AI Assistant)

📖 **Navigation**: [← Previous: Implementation Plan](04-implementation.md) | [Home: README →](README.md)

---

## Table of Contents

1. [Overview](#overview)
2. [Core APIs](#core-apis)
3. [Annotation APIs](#annotation-apis)
4. [Compiler Plugin APIs](#compiler-plugin-apis)
5. [Validation APIs](#validation-apis)
6. [Export APIs](#export-apis)
7. [Configuration APIs](#configuration-apis)
8. [Events APIs](#events-apis)
9. [Security APIs](#security-apis)
10. [Integration Examples](#integration-examples)

---

## Overview

This document provides complete API specifications for integrating with the **Interactive Compiler Workflow** feature in Model Explorer. It covers:

- Core editing operations
- Annotation management
- Compiler plugin integration
- Validation system
- Export functionality
- Configuration options
- Event system
- Security controls

### API Design Principles

1. **Type Safety**: All APIs use TypeScript for type checking
2. **Immutability**: Operations do not mutate input data
3. **Event-Driven**: State changes emit events for reactive UIs
4. **Extensibility**: Plugin architecture allows custom compiler backends
5. **Validation-First**: All operations are validated before execution

---

## Core APIs

### EditModeManager

Central API for all editing and compilation operations.

```typescript
class EditModeManager {
  // ========================================
  // State Observables
  // ========================================

  /** Observable for edit mode state */
  readonly editMode$: Observable<boolean>;

  /** Observable for dirty state (unsaved changes) */
  readonly isDirty$: Observable<boolean>;

  /** Observable for validation reports */
  readonly validationReport$: Observable<ValidationReport>;

  /** Observable for compilation state */
  readonly compilationState$: Observable<CompilationState>;

  // ========================================
  // Mode Control
  // ========================================

  /**
   * Enable edit mode
   * Creates a snapshot of the current graph for undo/redo
   */
  enableEditMode(): void;

  /**
   * Disable edit mode
   * @param saveChanges - If true, keep changes; if false, revert to original
   */
  disableEditMode(saveChanges?: boolean): void;

  /**
   * Check if currently in edit mode
   */
  isEditMode(): boolean;

  // ========================================
  // Node Operations
  // ========================================

  /**
   * Add a new node to the graph
   * @param type - Node type (e.g., "Conv2D", "MatMul")
   * @param position - Position on canvas { x, y }
   * @param attributes - Optional node attributes
   * @returns Node ID of the created node
   */
  addNode(type: string, position: Point, attributes?: any): string;

  /**
   * Delete a single node
   * @param nodeId - ID of the node to delete
   */
  deleteNode(nodeId: string): void;

  /**
   * Delete multiple nodes
   * @param nodeIds - Array of node IDs to delete
   */
  deleteNodes(nodeIds: string[]): void;

  /**
   * Modify node attributes
   * @param nodeId - ID of the node to modify
   * @param updates - Partial node updates
   */
  modifyNode(nodeId: string, updates: Partial<Node>): void;

  /**
   * Clone a single node
   * @param nodeId - ID of the node to clone
   * @returns Node ID of the cloned node
   */
  cloneNode(nodeId: string): string;

  /**
   * Clone multiple nodes
   * @param nodeIds - Array of node IDs to clone
   * @returns Array of new node IDs
   */
  cloneNodes(nodeIds: string[]): string[];

  /**
   * Rename a node
   * @param nodeId - ID of the node to rename
   * @param newLabel - New label for the node
   */
  renameNode(nodeId: string, newLabel: string): void;

  // ========================================
  // Edge Operations
  // ========================================

  /**
   * Add an edge between two nodes
   * @param source - Source node and port { nodeId, port }
   * @param target - Target node and port { nodeId, port }
   * @returns Edge ID if successful, ValidationError if invalid
   */
  addEdge(source: EdgeEndpoint, target: EdgeEndpoint): string | ValidationError;

  /**
   * Delete a single edge
   * @param edgeId - ID of the edge to delete
   */
  deleteEdge(edgeId: string): void;

  /**
   * Delete multiple edges
   * @param edgeIds - Array of edge IDs to delete
   */
  deleteEdges(edgeIds: string[]): void;

  /**
   * Rewire an edge to a new target
   * @param edgeId - ID of the edge to rewire
   * @param newTarget - New target endpoint
   * @returns Validation result
   */
  rewireEdge(edgeId: string, newTarget: EdgeEndpoint): ValidationResult;

  // ========================================
  // Selection
  // ========================================

  /**
   * Select a node
   * @param nodeId - ID of the node to select
   * @param addToSelection - If true, add to existing selection; if false, replace
   */
  selectNode(nodeId: string, addToSelection?: boolean): void;

  /**
   * Select multiple nodes
   * @param nodeIds - Array of node IDs to select
   */
  selectNodes(nodeIds: string[]): void;

  /**
   * Select an edge
   * @param edgeId - ID of the edge to select
   */
  selectEdge(edgeId: string): void;

  /**
   * Clear all selections
   */
  deselectAll(): void;

  /**
   * Get current selection
   * @returns Object with arrays of selected node and edge IDs
   */
  getSelection(): { nodes: string[], edges: string[] };

  // ========================================
  // History
  // ========================================

  /**
   * Undo the last operation
   * @returns True if undo was successful
   */
  undo(): boolean;

  /**
   * Redo the last undone operation
   * @returns True if redo was successful
   */
  redo(): boolean;

  /**
   * Check if undo is available
   */
  canUndo(): boolean;

  /**
   * Check if redo is available
   */
  canRedo(): boolean;

  /**
   * Get operation history
   * @returns Array of operations
   */
  getHistory(): Operation[];

  /**
   * Clear operation history
   */
  clearHistory(): void;

  // ========================================
  // Validation
  // ========================================

  /**
   * Run full validation on the graph
   * @returns Validation report with errors and warnings
   */
  validate(): ValidationReport;

  /**
   * Validate a potential edge connection
   * @param source - Source endpoint
   * @param target - Target endpoint
   * @returns Validation result
   */
  validateConnection(source: EdgeEndpoint, target: EdgeEndpoint): ValidationResult;

  /**
   * Check if graph has validation errors
   */
  hasErrors(): boolean;

  /**
   * Check if graph has validation warnings
   */
  hasWarnings(): boolean;

  // ========================================
  // Export
  // ========================================

  /**
   * Export the graph to a specified format
   * @param format - Export format (json, onnx, graphdef, mlir)
   * @returns Blob containing the exported graph
   * @throws Error if validation fails
   */
  exportGraph(format: ExportFormat): Blob;

  /**
   * Export a diff patch between original and modified graph
   * @returns Blob containing the patch
   */
  exportPatch(): Blob;

  // ========================================
  // Session
  // ========================================

  /**
   * Manually save the current session to localStorage
   */
  saveSession(): void;

  /**
   * Load a previously saved session
   * @returns True if session was loaded successfully
   */
  loadSession(): boolean;

  /**
   * Clear the saved session
   */
  clearSession(): void;

  /**
   * Configure auto-save
   * @param enabled - Enable or disable auto-save
   * @param intervalSeconds - Auto-save interval in seconds (default: 30)
   */
  autoSave(enabled: boolean, intervalSeconds?: number): void;
}
```

### Type Definitions

```typescript
interface Point {
  x: number;
  y: number;
}

interface EdgeEndpoint {
  nodeId: string;
  port: number;
}

interface ValidationResult {
  isValid: boolean;
  error?: string;
  suggestion?: string;
}

interface ValidationError {
  type: 'error' | 'warning';
  message: string;
  nodeId?: string;
  edgeId?: string;
  annotationId?: string;
  suggestion?: string;
}

interface ValidationReport {
  isValid: boolean;
  hasErrors: boolean;
  hasWarnings: boolean;
  errors: ValidationError[];
  warnings: ValidationError[];
}

type ExportFormat = 'json' | 'onnx' | 'graphdef' | 'mlir' | 'patch';

interface Operation {
  readonly type: string;
  readonly timestamp: Date;
  execute(graph: Graph): void;
  undo(graph: Graph): void;
  getDescription(): string;
}

interface CompilationState {
  inProgress: boolean;
  pluginName: string | null;
  progress: number; // 0-100
  status: 'idle' | 'compiling' | 'success' | 'error';
  error?: string;
}
```

---

## Annotation APIs

### AnnotationManager

Manages optimization hints and constraints for compiler integration.

```typescript
class AnnotationManager {
  // ========================================
  // Annotation CRUD
  // ========================================

  /**
   * Add an annotation to a node
   * @param nodeId - ID of the node to annotate
   * @param annotation - Annotation object
   * @throws Error if annotation is invalid
   */
  addAnnotation(nodeId: string, annotation: Annotation): void;

  /**
   * Remove an annotation by ID
   * @param annotationId - ID of the annotation to remove
   */
  removeAnnotation(annotationId: string): void;

  /**
   * Modify an existing annotation
   * @param annotationId - ID of the annotation to modify
   * @param updates - Partial annotation updates
   */
  modifyAnnotation(annotationId: string, updates: Partial<Annotation>): void;

  /**
   * Get all annotations for a specific node
   * @param nodeId - ID of the node
   * @returns Array of annotations
   */
  getAnnotationsForNode(nodeId: string): Annotation[];

  /**
   * Get all annotations in the graph
   * @returns Array of all annotations
   */
  getAllAnnotations(): Annotation[];

  /**
   * Get a single annotation by ID
   * @param annotationId - ID of the annotation
   * @returns Annotation object or null if not found
   */
  getAnnotation(annotationId: string): Annotation | null;

  /**
   * Clear all annotations
   */
  clearAnnotations(): void;

  // ========================================
  // Batch Operations
  // ========================================

  /**
   * Apply annotation to multiple nodes matching a pattern
   * @param pattern - Pattern to match nodes
   * @param annotation - Annotation to apply
   * @returns Array of affected node IDs
   */
  batchAnnotate(pattern: AnnotationPattern, annotation: Annotation): string[];

  /**
   * Remove all annotations from specified nodes
   * @param nodeIds - Array of node IDs
   */
  clearAnnotationsForNodes(nodeIds: string[]): void;

  // ========================================
  // Validation
  // ========================================

  /**
   * Validate an annotation
   * @param annotation - Annotation to validate
   * @returns Validation result
   */
  validateAnnotation(annotation: Annotation): ValidationResult;

  /**
   * Check if annotations are compatible with a compiler plugin
   * @param annotations - Array of annotations
   * @param pluginName - Name of the compiler plugin
   * @returns Compatibility result
   */
  checkPluginCompatibility(annotations: Annotation[], pluginName: string): CompatibilityResult;
}
```

### Annotation Types

```typescript
interface Annotation {
  /** Unique annotation ID */
  id: string;

  /** Node ID this annotation applies to */
  nodeId: string;

  /** Type of annotation */
  type: 'optimization_hint' | 'constraint' | 'hardware_mapping';

  /** Specific hint type (for optimization hints) */
  hintType?: 'fusion' | 'quantization' | 'layout' | 'scheduling' | 'custom';

  /** Annotation parameters */
  parameters: Record<string, any>;

  /** Priority level */
  priority: 'required' | 'preferred' | 'optional';

  /** Optional human rationale */
  rationale?: string;

  /** Creation timestamp */
  createdAt: Date;
}

interface AnnotationPattern {
  /** Pattern type */
  patternType: 'sequence' | 'subgraph' | 'node_type' | 'custom';

  /** Pattern selector */
  selector: PatternSelector;
}

interface PatternSelector {
  /** For sequence patterns: array of node types in order */
  sequencePattern?: string[];

  /** For subgraph patterns: root node ID */
  subgraphRoot?: string;

  /** For node type patterns: type to match */
  nodeType?: string;

  /** For custom patterns: predicate function */
  predicate?: (node: Node, graph: Graph) => boolean;
}

interface CompatibilityResult {
  isCompatible: boolean;
  incompatibleAnnotations?: string[];
  warnings?: string[];
}
```

---

## Compiler Plugin APIs

### CompilerCoordinator

Orchestrates compiler plugin execution.

```typescript
class CompilerCoordinator {
  /**
   * Select a compiler plugin for use
   * @param pluginName - Name of the plugin to activate
   * @throws Error if plugin not found
   */
  selectPlugin(pluginName: string): void;

  /**
   * Invoke the active compiler plugin
   * @param graph - Graph to compile
   * @param annotations - Array of annotations
   * @param pluginName - Optional plugin name (overrides active plugin)
   * @returns Promise resolving to compilation result
   */
  invokeCompiler(
    graph: Graph,
    annotations: Annotation[],
    pluginName?: string
  ): Promise<CompilationResult>;

  /**
   * Preview transformation without executing
   * @param graph - Graph to preview
   * @param annotations - Array of annotations
   * @returns Promise resolving to transformation preview
   */
  previewTransformation(
    graph: Graph,
    annotations: Annotation[]
  ): Promise<TransformationPreview>;

  /**
   * Get list of available compiler plugins
   * @returns Array of plugin metadata
   */
  getAvailablePlugins(): CompilerPluginMetadata[];

  /**
   * Get performance metrics from last compilation
   * @returns Performance metrics or null
   */
  getPerformanceMetrics(): PerformanceMetrics | null;
}
```

### PluginRegistry

Manages compiler plugin registration.

```typescript
class PluginRegistry {
  /**
   * Register a new compiler plugin
   * @param plugin - Plugin instance
   */
  register(plugin: CompilerPlugin): void;

  /**
   * Unregister a compiler plugin
   * @param pluginName - Name of the plugin
   */
  unregister(pluginName: string): void;

  /**
   * Get a plugin by name
   * @param name - Plugin name
   * @returns Plugin instance or null
   */
  getPlugin(name: string): CompilerPlugin | null;

  /**
   * List all registered plugins
   * @returns Array of plugins
   */
  listPlugins(): CompilerPlugin[];

  /**
   * Check if a plugin is registered
   * @param name - Plugin name
   * @returns True if registered
   */
  hasPlugin(name: string): boolean;
}
```

### CompilerPlugin Interface

Interface that all compiler plugins must implement.

```typescript
interface CompilerPlugin {
  // ========================================
  // Metadata
  // ========================================

  /** Plugin name */
  readonly name: string;

  /** Plugin version */
  readonly version: string;

  /** Plugin description */
  readonly description: string;

  /** Supported annotation types */
  readonly supportedAnnotations: string[];

  // ========================================
  // Capability Checking
  // ========================================

  /**
   * Check if plugin can handle the given graph and annotations
   * @param graph - Graph to compile
   * @param annotations - Annotations to apply
   * @returns True if plugin can handle
   */
  canHandle(graph: Graph, annotations: Annotation[]): boolean;

  // ========================================
  // Compilation
  // ========================================

  /**
   * Preview transformation without executing
   * @param graph - Graph to preview
   * @param annotations - Annotations to apply
   * @returns Transformation preview
   */
  preview(graph: Graph, annotations: Annotation[]): TransformationPreview;

  /**
   * Compile the graph with annotations
   * @param graph - Graph to compile
   * @param annotations - Annotations to apply
   * @returns Promise resolving to compilation result
   */
  compile(graph: Graph, annotations: Annotation[]): Promise<CompilationResult>;

  /**
   * Estimate performance impact of compilation
   * @param graph - Original graph
   * @param result - Compilation result
   * @returns Promise resolving to performance metrics
   */
  estimatePerformance(
    graph: Graph,
    result: CompilationResult
  ): Promise<PerformanceMetrics>;
}
```

### Compiler Types

```typescript
interface CompilerPluginMetadata {
  name: string;
  version: string;
  description: string;
  supportedAnnotations: string[];
  author?: string;
  homepage?: string;
}

interface CompilationResult {
  /** Compilation success status */
  success: boolean;

  /** Transformed graph */
  transformedGraph: Graph;

  /** Array of applied transformations */
  appliedTransformations: Transformation[];

  /** Performance metrics */
  performanceMetrics: PerformanceMetrics;

  /** Warnings (non-fatal) */
  warnings: string[];

  /** Errors (if success is false) */
  errors?: string[];
}

interface Transformation {
  /** Transformation type */
  type: string;

  /** Description */
  description: string;

  /** Affected node IDs */
  affectedNodes: string[];

  /** Transformation parameters */
  parameters: Record<string, any>;
}

interface TransformationPreview {
  /** Node changes */
  nodeChanges: {
    added: Node[];
    removed: Node[];
    modified: Node[];
  };

  /** Edge changes */
  edgeChanges: {
    added: Edge[];
    removed: Edge[];
  };

  /** Estimated impact */
  estimatedImpact: {
    nodeReduction: number;
    speedupFactor: number;
    memoryReduction: number;
  };
}

interface PerformanceMetrics {
  /** Node count before compilation */
  nodeCountBefore: number;

  /** Node count after compilation */
  nodeCountAfter: number;

  /** Node reduction percentage */
  nodeReductionPercent: number;

  /** Estimated speedup factor */
  estimatedSpeedup: number;

  /** Memory reduction in bytes */
  memoryReduction: number;

  /** Per-layer latency estimates */
  layerLatencies: Map<string, number>;
}
```

---

## Validation APIs

### GraphValidator

Comprehensive graph validation system.

```typescript
class GraphValidator {
  /**
   * Validate entire graph
   * @param graph - Graph to validate
   * @returns Validation report
   */
  validate(graph: Graph): ValidationReport;

  /**
   * Validate annotations
   * @param annotations - Annotations to validate
   * @returns Validation report
   */
  validateAnnotations(annotations: Annotation[]): ValidationReport;

  /**
   * Validate compiled graph against original
   * @param original - Original graph
   * @param compiled - Compiled graph
   * @returns Validation report
   */
  validateCompiledGraph(original: Graph, compiled: Graph): ValidationReport;

  /**
   * Validate a potential edge connection
   * @param sourceNode - Source node
   * @param sourcePort - Source port index
   * @param targetNode - Target node
   * @param targetPort - Target port index
   * @returns Validation result
   */
  validateConnection(
    sourceNode: Node,
    sourcePort: number,
    targetNode: Node,
    targetPort: number
  ): ValidationResult;
}
```

### Validation Rules

Custom validation rules can be added to extend the validator.

```typescript
interface ValidationRule {
  /** Rule name */
  name: string;

  /** Validate and return errors */
  validate(graph: Graph): ValidationError[];
}

// Example: Node validation rule
class RequiredInputsRule implements ValidationRule {
  name = 'required-inputs';

  validate(graph: Graph): ValidationError[] {
    const errors: ValidationError[] = [];

    graph.nodes.forEach(node => {
      node.inputs.forEach((input, index) => {
        if (input.required) {
          const hasConnection = graph.edges.some(e =>
            e.targetNodeId === node.id && e.targetPort === index
          );

          if (!hasConnection) {
            errors.push({
              type: 'error',
              message: `Node '${node.label}' missing required input '${input.name}'`,
              nodeId: node.id,
              suggestion: `Connect to output port of type '${input.type}'`
            });
          }
        }
      });
    });

    return errors;
  }
}

// Example: Edge validation rule
class EdgeTypeCompatibilityRule implements ValidationRule {
  name = 'edge-type-compatibility';

  validate(graph: Graph): ValidationError[] {
    const errors: ValidationError[] = [];

    graph.edges.forEach(edge => {
      const sourceNode = graph.getNode(edge.sourceNodeId);
      const targetNode = graph.getNode(edge.targetNodeId);

      const sourceOutput = sourceNode.outputs[edge.sourcePort];
      const targetInput = targetNode.inputs[edge.targetPort];

      if (!this.areTypesCompatible(sourceOutput.type, targetInput.type)) {
        errors.push({
          type: 'error',
          message: `Type mismatch: ${sourceOutput.type} → ${targetInput.type}`,
          edgeId: edge.id,
          suggestion: 'Add type conversion node or choose compatible ports'
        });
      }
    });

    return errors;
  }

  private areTypesCompatible(sourceType: string, targetType: string): boolean {
    if (sourceType === targetType) return true;
    if (targetType === 'any' || sourceType === 'any') return true;
    return false;
  }
}
```

### Shape Inference

```typescript
class ShapeInference {
  /**
   * Infer shapes for all tensors in the graph
   * @param graph - Graph to infer shapes for
   * @returns Map of tensor shapes (nodeId:portIndex -> shape)
   */
  inferShapes(graph: Graph): Map<string, TensorShape>;

  /**
   * Validate tensor shapes in the graph
   * @param graph - Graph to validate
   * @returns Array of shape validation errors
   */
  validateShapes(graph: Graph): ValidationError[];
}

type TensorShape = number[];  // e.g., [1, 224, 224, 3]
```

---

## Export APIs

### Exporter Interface

```typescript
interface Exporter {
  /**
   * Export a graph to specific format
   * @param graph - Graph to export
   * @returns Blob containing exported data
   */
  export(graph: Graph): Blob;
}
```

### JSON Exporter

```typescript
class JsonExporter implements Exporter {
  export(graph: Graph): Blob {
    const json = {
      version: '1.0',
      modelExplorerFormat: true,
      graphs: [{
        id: graph.id,
        nodes: graph.nodes.map(node => ({
          id: node.id,
          type: node.type,
          label: node.label,
          namespace: node.namespace,
          attributes: node.attributes,
          inputs: node.inputs,
          outputs: node.outputs,
          metadata: {
            position: node.position,
            isGraphInput: node.isGraphInput,
            isGraphOutput: node.isGraphOutput
          }
        })),
        edges: graph.edges.map(edge => ({
          id: edge.id,
          sourceNodeId: edge.sourceNodeId,
          sourcePort: edge.sourcePort,
          targetNodeId: edge.targetNodeId,
          targetPort: edge.targetPort,
          label: edge.label
        }))
      }]
    };

    const jsonString = JSON.stringify(json, null, 2);
    return new Blob([jsonString], { type: 'application/json' });
  }
}
```

### ONNX Exporter

```typescript
class OnnxExporter implements Exporter {
  /**
   * Export graph to ONNX format
   * @param graph - Graph to export
   * @returns Blob containing ONNX protobuf
   */
  export(graph: Graph): Blob;

  /**
   * Map Model Explorer node type to ONNX op type
   * @param nodeType - Model Explorer node type
   * @returns ONNX op type
   */
  private mapToOnnxOpType(nodeType: string): string;
}
```

### Patch Exporter

```typescript
class PatchExporter {
  /**
   * Export diff patch between two graphs
   * @param original - Original graph
   * @param modified - Modified graph
   * @returns Blob containing JSON patch
   */
  exportDiff(original: Graph, modified: Graph): Blob;

  /**
   * Apply a patch to a graph
   * @param graph - Graph to apply patch to
   * @param patch - Patch data
   * @returns Modified graph
   */
  applyPatch(graph: Graph, patch: any): Graph;
}

interface GraphPatch {
  version: string;
  format: 'model-explorer-patch';
  timestamp: string;

  addedNodes: Node[];
  deletedNodes: string[];  // Node IDs
  modifiedNodes: NodeModification[];

  addedEdges: Edge[];
  deletedEdges: string[];  // Edge IDs

  stats: {
    nodesAdded: number;
    nodesDeleted: number;
    nodesModified: number;
    edgesAdded: number;
    edgesDeleted: number;
  };
}

interface NodeModification {
  id: string;
  changes: {
    attributes?: { old: any; new: any };
    label?: { old: string; new: string };
  };
}
```

---

## Configuration APIs

### VisualizerConfig

```typescript
interface VisualizerConfig {
  // ... existing properties

  /** Edit mode configuration */
  editConfig?: EditConfig;
}

interface EditConfig {
  /** Enable edit mode features */
  enabled: boolean;

  /** Start in edit mode by default */
  startInEditMode?: boolean;

  /** Auto-save interval in seconds (0 = disabled) */
  autoSaveInterval?: number;

  /** Maximum undo/redo stack size */
  maxHistorySize?: number;

  /** Validation level */
  validationLevel?: 'strict' | 'lenient' | 'off';

  /** Show validation panel by default */
  showValidationPanel?: boolean;

  /** Available node types in palette */
  nodeTypes?: string[];

  /** Available export formats */
  exportFormats?: ExportFormat[];

  /** Custom validation rules */
  customValidators?: ValidationRule[];

  /** Compiler configuration */
  compilerConfig?: CompilerConfig;
}

interface CompilerConfig {
  /** Enable compiler integration */
  enabled: boolean;

  /** Available compiler plugins */
  availablePlugins?: string[];

  /** Default compiler plugin */
  defaultPlugin?: string;

  /** Compilation timeout in seconds */
  compilationTimeout?: number;

  /** Enable compilation caching */
  enableCaching?: boolean;

  /** Cache TTL in seconds */
  cacheTTL?: number;
}
```

---

## Events APIs

### Event Types

```typescript
interface EditEvents {
  // ========================================
  // Mode Events
  // ========================================

  'edit-mode-enabled': void;
  'edit-mode-disabled': void;

  // ========================================
  // Graph Change Events
  // ========================================

  'node-added': { nodeId: string, node: Node };
  'node-deleted': { nodeId: string };
  'node-modified': { nodeId: string, changes: Partial<Node> };
  'edge-added': { edgeId: string, edge: Edge };
  'edge-deleted': { edgeId: string };
  'graph-changed': void;

  // ========================================
  // Selection Events
  // ========================================

  'selection-changed': { nodes: string[], edges: string[] };

  // ========================================
  // Annotation Events
  // ========================================

  'annotation-added': { annotationId: string, annotation: Annotation };
  'annotation-removed': { annotationId: string };
  'annotation-modified': { annotationId: string, changes: Partial<Annotation> };

  // ========================================
  // Compilation Events
  // ========================================

  'compilation-started': { pluginName: string };
  'compilation-progress': { progress: number };  // 0-100
  'compilation-completed': { result: CompilationResult };
  'compilation-failed': { error: Error };

  // ========================================
  // Validation Events
  // ========================================

  'validation-started': void;
  'validation-completed': ValidationReport;

  // ========================================
  // History Events
  // ========================================

  'operation-executed': Operation;
  'operation-undone': Operation;
  'operation-redone': Operation;

  // ========================================
  // Export Events
  // ========================================

  'export-started': { format: ExportFormat };
  'export-completed': { format: ExportFormat, blob: Blob };
  'export-failed': { format: ExportFormat, error: Error };
}
```

### Event Subscription

```typescript
class EventBus {
  /**
   * Subscribe to an event
   * @param eventName - Name of the event
   * @param handler - Event handler function
   * @returns Unsubscribe function
   */
  on<K extends keyof EditEvents>(
    eventName: K,
    handler: (data: EditEvents[K]) => void
  ): () => void;

  /**
   * Emit an event
   * @param eventName - Name of the event
   * @param data - Event data
   */
  emit<K extends keyof EditEvents>(
    eventName: K,
    data: EditEvents[K]
  ): void;
}
```

---

## Security APIs

### SecurityValidator

```typescript
class SecurityValidator {
  /**
   * Validate node label for XSS and injection attacks
   * @param label - Node label
   * @returns Validation result
   */
  validateNodeLabel(label: string): ValidationResult;

  /**
   * Validate attribute value
   * @param value - Attribute value
   * @param expectedType - Expected type
   * @returns Validation result
   */
  validateAttributeValue(value: any, expectedType: string): ValidationResult;

  /**
   * Validate graph before export
   * @param graph - Graph to validate
   * @returns Validation result
   */
  validateGraphForExport(graph: Graph): ValidationResult;

  /**
   * Sanitize HTML in user input
   * @param input - Input string
   * @returns Sanitized string
   */
  sanitizeHtml(input: string): string;
}
```

### PermissionManager

```typescript
interface EditPermissions {
  /** Can add new nodes */
  canAddNodes: boolean;

  /** Can delete nodes */
  canDeleteNodes: boolean;

  /** Can modify node attributes */
  canModifyNodes: boolean;

  /** Can add/delete edges */
  canModifyEdges: boolean;

  /** Can add annotations */
  canAddAnnotations: boolean;

  /** Can invoke compiler */
  canCompile: boolean;

  /** Can export graphs */
  canExport: boolean;

  /** Can import graphs */
  canImport: boolean;

  /** Maximum nodes in graph */
  maxNodes?: number;

  /** Allowed node types */
  allowedNodeTypes?: string[];

  /** Allowed export formats */
  allowedExportFormats?: ExportFormat[];

  /** Allowed compiler plugins */
  allowedCompilerPlugins?: string[];
}

class PermissionManager {
  constructor(permissions: EditPermissions);

  /**
   * Check if user has permission for an action
   * @param action - Permission to check
   * @returns True if permitted
   */
  checkPermission(action: keyof EditPermissions): boolean;

  /**
   * Check if user can add a specific node type
   * @param type - Node type
   * @returns True if permitted
   */
  canAddNode(type: string): boolean;

  /**
   * Check if user can export in a specific format
   * @param format - Export format
   * @returns True if permitted
   */
  canExport(format: ExportFormat): boolean;

  /**
   * Check if user can use a compiler plugin
   * @param pluginName - Plugin name
   * @returns True if permitted
   */
  canUseCompiler(pluginName: string): boolean;
}
```

---

## Integration Examples

### Basic Editing Workflow

```typescript
// Initialize edit mode manager
const editManager = new EditModeManager(appService, webglRenderer);

// Enable edit mode
editManager.enableEditMode();

// Add a node
const nodeId = editManager.addNode('Conv2D', { x: 100, y: 100 }, {
  kernel_size: [3, 3],
  stride: [1, 1],
  padding: 'same',
  filters: 64
});

// Add an edge
editManager.addEdge(
  { nodeId: 'input', port: 0 },
  { nodeId: nodeId, port: 0 }
);

// Validate
const report = editManager.validate();
if (report.hasErrors) {
  console.error('Validation errors:', report.errors);
}

// Export
if (!report.hasErrors) {
  const blob = editManager.exportGraph('json');
  // Trigger download...
}
```

### Annotation and Compilation Workflow

```typescript
// Initialize managers
const editManager = new EditModeManager(appService, webglRenderer);
const annotationManager = new AnnotationManager();
const compilerCoordinator = new CompilerCoordinator();

editManager.enableEditMode();

// Add annotations to nodes
const fusionAnnotation: Annotation = {
  id: 'ann1',
  nodeId: 'conv_5',
  type: 'optimization_hint',
  hintType: 'fusion',
  parameters: {
    fusionStrategy: 'conv_bn_relu',
    preservePrecision: true
  },
  priority: 'preferred',
  rationale: 'Reduces memory bandwidth by 40%',
  createdAt: new Date()
};

annotationManager.addAnnotation('conv_5', fusionAnnotation);

// Batch annotate a pattern
const pattern: AnnotationPattern = {
  patternType: 'sequence',
  selector: {
    sequencePattern: ['Conv2D', 'BatchNorm', 'ReLU']
  }
};

const affectedNodes = annotationManager.batchAnnotate(pattern, {
  id: 'batch_ann1',
  nodeId: '', // Will be set by batch operation
  type: 'optimization_hint',
  hintType: 'fusion',
  parameters: { fusionStrategy: 'conv_bn_relu' },
  priority: 'preferred',
  createdAt: new Date()
});

console.log(`Applied fusion annotation to ${affectedNodes.length} node sequences`);

// Select compiler plugin
compilerCoordinator.selectPlugin('TVM');

// Compile with annotations
const result = await compilerCoordinator.invokeCompiler(
  editManager.getGraph(),
  annotationManager.getAllAnnotations()
);

if (result.success) {
  console.log('Compilation successful!');
  console.log(`Node reduction: ${result.performanceMetrics.nodeReductionPercent}%`);
  console.log(`Estimated speedup: ${result.performanceMetrics.estimatedSpeedup}x`);

  // Display compiled graph in diff viewer
  displayDiffView(editManager.getGraph(), result.transformedGraph);
} else {
  console.error('Compilation failed:', result.errors);
}
```

### Custom Compiler Plugin

```typescript
class MyCustomPlugin implements CompilerPlugin {
  name = 'My Custom Optimizer';
  version = '1.0.0';
  description = 'Custom optimization plugin';
  supportedAnnotations = ['fusion', 'quantization'];

  canHandle(graph: Graph, annotations: Annotation[]): boolean {
    // Check if all annotations are supported
    return annotations.every(ann =>
      this.supportedAnnotations.includes(ann.hintType || '')
    );
  }

  preview(graph: Graph, annotations: Annotation[]): TransformationPreview {
    // Analyze and return preview
    return {
      nodeChanges: {
        added: [],
        removed: [],
        modified: []
      },
      edgeChanges: {
        added: [],
        removed: []
      },
      estimatedImpact: {
        nodeReduction: 0,
        speedupFactor: 1.0,
        memoryReduction: 0
      }
    };
  }

  async compile(graph: Graph, annotations: Annotation[]): Promise<CompilationResult> {
    // Parse annotations
    const fusionAnnotations = annotations.filter(a => a.hintType === 'fusion');

    // Apply transformations
    const transformedGraph = this.applyOptimizations(graph, fusionAnnotations);

    // Generate metrics
    const metrics = this.calculateMetrics(graph, transformedGraph);

    return {
      success: true,
      transformedGraph,
      appliedTransformations: [],
      performanceMetrics: metrics,
      warnings: []
    };
  }

  async estimatePerformance(
    graph: Graph,
    result: CompilationResult
  ): Promise<PerformanceMetrics> {
    // Calculate performance metrics
    return {
      nodeCountBefore: graph.nodes.length,
      nodeCountAfter: result.transformedGraph.nodes.length,
      nodeReductionPercent: 20,
      estimatedSpeedup: 1.5,
      memoryReduction: 1024 * 1024,  // 1 MB
      layerLatencies: new Map()
    };
  }

  private applyOptimizations(graph: Graph, annotations: Annotation[]): Graph {
    // Implementation...
    return graph;
  }

  private calculateMetrics(original: Graph, optimized: Graph): PerformanceMetrics {
    // Implementation...
    return {
      nodeCountBefore: original.nodes.length,
      nodeCountAfter: optimized.nodes.length,
      nodeReductionPercent: 0,
      estimatedSpeedup: 1.0,
      memoryReduction: 0,
      layerLatencies: new Map()
    };
  }
}

// Register plugin
const registry = new PluginRegistry();
registry.register(new MyCustomPlugin());
```

### Event Handling

```typescript
const eventBus = new EventBus();

// Subscribe to compilation events
const unsubscribe = eventBus.on('compilation-completed', (data) => {
  console.log('Compilation completed!');
  console.log(`Speedup: ${data.result.performanceMetrics.estimatedSpeedup}x`);
});

// Subscribe to validation events
eventBus.on('validation-completed', (report) => {
  if (report.hasErrors) {
    displayValidationErrors(report.errors);
  }
});

// Later: unsubscribe
unsubscribe();
```

---

## Related Documents

- **[Overview](00-overview.md)**: Executive summary and motivation
- **[Interactive Compiler Workflow](01-interactive-compiler-workflow.md)**: Core feature specification and compiler integration examples
- **[User Experience Design](02-user-experience.md)**: User stories, workflows, and UI/UX design
- **[Technical Architecture](03-architecture.md)**: System design and implementation details
- **[Implementation Plan](04-implementation.md)**: Phase-by-phase development roadmap

---

**Document Metadata**:

- **Version**: 2.0 (focus on interactive compiler workflows)
- **Last Updated**: 2025-10-15
- **Target Audience**: Plugin Developers, Integration Engineers, API Consumers
- **Prerequisites**: Familiarity with TypeScript and [Technical Architecture](03-architecture.md)

📖 **Navigation**: [← Previous: Implementation Plan](04-implementation.md) | [Home: README →](README.md)
