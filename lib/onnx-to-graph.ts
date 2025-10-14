/**
 * ONNX to Model Explorer Graph Converter
 *
 * This module converts ONNX (Open Neural Network Exchange) models
 * into a graph format compatible with Google's Model Explorer visualization tool.
 */

import onnxProto from 'onnx-proto'
import { execFileSync } from 'child_process'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'

const { onnx } = onnxProto
const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

export interface GraphNode {
	id: string
	label: string
	namespace: string
	attrs: Array<{key: string; value: string}>
	outputsMetadata?: Array<{
		id: string
		attrs: Array<{key: string; value: string}>
	}>
	inputsMetadata?: Array<{
		id: string
		attrs: Array<{key: string; value: string}>
	}>
	incomingEdges: Array<{
		sourceNodeId: string
		sourceNodeOutputId?: string
		targetNodeInputId?: string
	}>
}

export interface ModelExplorerGraph {
	id: string
	nodes: GraphNode[]
}

/**
 * Format tensor type and shape for display
 * @param type The tensor element type (ONNX DataType enum)
 * @param shape The tensor shape dimensions
 * @returns Formatted string like "float32[1x3x224x224]"
 */
function formatTensorShape(type: number | undefined, shape: any): string {
	// ONNX tensor element types mapping (from onnx.proto TensorProto.DataType)
	const typeMap: Record<number, string> = {
		0: 'UNDEFINED',
		1: 'float32',
		2: 'uint8',
		3: 'int8',
		4: 'uint16',
		5: 'int16',
		6: 'int32',
		7: 'int64',
		8: 'string',
		9: 'bool',
		10: 'float16',
		11: 'float64',
		12: 'uint32',
		13: 'uint64',
		14: 'complex64',
		15: 'complex128',
		16: 'bfloat16'
	}

	const typeName = type !== undefined ? (typeMap[type] || `type${type}`) : 'unknown'

	// Handle shape - it may be a TensorShapeProto with dim array
	let shapeStr = '[]'
	if (shape?.dim && Array.isArray(shape.dim)) {
		const dims = shape.dim.map((d: any) => {
			// Each dim has dimValue (number) or dimParam (string) - camelCase from protobuf
			if (d.dimValue !== undefined && d.dimValue !== null) {
				return String(d.dimValue)
			} else if (d.dimParam) {
				return d.dimParam
			}
			return '?'
		})
		if (dims.length > 0) {
			shapeStr = `[${dims.join('x')}]`
		}
	}

	return `${typeName}${shapeStr}`
}

/**
 * Get attribute value as string
 */
function getAttributeValue(attr: any): string {
	if (attr.i !== undefined && attr.i !== null) return String(attr.i)
	if (attr.f !== undefined && attr.f !== null) return String(attr.f)
	if (attr.s !== undefined && attr.s !== null) {
		// s is a buffer, convert to string
		return attr.s.toString('utf8')
	}
	if (attr.ints && attr.ints.length > 0) return `[${attr.ints.join(', ')}]`
	if (attr.floats && attr.floats.length > 0) return `[${attr.floats.join(', ')}]`
	if (attr.strings && attr.strings.length > 0) {
		return `[${attr.strings.map((s: Buffer) => s.toString('utf8')).join(', ')}]`
	}
	if (attr.t) return 'tensor'
	if (attr.g) return 'graph'
	return 'unknown'
}

/**
 * Build tensor shape map from ONNX protobuf annotations only
 * Shape inference is handled by Python's official ONNX library
 */
function buildTensorShapeMap(graph: any): Map<string, string> {
	const shapeMap = new Map<string, string>()

	// Helper to format dimensions from initializers
	const formatShapeFromDims = (dims: any[]): string => {
		if (!dims || dims.length === 0) return '[]'
		const dimStrs = dims.map(d => {
			// Handle protobuf Long objects
			if (d && typeof d === 'object' && 'low' in d) {
				return String(d.low)
			}
			return String(d)
		})
		return `[${dimStrs.join('x')}]`
	}

	// Add shapes from initializers (weights/constants)
	if (graph.initializer) {
		graph.initializer.forEach((init: any) => {
			if (init.name && init.dims) {
				const shape = formatShapeFromDims(init.dims)
				const type = init.dataType !== undefined ? init.dataType : undefined
				const typeMap: Record<number, string> = {
					1: 'float32', 2: 'uint8', 3: 'int8', 6: 'int32', 7: 'int64',
					10: 'float16', 11: 'float64'
				}
				const typeName = type !== undefined ? (typeMap[type] || `type${type}`) : ''
				shapeMap.set(init.name, typeName ? `${typeName}${shape}` : shape)
			}
		})
	}

	// Add shapes from value_info (intermediate tensors with shape annotations from Python inference)
	if (graph.valueInfo) {
		graph.valueInfo.forEach((vi: any) => {
			if (vi.name && vi.type?.tensorType) {
				const elemType = vi.type.tensorType.elemType
				const shape = vi.type.tensorType.shape
				shapeMap.set(vi.name, formatTensorShape(elemType, shape))
			}
		})
	}

	// Add shapes from graph inputs
	if (graph.input) {
		graph.input.forEach((input: any) => {
			if (input.name && input.type?.tensorType) {
				const elemType = input.type.tensorType.elemType
				const shape = input.type.tensorType.shape
				shapeMap.set(input.name, formatTensorShape(elemType, shape))
			}
		})
	}

	// Add shapes from graph outputs
	if (graph.output) {
		graph.output.forEach((output: any) => {
			if (output.name && output.type?.tensorType) {
				const elemType = output.type.tensorType.elemType
				const shape = output.type.tensorType.shape
				shapeMap.set(output.name, formatTensorShape(elemType, shape))
			}
		})
	}

	return shapeMap
}

/**
 * Try to enrich ONNX model with shape inference using Python's official ONNX library
 * Falls back to original buffer if Python is not available or inference fails
 *
 * @param onnxBuffer The original ONNX model buffer
 * @returns Enriched model buffer with inferred shapes, or original buffer if failed
 */
function tryPythonShapeInference(onnxBuffer: Buffer): Buffer {
	try {
		// Path to Python script (relative to compiled JS location in dist/)
		const scriptPath = join(__dirname, '..', 'scripts', 'infer_onnx_shapes.py')

		// Run Python shape inference using official ONNX library
		// Use python3 explicitly as it's more commonly available
		const enrichedBuffer = execFileSync('python3', [scriptPath], {
			input: onnxBuffer,
			maxBuffer: 500 * 1024 * 1024, // 500MB max buffer for large models (enriched models can be much larger)
			timeout: 60000, // 60 second timeout for large models
			encoding: 'buffer' // Return as Buffer not string
		})

		console.log('✓ Python ONNX shape inference successful')
		return enrichedBuffer

	} catch (error: any) {
		// Python not available or inference failed - return original model
		// Shape information will be limited to what's already in the model
		if (error.code === 'ENOENT') {
			console.log('⚠ Python3 not found - shape information may be incomplete')
		} else if (error.status !== undefined) {
			console.log(`⚠ Python shape inference failed (exit code ${error.status}) - shape information may be incomplete`)
		} else {
			console.log('⚠ Python shape inference error - shape information may be incomplete')
		}
		return onnxBuffer // Return original buffer unchanged
	}
}

/**
 * Convert ONNX model buffer to Model Explorer graph format
 * @param onnxBuffer The ONNX model as a Buffer
 * @param filename The filename to use as the graph ID
 * @returns A graph object compatible with Model Explorer
 */
export async function convertONNXToGraph(onnxBuffer: Buffer, filename: string): Promise<ModelExplorerGraph> {
	const nodes: GraphNode[] = []
	const valueToProducer = new Map<string, {nodeId: string; outputIndex: number}>()

	try {
		// Try to enrich model with Python shape inference first
		// This will populate value_info with all intermediate tensor shapes
		const enrichedBuffer = tryPythonShapeInference(onnxBuffer)

		// Decode the ONNX model protobuf (either enriched or original)
		const modelProto = onnx.ModelProto.decode(enrichedBuffer)
		const graph = modelProto.graph

		if (!graph) {
			throw new Error('No graph found in ONNX model')
		}

		// Build tensor shape map from ONNX protobuf annotations
		// If Python inference succeeded, value_info will contain all intermediate tensor shapes
		const tensorShapeMap = buildTensorShapeMap(graph)

		let nodeIdCounter = 0

		// Process graph inputs (these are the model's external inputs)
		if (graph.input) {
			graph.input.forEach((valueInfo: any, idx: number) => {
				const inputNodeId = `input_${nodeIdCounter++}`
				const inputName = valueInfo.name || `input_${idx}`

				// Get type and shape information (using camelCase from protobuf)
				const tensorType = valueInfo.type?.tensorType
				const elemType = tensorType?.elemType
				const shape = tensorType?.shape
				const shapeStr = formatTensorShape(elemType, shape)

				// Build label with shape information
				let label = 'Input'
				if (shapeStr && shapeStr !== 'unknown[]') {
					label += `\n${shapeStr}`
				}

				const node: GraphNode = {
					id: inputNodeId,
					label: label,
					namespace: 'io',
					attrs: [
						{ key: 'name', value: inputName }
					],
					outputsMetadata: [{
						id: '0',
						attrs: [{
							key: 'tensor_shape',
							value: shapeStr
						}]
					}],
					incomingEdges: []
				}

				nodes.push(node)
				valueToProducer.set(inputName, { nodeId: inputNodeId, outputIndex: 0 })
			})
		}

		// Process all computation nodes
		if (graph.node) {
			graph.node.forEach((onnxNode: any, idx: number) => {
				const opNodeId = `node_${nodeIdCounter++}`
				const opType = onnxNode.opType || 'Unknown'
				const nodeName = onnxNode.name || `${opType}_${idx}`

				// Build attributes array
				const attrs: Array<{key: string; value: string}> = [
					{ key: 'op_type', value: opType },
					{ key: 'name', value: nodeName }
				]

				// Add node attributes
				if (onnxNode.attribute) {
					onnxNode.attribute.forEach((attr: any) => {
						attrs.push({
							key: attr.name || 'attr',
							value: getAttributeValue(attr)
						})
					})
				}

				// Build incoming edges from inputs
				const incomingEdges: Array<{
					sourceNodeId: string
					sourceNodeOutputId?: string
					targetNodeInputId?: string
				}> = []

				if (onnxNode.input) {
					onnxNode.input.forEach((inputName: string, inputIdx: number) => {
						const producer = valueToProducer.get(inputName)
						if (producer) {
							incomingEdges.push({
								sourceNodeId: producer.nodeId,
								sourceNodeOutputId: String(producer.outputIndex),
								targetNodeInputId: String(inputIdx)
							})
						}
					})
				}

				// Build inputs metadata with shape information
				const inputsMetadata = onnxNode.input ? onnxNode.input.map((inputName: string, inputIdx: number) => {
					const attrs: Array<{key: string; value: string}> = [{ key: 'name', value: inputName }]
					const shapeInfo = tensorShapeMap.get(inputName)
					if (shapeInfo) {
						attrs.push({ key: 'tensor_shape', value: shapeInfo })
					}
					return {
						id: String(inputIdx),
						attrs
					}
				}) : undefined

				// Build outputs metadata with shape information
				const outputsMetadata = onnxNode.output ? onnxNode.output.map((outputName: string, outputIdx: number) => {
					const attrs: Array<{key: string; value: string}> = [{ key: 'name', value: outputName }]
					const shapeInfo = tensorShapeMap.get(outputName)
					if (shapeInfo) {
						attrs.push({ key: 'tensor_shape', value: shapeInfo })
					}
					return {
						id: String(outputIdx),
						attrs
					}
				}) : undefined

				// Build enhanced label with tensor shape information
				let enhancedLabel = opType

				// Collect input shapes (show [?] for unknown)
				const inputShapes: string[] = []
				if (onnxNode.input) {
					onnxNode.input.forEach((inputName: string) => {
						const shapeInfo = tensorShapeMap.get(inputName)
						inputShapes.push(shapeInfo || '[?]')
					})
				}

				// Collect output shapes (show [?] for unknown)
				const outputShapes: string[] = []
				if (onnxNode.output) {
					onnxNode.output.forEach((outputName: string) => {
						const shapeInfo = tensorShapeMap.get(outputName)
						outputShapes.push(shapeInfo || '[?]')
					})
				}

				// Add shapes to label (always show if operation has inputs/outputs)
				if (inputShapes.length > 0) {
					enhancedLabel += `\nin: ${inputShapes.join(', ')}`
				}
				if (outputShapes.length > 0) {
					enhancedLabel += `\nout: ${outputShapes.join(', ')}`
				}

				// Limit label length for display
				if (enhancedLabel.length > 150) {
					enhancedLabel = enhancedLabel.substring(0, 147) + '...'
				}

				const node: GraphNode = {
					id: opNodeId,
					label: enhancedLabel,
					namespace: 'ops',
					attrs,
					inputsMetadata,
					outputsMetadata,
					incomingEdges
				}

				nodes.push(node)

				// Register outputs as producers for downstream nodes
				if (onnxNode.output) {
					onnxNode.output.forEach((outputName: string, outputIdx: number) => {
						valueToProducer.set(outputName, { nodeId: opNodeId, outputIndex: outputIdx })
					})
				}
			})
		}

		// Process graph outputs (these are the model's external outputs)
		if (graph.output) {
			graph.output.forEach((valueInfo: any, idx: number) => {
				const outputNodeId = `output_${nodeIdCounter++}`
				const outputName = valueInfo.name || `output_${idx}`

				// Get type and shape information (using camelCase from protobuf)
				const tensorType = valueInfo.type?.tensorType
				const elemType = tensorType?.elemType
				const shape = tensorType?.shape
				const shapeStr = formatTensorShape(elemType, shape)

				// Build label with shape information
				let label = 'Output'
				if (shapeStr && shapeStr !== 'unknown[]') {
					label += `\n${shapeStr}`
				}

				// Find the producer of this output
				const producer = valueToProducer.get(outputName)
				const incomingEdges = producer ? [{
					sourceNodeId: producer.nodeId,
					sourceNodeOutputId: String(producer.outputIndex),
					targetNodeInputId: '0'
				}] : []

				const node: GraphNode = {
					id: outputNodeId,
					label: label,
					namespace: 'io',
					attrs: [
						{ key: 'name', value: outputName }
					],
					inputsMetadata: [{
						id: '0',
						attrs: [{
							key: 'tensor_shape',
							value: shapeStr
						}]
					}],
					incomingEdges
				}

				nodes.push(node)
			})
		}

	} catch (error) {
		console.error('Error parsing ONNX model:', error)
		// Create a placeholder graph with error information
		nodes.push({
			id: 'node_0',
			label: 'ONNX Parsing Error',
			namespace: 'error',
			attrs: [
				{ key: 'error', value: (error as Error).message },
				{ key: 'info', value: 'Failed to parse ONNX model. Model may be corrupted or in an unsupported format.' }
			],
			incomingEdges: []
		})
	}

	// If no nodes were created, add a placeholder
	if (nodes.length === 0) {
		nodes.push({
			id: 'node_0',
			label: 'Empty ONNX Model',
			namespace: 'default',
			attrs: [
				{ key: 'info', value: 'No operations found in model' }
			],
			incomingEdges: []
		})
	}

	return {
		id: filename,
		nodes: nodes
	}
}
