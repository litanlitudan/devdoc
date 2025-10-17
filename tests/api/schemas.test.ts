/**
 * Tests for API schemas
 * Validates zod schemas for graph API
 */

import { describe, it, expect } from 'vitest'
import {
	GraphNodeSchema,
	GraphSchema,
	MLIRParseRequestSchema,
	MLIRParseResponseSchema,
	ONNXShapeInferenceRequestSchema,
	ONNXShapeInferenceResponseSchema,
	ApiResponseSchema,
} from '../../lib/api/schemas/graph.js'
import { z } from 'zod'

describe('Graph API Schemas', () => {
	describe('GraphNodeSchema', () => {
		it('should validate valid graph node', () => {
			const node = {
				id: 'node1',
				label: 'func.func',
				namespace: 'main',
				attrs: [{ key: 'name', value: 'test' }],
				incomingEdges: [
					{
						sourceNodeId: 'node0',
						sourceNodeOutputId: '0',
						targetNodeInputId: '0',
					},
				],
			}

			const result = GraphNodeSchema.parse(node)
			expect(result).toEqual(node)
		})

		it('should validate minimal graph node', () => {
			const node = {
				id: 'node1',
				label: 'operation',
			}

			const result = GraphNodeSchema.parse(node)
			expect(result.id).toBe('node1')
			expect(result.label).toBe('operation')
		})

		it('should reject missing required fields', () => {
			expect(() => GraphNodeSchema.parse({ id: 'node1' })).toThrow()
			expect(() => GraphNodeSchema.parse({ label: 'operation' })).toThrow()
		})

		it('should validate optional fields', () => {
			const node = {
				id: 'node1',
				label: 'operation',
				inputsMetadata: [
					{
						id: '0',
						attrs: [{ key: 'shape', value: 'tensor<2x3xf32>' }],
					},
				],
				outputsMetadata: [
					{
						id: '0',
						attrs: [{ key: 'shape', value: 'tensor<2x3xf32>' }],
					},
				],
			}

			const result = GraphNodeSchema.parse(node)
			expect(result.inputsMetadata).toHaveLength(1)
			expect(result.outputsMetadata).toHaveLength(1)
		})
	})

	describe('GraphSchema', () => {
		it('should validate complete graph', () => {
			const graph = {
				id: 'test.mlir',
				nodes: [
					{ id: 'node1', label: 'input' },
					{ id: 'node2', label: 'operation' },
				],
			}

			const result = GraphSchema.parse(graph)
			expect(result.nodes).toHaveLength(2)
		})

		it('should validate empty graph', () => {
			const graph = {
				id: 'empty.mlir',
				nodes: [],
			}

			const result = GraphSchema.parse(graph)
			expect(result.nodes).toHaveLength(0)
		})

		it('should reject missing required fields', () => {
			expect(() => GraphSchema.parse({ nodes: [] })).toThrow()
			expect(() => GraphSchema.parse({ id: 'test' })).toThrow()
		})
	})

	describe('MLIRParseRequestSchema', () => {
		it('should validate valid request', () => {
			const request = {
				content: 'func.func @main() { return }',
				filename: 'test.mlir',
			}

			const result = MLIRParseRequestSchema.parse(request)
			expect(result.content).toBe(request.content)
			expect(result.filename).toBe(request.filename)
		})

		it('should apply default filename', () => {
			const request = {
				content: 'func.func @main() { return }',
			}

			const result = MLIRParseRequestSchema.parse(request)
			expect(result.filename).toBe('input.mlir')
		})

		it('should reject empty content', () => {
			expect(() =>
				MLIRParseRequestSchema.parse({ content: '' }),
			).toThrow()
		})

		it('should reject missing content', () => {
			expect(() =>
				MLIRParseRequestSchema.parse({ filename: 'test.mlir' }),
			).toThrow()
		})
	})

	describe('MLIRParseResponseSchema', () => {
		it('should validate successful response', () => {
			const response = {
				success: true,
				data: {
					id: 'test.mlir',
					nodes: [{ id: 'node1', label: 'operation' }],
				},
			}

			const result = MLIRParseResponseSchema.parse(response)
			expect(result.success).toBe(true)
			expect(result.data).toBeDefined()
		})

		it('should validate error response', () => {
			const response = {
				success: false,
				error: 'Parse error',
			}

			const result = MLIRParseResponseSchema.parse(response)
			expect(result.success).toBe(false)
			expect(result.error).toBe('Parse error')
		})

		it('should allow response without data or error', () => {
			const response = {
				success: true,
			}

			const result = MLIRParseResponseSchema.parse(response)
			expect(result.success).toBe(true)
		})
	})

	describe('ONNXShapeInferenceRequestSchema', () => {
		it('should validate valid request', () => {
			const request = {
				modelPath: '/path/to/model.onnx',
			}

			const result = ONNXShapeInferenceRequestSchema.parse(request)
			expect(result.modelPath).toBe(request.modelPath)
		})

		it('should reject empty model path', () => {
			expect(() =>
				ONNXShapeInferenceRequestSchema.parse({ modelPath: '' }),
			).toThrow()
		})

		it('should reject missing model path', () => {
			expect(() => ONNXShapeInferenceRequestSchema.parse({})).toThrow()
		})
	})

	describe('ONNXShapeInferenceResponseSchema', () => {
		it('should validate successful response', () => {
			const response = {
				success: true,
				message: 'Shape inference completed',
			}

			const result = ONNXShapeInferenceResponseSchema.parse(response)
			expect(result.success).toBe(true)
			expect(result.message).toBeDefined()
		})

		it('should validate error response', () => {
			const response = {
				success: false,
				error: 'Shape inference failed',
			}

			const result = ONNXShapeInferenceResponseSchema.parse(response)
			expect(result.success).toBe(false)
			expect(result.error).toBeDefined()
		})
	})

	describe('ApiResponseSchema factory', () => {
		it('should create schema with custom data type', () => {
			const dataSchema = z.object({
				name: z.string(),
				value: z.number(),
			})

			const ResponseSchema = ApiResponseSchema(dataSchema)

			const response = {
				success: true,
				data: { name: 'test', value: 42 },
			}

			const result = ResponseSchema.parse(response)
			expect(result.data).toEqual({ name: 'test', value: 42 })
		})

		it('should validate response without data', () => {
			const dataSchema = z.string()
			const ResponseSchema = ApiResponseSchema(dataSchema)

			const response = {
				success: false,
				error: 'Error occurred',
			}

			const result = ResponseSchema.parse(response)
			expect(result.success).toBe(false)
			expect(result.error).toBeDefined()
		})

		it('should reject invalid data type', () => {
			const dataSchema = z.object({
				value: z.number(),
			})

			const ResponseSchema = ApiResponseSchema(dataSchema)

			expect(() =>
				ResponseSchema.parse({
					success: true,
					data: { value: 'not-a-number' },
				}),
			).toThrow()
		})
	})

	describe('Type inference', () => {
		it('should infer correct types from schemas', () => {
			// This is a compile-time test
			type GraphNode = z.infer<typeof GraphNodeSchema>
			type Graph = z.infer<typeof GraphSchema>
			type MLIRRequest = z.infer<typeof MLIRParseRequestSchema>

			const node: GraphNode = {
				id: 'test',
				label: 'operation',
			}

			const graph: Graph = {
				id: 'test',
				nodes: [node],
			}

			const request: MLIRRequest = {
				content: 'test',
				filename: 'test.mlir',
			}

			expect(node.id).toBeDefined()
			expect(graph.nodes).toBeDefined()
			expect(request.content).toBeDefined()
		})
	})
})
