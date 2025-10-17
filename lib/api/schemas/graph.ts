/**
 * Zod schemas for graph API endpoints
 * Provides type-safe validation for MLIR/ONNX graph requests and responses
 */

import { z } from 'zod'

// ============================================================================
// Graph Node Schemas
// ============================================================================

export const GraphNodeSchema = z.object({
	id: z.string(),
	label: z.string(),
	namespace: z.string().optional(),
	attrs: z.array(z.object({ key: z.string(), value: z.string() })).optional(),
	incomingEdges: z
		.array(
			z.object({
				sourceNodeId: z.string(),
				sourceNodeOutputId: z.string().optional(),
				targetNodeInputId: z.string().optional(),
			}),
		)
		.optional(),
	inputsMetadata: z
		.array(
			z.object({
				id: z.string(),
				attrs: z.array(z.object({ key: z.string(), value: z.string() })),
			}),
		)
		.optional(),
	outputsMetadata: z
		.array(
			z.object({
				id: z.string(),
				attrs: z.array(z.object({ key: z.string(), value: z.string() })),
			}),
		)
		.optional(),
})

export const GraphSchema = z.object({
	id: z.string(),
	nodes: z.array(GraphNodeSchema),
})

export type GraphNode = z.infer<typeof GraphNodeSchema>
export type Graph = z.infer<typeof GraphSchema>

// ============================================================================
// MLIR Request/Response Schemas
// ============================================================================

export const MLIRParseRequestSchema = z.object({
	content: z.string().min(1, 'MLIR content cannot be empty'),
	filename: z.string().default('input.mlir'),
})

export const MLIRParseResponseSchema = z.object({
	success: z.boolean(),
	data: GraphSchema.optional(),
	error: z.string().optional(),
})

export type MLIRParseRequest = z.infer<typeof MLIRParseRequestSchema>
export type MLIRParseResponse = z.infer<typeof MLIRParseResponseSchema>

// ============================================================================
// ONNX Request/Response Schemas
// ============================================================================

export const ONNXShapeInferenceRequestSchema = z.object({
	modelPath: z.string().min(1, 'Model path cannot be empty'),
})

export const ONNXShapeInferenceResponseSchema = z.object({
	success: z.boolean(),
	message: z.string().optional(),
	error: z.string().optional(),
})

export type ONNXShapeInferenceRequest = z.infer<
	typeof ONNXShapeInferenceRequestSchema
>
export type ONNXShapeInferenceResponse = z.infer<
	typeof ONNXShapeInferenceResponseSchema
>

// ============================================================================
// Generic API Response Schema
// ============================================================================

export const ApiResponseSchema = <T extends z.ZodTypeAny>(dataSchema: T) =>
	z.object({
		success: z.boolean(),
		data: dataSchema.optional(),
		error: z.string().optional(),
	})
