/**
 * Python Bridge - Typed wrappers for Python script execution
 * Provides clean TypeScript interfaces for MLIR/ONNX Python utilities
 */

import { execa } from 'execa'
import { z } from 'zod'
import { createLogger } from './logger.js'

const logger = createLogger('python-bridge')

/**
 * Determine the correct Python command
 * Use 'python' in conda environment, otherwise 'python3'
 */
function getPythonCommand(): string {
	return process.env.CONDA_DEFAULT_ENV ? 'python' : 'python3'
}

/**
 * Execute a Python script with input via stdin
 */
export async function executePythonScript<T>(
	scriptPath: string,
	args: string[] = [],
	options: {
		input?: string
		timeout?: number
		maxBuffer?: number
		schema?: z.ZodSchema<T>
	} = {},
): Promise<T> {
	const {
		input,
		timeout = 30000,
		maxBuffer = 50 * 1024 * 1024, // 50MB
		schema,
	} = options

	logger.debug({ scriptPath, args }, 'Executing Python script')

	try {
		const result = await execa(getPythonCommand(), [scriptPath, ...args], {
			input,
			timeout,
			maxBuffer,
			encoding: 'utf8',
			reject: true,
		})

		// Parse JSON output
		const output = JSON.parse(result.stdout)

		// Check for error in output
		if (output.error) {
			throw new Error(`Python script error: ${output.message}`)
		}

		// Validate with schema if provided
		if (schema) {
			return schema.parse(output)
		}

		return output as T
	} catch (error: unknown) {
		const err =
			error instanceof Error
				? error
				: new Error(typeof error === 'string' ? error : 'Unknown error')

		// Enhance error messages
		if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
			throw new Error(
				'Python not found. Please install Python 3.9+ and ensure it is in your PATH.',
			)
		} else if ((err as { timedOut?: boolean }).timedOut) {
			throw new Error(
				`Python script timed out after ${timeout}ms: ${scriptPath}`,
			)
		} else if (err.message?.includes('maxBuffer')) {
			throw new Error(
				`Python script output exceeded ${maxBuffer} bytes: ${scriptPath}`,
			)
		}

		throw err
	}
}

/**
 * Check if Python is available
 */
export async function checkPythonAvailable(): Promise<{
	available: boolean
	version?: string
	error?: string
}> {
	try {
		const result = await execa(getPythonCommand(), ['--version'], {
			timeout: 5000,
		})
		return {
			available: true,
			version: result.stdout || result.stderr,
		}
	} catch (error: unknown) {
		return {
			available: false,
			error:
				error instanceof Error
					? error.message
					: typeof error === 'string'
						? error
						: 'Unknown error',
		}
	}
}

/**
 * Parse MLIR file to graph format
 */
export interface MLIRGraphNode {
	id: string
	label: string
	namespace: string
	attrs: Array<{ key: string; value: string }>
	incomingEdges: Array<{
		sourceNodeId: string
		sourceNodeOutputId?: string
		targetNodeInputId?: string
	}>
	inputsMetadata?: Array<{
		id: string
		attrs: Array<{ key: string; value: string }>
	}>
	outputsMetadata?: Array<{
		id: string
		attrs: Array<{ key: string; value: string }>
	}>
}

export interface MLIRGraph {
	id: string
	nodes: MLIRGraphNode[]
}

const MLIRGraphSchema = z.object({
	id: z.string(),
	nodes: z.array(
		z.object({
			id: z.string(),
			label: z.string(),
			namespace: z.string(),
			attrs: z.array(z.object({ key: z.string(), value: z.string() })),
			incomingEdges: z.array(
				z.object({
					sourceNodeId: z.string(),
					sourceNodeOutputId: z.string().optional(),
					targetNodeInputId: z.string().optional(),
				}),
			),
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
		}),
	),
})

export async function parseMLIR(
	mlirContent: string,
	filename: string,
	scriptPath: string,
): Promise<MLIRGraph> {
	return executePythonScript<MLIRGraph>(scriptPath, [filename], {
		input: mlirContent,
		schema: MLIRGraphSchema,
	})
}

/**
 * Infer ONNX shapes
 */
export interface ONNXShapeInfo {
	success: boolean
	model?: Buffer
	error?: string
}

export async function inferONNXShapes(
	modelPath: string,
	scriptPath: string,
): Promise<ONNXShapeInfo> {
	try {
		const result = await execa(getPythonCommand(), [scriptPath, modelPath], {
			timeout: 60000,
			encoding: 'buffer',
		})

		return {
			success: true,
			model: result.stdout as Buffer,
		}
	} catch (error: any) {
		return {
			success: false,
			error: error.message,
		}
	}
}
