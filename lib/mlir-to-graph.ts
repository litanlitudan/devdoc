/**
 * MLIR to Model Explorer Graph Converter
 *
 * This module converts MLIR (Multi-Level Intermediate Representation) text
 * into a graph format compatible with Google's Model Explorer visualization tool.
 *
 * Uses Model Explorer's pre-built adapter package (ai-edge-model-explorer-adapter)
 * which contains Google's official C++ MLIR parser compiled for multiple platforms.
 *
 * Note: Requires Python 3.9+ with ai-edge-model-explorer-adapter installed.
 * Install with: pip install ai-edge-model-explorer-adapter
 */

import { execFileSync } from 'child_process'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'

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
 * Convert MLIR text to Model Explorer graph format using Python adapter
 *
 * Uses Model Explorer's pre-built C++ MLIR parser via the
 * ai-edge-model-explorer-adapter Python package.
 *
 * @param mlirContent The MLIR text content to parse
 * @param filename The filename to use as the graph ID
 * @returns A graph object compatible with Model Explorer
 * @throws Error if Python or adapter is not available, or parsing fails
 */
export function convertMLIRToGraph(mlirContent: string, filename: string): ModelExplorerGraph {
	try {
		// Path to Python script (relative to compiled JS location in dist/)
		const scriptPath = join(__dirname, '..', 'scripts', 'parse_mlir_with_adapter.py')

		// Run Python MLIR parser with filename as argument
		// Use 'python' in conda environment, otherwise 'python3'
		const pythonCmd = process.env.CONDA_DEFAULT_ENV ? 'python' : 'python3'
		const resultJson = execFileSync(pythonCmd, [scriptPath, filename], {
			input: mlirContent,
			maxBuffer: 50 * 1024 * 1024, // 50MB max buffer
			timeout: 30000, // 30 second timeout
			encoding: 'utf-8'
		})

		const result = JSON.parse(resultJson)

		// Check if parsing succeeded
		if (result.error) {
			// Distinguish between adapter not installed vs parsing errors
			if (result.error === 'Model Explorer adapter not available') {
				throw new Error(`MLIR parser not installed: ${result.message}`)
			} else {
				// Show parsing error directly without misleading prefix
				throw new Error(result.message)
			}
		}

		console.log('✓ Python MLIR parsing successful')
		return result as ModelExplorerGraph

	} catch (error: any) {
		// Provide helpful error messages for system-level errors
		if (error.code === 'ENOENT') {
			throw new Error(
				'Python not found. Please install Python 3.9+ and ensure it is in your PATH.\n' +
				'Then install MLIR parser: pip install ai-edge-model-explorer-adapter'
			)
		} else if (error.message?.includes('MLIR parser not installed')) {
			throw error // Re-throw adapter installation error
		} else if (error.message?.includes('INVALID_ARGUMENT') || error.message?.includes('Failed to parse')) {
			throw error // Re-throw parsing errors directly (these are MLIR syntax errors)
		} else if (error.status !== undefined) {
			throw new Error(
				`Python MLIR parser failed (exit code ${error.status}).\n` +
				'Install with: pip install ai-edge-model-explorer-adapter'
			)
		} else {
			throw new Error(
				`MLIR parsing error: ${error.message}\n` +
				'Ensure MLIR parser is installed: pip install ai-edge-model-explorer-adapter'
			)
		}
	}
}

/**
 * Create a minimal test graph for debugging
 * @param filename The filename to use as the graph ID
 * @returns A minimal graph with 3 connected nodes
 */
export function createTestGraph(filename: string): ModelExplorerGraph {
	return {
		id: filename,
		nodes: [
			{
				id: 'node_0',
				label: 'Start',
				namespace: '',
				attrs: [],
				incomingEdges: []
			},
			{
				id: 'node_1',
				label: 'Middle',
				namespace: '',
				attrs: [],
				incomingEdges: [
					{
						sourceNodeId: 'node_0'
					}
				]
			},
			{
				id: 'node_2',
				label: 'End',
				namespace: '',
				attrs: [],
				incomingEdges: [
					{
						sourceNodeId: 'node_1'
					}
				]
			}
		]
	}
}
