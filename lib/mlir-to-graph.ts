/**
 * MLIR to Model Explorer Graph Converter
 *
 * This module converts MLIR (Multi-Level Intermediate Representation) text
 * into a graph format compatible with Google's Model Explorer visualization tool.
 *
 * **Parser Implementation:**
 * - **Primary**: C++ MLIR context-based parser (if built) - Implements the documented
 *   universal MLIR parser pipeline with proper dialect registration, ModuleOp parsing,
 *   and full region traversal
 * - **Fallback**: Python regex-based parser - Lightweight, dependency-free parsing
 *   that handles arbitrary MLIR dialects as generic operations
 *
 * The C++ parser provides better accuracy and performance but requires building
 * LLVM/MLIR. See `src/mlir/BUILD.md` for build instructions.
 *
 * Note: Requires Python 3.9+. C++ parser is optional but recommended.
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
	attrs: Array<{ key: string; value: string }>
	outputsMetadata?: Array<{
		id: string
		attrs: Array<{ key: string; value: string }>
	}>
	inputsMetadata?: Array<{
		id: string
		attrs: Array<{ key: string; value: string }>
	}>
	incomingEdges: Array<{
		sourceNodeId: string
		sourceNodeOutputId?: string
		targetNodeInputId?: string
	}>
	subgraphIds?: string[]
}

export interface ModelExplorerGraph {
	id: string
	nodes: GraphNode[]
}

export interface ModelExplorerGraphs {
	graphs: ModelExplorerGraph[]
}

/**
 * Convert MLIR text to Model Explorer graph format
 *
 * Uses C++ MLIR context-based parser if available, with automatic fallback
 * to Python regex parser. The C++ parser implements the full documented
 * pipeline with proper dialect registration and region traversal.
 *
 * Returns multi-graph format with one graph per function.
 *
 * @param mlirContent The MLIR text content to parse
 * @param filename The filename to use as the base graph ID
 * @returns A graphs object containing one graph per function
 * @throws Error if Python is not available or parsing fails
 */
export function convertMLIRToGraph(
	mlirContent: string,
	filename: string,
): ModelExplorerGraphs {
	try {
		// Path to wrapper script that tries C++ parser first, falls back to Python regex
		// (relative to compiled JS location in dist/lib/)
		const scriptPath = join(
			__dirname,
			'..',
			'..',
			'scripts',
			'parse_mlir_cpp.py',
		)

		// Run Python MLIR parser with filename as argument
		// Use 'python' in conda environment, otherwise 'python3'
		const pythonCmd = process.env.CONDA_DEFAULT_ENV ? 'python' : 'python3'
		const resultJson = execFileSync(pythonCmd, [scriptPath, filename], {
			input: mlirContent,
			maxBuffer: 50 * 1024 * 1024, // 50MB max buffer
			timeout: 30000, // 30 second timeout
			encoding: 'utf-8',
		})

		const result = JSON.parse(resultJson)

		// Check if parsing succeeded
		if (result.error) {
			// Show parsing error directly
			throw new Error(result.message)
		}

		console.log(
			`✓ Python MLIR parsing successful (${result.graphs?.length || 0} graphs)`,
		)
		return result as ModelExplorerGraphs
	} catch (error: any) {
		// Provide helpful error messages for system-level errors
		if (error.code === 'ENOENT') {
			throw new Error(
				'Python not found. Please install Python 3.9+ and ensure it is in your PATH.',
			)
		} else if (
			error.message?.includes('INVALID_ARGUMENT') ||
			error.message?.includes('Failed to parse')
		) {
			throw error // Re-throw parsing errors directly (these are MLIR syntax errors)
		} else if (error.status !== undefined) {
			throw new Error(`Python MLIR parser failed (exit code ${error.status}).`)
		} else {
			throw new Error(`MLIR parsing error: ${error.message}`)
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
				incomingEdges: [],
			},
			{
				id: 'node_1',
				label: 'Middle',
				namespace: '',
				attrs: [],
				incomingEdges: [
					{
						sourceNodeId: 'node_0',
					},
				],
			},
			{
				id: 'node_2',
				label: 'End',
				namespace: '',
				attrs: [],
				incomingEdges: [
					{
						sourceNodeId: 'node_1',
					},
				],
			},
		],
	}
}
