/**
 * MLIR to Model Explorer Graph Converter
 *
 * This module converts MLIR (Multi-Level Intermediate Representation) text
 * into a graph format compatible with Google's Model Explorer visualization tool.
 *
 * Parses MLIR text directly in Python using regex patterns and constructs
 * Model Explorer graph structures. Supports ALL MLIR dialects by treating
 * operations as generic graph nodes.
 *
 * Note: Requires Python 3.9+. No external dependencies required.
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
 * Convert MLIR text to Model Explorer graph format using direct Python parser
 *
 * Parses MLIR text directly using Python regex patterns and constructs
 * Model Explorer graph structures. Supports arbitrary MLIR dialects.
 *
 * @param mlirContent The MLIR text content to parse
 * @param filename The filename to use as the graph ID
 * @returns A graph object compatible with Model Explorer
 * @throws Error if Python is not available or parsing fails
 */
export function convertMLIRToGraph(mlirContent: string, filename: string): ModelExplorerGraph {
	try {
		// Path to Python script (relative to compiled JS location in dist/)
		const scriptPath = join(__dirname, '..', 'scripts', 'parse_mlir.py')

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
			// Show parsing error directly
			throw new Error(result.message)
		}

		console.log('✓ Python MLIR parsing successful')
		return result as ModelExplorerGraph

	} catch (error: any) {
		// Provide helpful error messages for system-level errors
		if (error.code === 'ENOENT') {
			throw new Error(
				'Python not found. Please install Python 3.9+ and ensure it is in your PATH.'
			)
		} else if (error.message?.includes('INVALID_ARGUMENT') || error.message?.includes('Failed to parse')) {
			throw error // Re-throw parsing errors directly (these are MLIR syntax errors)
		} else if (error.status !== undefined) {
			throw new Error(
				`Python MLIR parser failed (exit code ${error.status}).`
			)
		} else {
			throw new Error(
				`MLIR parsing error: ${error.message}`
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
