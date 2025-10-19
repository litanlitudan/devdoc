/**
 * MLIR to Model Explorer Graph Converter
 *
 * Attempts to use the native C++ MLIR parser when available, falling back to a
 * TypeScript regex-based implementation when the native binary is missing or
 * fails. The return value conforms to Model Explorer's graph schema.
 */

import { spawnSync } from 'child_process'
import { existsSync } from 'fs'
import { dirname, join } from 'path'
import { fileURLToPath } from 'url'
import { ModelExplorerGraph, ModelExplorerGraphs } from './mlir-graph-types.js'
import { parseMlirWithRegex } from './mlir-regex-parser.js'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

function candidateCppParsers(): string[] {
	const repoRoot = join(__dirname, '..', '..')
	return [
		join(repoRoot, 'src', 'mlir', 'build', 'mlir_parser'),
		join(repoRoot, 'src', 'mlir', 'mlir_parser'),
	]
}

function findCppParser(): string | null {
	for (const candidate of candidateCppParsers()) {
		if (existsSync(candidate)) {
			return candidate
		}
	}

	const delimiter = process.platform === 'win32' ? ';' : ':'
	const systemPath = process.env.PATH ?? ''
	for (const pathEntry of systemPath.split(delimiter)) {
		if (!pathEntry) continue
		const candidate = join(pathEntry, 'mlir_parser')
		if (existsSync(candidate)) {
			return candidate
		}
	}

	return null
}

function runCppParser(
	parserPath: string,
	mlirContent: string,
	filename: string,
): { status: number | null; stdout: string; stderr: string } {
	const result = spawnSync(parserPath, [filename], {
		input: mlirContent,
		encoding: 'utf-8',
		maxBuffer: 50 * 1024 * 1024,
	})

	if (result.error) {
		throw result.error
	}

	return {
		status: result.status,
		stdout: result.stdout ?? '',
		stderr: result.stderr ?? '',
	}
}

function parseWithCpp(
	parserPath: string,
	mlirContent: string,
	filename: string,
): ModelExplorerGraphs | null {
	const { status, stdout, stderr } = runCppParser(
		parserPath,
		mlirContent,
		filename,
	)

	if (status === 0) {
		try {
			const parsed = JSON.parse(stdout) as ModelExplorerGraphs
			if ((parsed as any)?.error) {
				console.warn(
					`⚠️ C++ parser reported error: ${(parsed as any).message ?? (parsed as any).error}`,
				)
				return null
			}
			console.log(
				`✓ C++ MLIR context parser successful (${parsed.graphs.length} graphs)`,
			)
			return parsed
		} catch (err) {
			console.warn(
				'⚠️ C++ parser returned invalid JSON. Falling back to TypeScript parser.',
				err,
			)
			return null
		}
	}

	console.warn(`⚠️ C++ parser exited with status ${status}. stderr: ${stderr}`)
	try {
		const parsed = JSON.parse(stdout) as Record<string, unknown>
		if (parsed?.error) {
			console.warn(`Parser message: ${parsed.message ?? parsed.error}`)
		}
	} catch {
		// Ignore JSON parsing failures; fallback will handle the content.
	}
	return null
}

/**
 * Convert MLIR text to Model Explorer graph format.
 *
 * @param mlirContent The MLIR text content to parse
 * @param filename The filename to use as the base graph ID
 * @returns Graph collection suitable for Model Explorer
 */
export function convertMLIRToGraph(
	mlirContent: string,
	filename: string,
): ModelExplorerGraphs {
	const parserPath = findCppParser()

	if (parserPath) {
		try {
			const result = parseWithCpp(parserPath, mlirContent, filename)
			if (result) {
				return result
			}
		} catch (error) {
			console.warn(
				`⚠️ Failed to execute C++ parser at ${parserPath}:`,
				error,
				'\nFalling back to TypeScript parser.',
			)
		}
	} else {
		console.warn(
			'ℹ️ C++ MLIR parser not found. Falling back to TypeScript regex parser.',
		)
	}

	return parseMlirWithRegex(mlirContent, filename)
}

/**
 * Create a minimal test graph for debugging.
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
