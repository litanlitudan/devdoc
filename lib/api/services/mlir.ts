/**
 * MLIR Service - Business logic for MLIR parsing and graph generation
 * Uses the Python bridge to execute MLIR parsing scripts
 */

import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { parseMLIR } from '../../../src/cli/services/python-bridge.js'
import type { Graph } from '../schemas/graph.js'
import { createLogger } from '../util/logger.js'

const logger = createLogger('mlir-service')

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

/**
 * Parse MLIR content and return Model Explorer graph
 */
export async function parseMlirToGraph(
	mlirContent: string,
	filename: string = 'input.mlir',
): Promise<Graph> {
	logger.info({ filename }, 'Parsing MLIR content')

	// Resolve path to Python script
	const scriptPath = join(__dirname, '../../../scripts/parse_mlir.py')

	try {
		const graph = await parseMLIR(mlirContent, filename, scriptPath)

		logger.info(
			{ nodeCount: graph.nodes.length, filename },
			'Successfully parsed MLIR',
		)

		return graph
	} catch (error: any) {
		logger.error({ error, filename }, 'Failed to parse MLIR')
		throw new Error(`MLIR parsing failed: ${error.message}`)
	}
}

/**
 * Validate MLIR syntax (basic check)
 */
export function validateMlirSyntax(mlirContent: string): {
	valid: boolean
	error?: string
} {
	// Basic validation checks
	if (!mlirContent.trim()) {
		return { valid: false, error: 'MLIR content is empty' }
	}

	// Check for basic MLIR structure
	if (!mlirContent.includes('func.func') && !mlirContent.includes('module')) {
		return {
			valid: false,
			error: 'MLIR content does not contain valid structure',
		}
	}

	return { valid: true }
}
