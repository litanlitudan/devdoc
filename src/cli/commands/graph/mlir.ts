/**
 * graph:mlir command - Parse MLIR files to graph format
 * Uses Python bridge to invoke parse_mlir.py script
 */

import { Command, Flags, Args } from '@oclif/core'
import { readFile, writeFile } from 'node:fs/promises'
import { resolve, join } from 'node:path'
import { parseMLIR } from '../../services/python-bridge.js'
import { createLogger } from '../../services/logger.js'

const logger = createLogger('graph:mlir')

export default class GraphMLIR extends Command {
	static override description = 'Parse MLIR files and export graph representation'

	static override examples = [
		'<%= config.bin %> <%= command.id %> model.mlir',
		'<%= config.bin %> <%= command.id %> model.mlir --output graph.json',
		'<%= config.bin %> <%= command.id %> model.mlir --format json',
	]

	static override args = {
		file: Args.string({
			description: 'Path to MLIR file',
			required: true,
		}),
	}

	static override flags = {
		output: Flags.string({
			char: 'o',
			description: 'Output file path (defaults to stdout)',
		}),
		format: Flags.string({
			char: 'f',
			description: 'Output format',
			options: ['json', 'pretty'],
			default: 'json',
		}),
	}

	async run(): Promise<void> {
		const { args, flags } = await this.parse(GraphMLIR)

		try {
			// Resolve file paths
			const inputPath = resolve(args.file)
			const scriptPath = resolve(join(process.cwd(), 'scripts/parse_mlir.py'))

			logger.info({ file: inputPath }, 'Parsing MLIR file...')

			// Read MLIR content
			const mlirContent = await readFile(inputPath, 'utf-8')

			// Parse using Python bridge
			const graph = await parseMLIR(mlirContent, inputPath, scriptPath)

			// Format output
			const output =
				flags.format === 'pretty'
					? JSON.stringify(graph, null, 2)
					: JSON.stringify(graph)

			// Write to file or stdout
			if (flags.output) {
				const outputPath = resolve(flags.output)
				await writeFile(outputPath, output, 'utf-8')
				logger.info({ file: outputPath }, 'Graph written to file')
				this.log(`✓ Graph exported to: ${outputPath}`)
			} else {
				this.log(output)
			}

			logger.info('MLIR parsing completed successfully')
		} catch (error: unknown) {
			const err = error instanceof Error ? error : new Error('Unknown error')
			logger.error({ error: err.message, file: args.file }, 'MLIR parsing failed')
			this.error(`Failed to parse MLIR file: ${err.message}`, { exit: 1 })
		}
	}
}
