/**
 * graph:onnx command - Infer shapes for ONNX models
 * Uses Python bridge to invoke infer_onnx_shapes.py script
 */

import { Command, Flags, Args } from '@oclif/core'
import { writeFile } from 'node:fs/promises'
import { resolve, join } from 'node:path'
import { inferONNXShapes } from '../../services/python-bridge.js'
import { createLogger } from '../../services/logger.js'

const logger = createLogger('graph:onnx')

export default class GraphONNX extends Command {
	static override description = 'Infer tensor shapes for ONNX models'

	static override examples = [
		'<%= config.bin %> <%= command.id %> model.onnx',
		'<%= config.bin %> <%= command.id %> model.onnx --output model_with_shapes.onnx',
	]

	static override args = {
		file: Args.string({
			description: 'Path to ONNX model file',
			required: true,
		}),
	}

	static override flags = {
		output: Flags.string({
			char: 'o',
			description: 'Output file path for model with inferred shapes',
			required: true,
		}),
	}

	async run(): Promise<void> {
		const { args, flags } = await this.parse(GraphONNX)

		try {
			// Resolve file paths
			const inputPath = resolve(args.file)
			const outputPath = resolve(flags.output)
			const scriptPath = resolve(join(process.cwd(), 'scripts/infer_onnx_shapes.py'))

			logger.info({ file: inputPath }, 'Inferring ONNX shapes...')

			// Infer shapes using Python bridge
			const result = await inferONNXShapes(inputPath, scriptPath)

			if (!result.success || !result.model) {
				throw new Error(result.error || 'Shape inference failed')
			}

			// Write output model
			await writeFile(outputPath, result.model)
			logger.info({ file: outputPath }, 'Model with inferred shapes written to file')
			this.log(`✓ Model with inferred shapes exported to: ${outputPath}`)

			logger.info('ONNX shape inference completed successfully')
		} catch (error: unknown) {
			const err = error instanceof Error ? error : new Error('Unknown error')
			logger.error({ error: err.message, file: args.file }, 'ONNX shape inference failed')
			this.error(`Failed to infer ONNX shapes: ${err.message}`, { exit: 1 })
		}
	}
}
