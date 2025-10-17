/**
 * dev:build command - Build the project
 * Wraps the Makefile build target with TypeScript orchestration
 */

import { Command, Flags } from '@oclif/core'
import { execa } from 'execa'
import { createLogger } from '../../services/logger.js'

const logger = createLogger('dev:build')

export default class DevBuild extends Command {
	static override description = 'Build the project using TypeScript compiler'

	static override examples = [
		'<%= config.bin %> <%= command.id %>',
		'<%= config.bin %> <%= command.id %> --watch',
	]

	static override flags = {
		watch: Flags.boolean({
			char: 'w',
			description: 'Build in watch mode',
			default: false,
		}),
	}

	async run(): Promise<void> {
		const { flags } = await this.parse(DevBuild)

		try {
			logger.info('Starting build process...')

			const makeTarget = flags.watch ? 'watch' : 'build'

			// Execute make target with streaming output
			await execa('make', [makeTarget], {
				stdio: 'inherit',
				cwd: process.cwd(),
			})

			if (!flags.watch) {
				logger.info('Build completed successfully')
			}
		} catch (error: unknown) {
			const err = error instanceof Error ? error : new Error('Unknown error')
			logger.error({ error: err.message }, 'Build failed')
			this.error(`Build failed: ${err.message}`, { exit: 1 })
		}
	}
}
