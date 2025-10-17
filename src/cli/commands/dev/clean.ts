/**
 * dev:clean command - Clean build artifacts
 * Wraps the Makefile clean target with TypeScript orchestration
 */

import { Command, Flags } from '@oclif/core'
import { execa } from 'execa'
import { createLogger } from '../../services/logger.js'

const logger = createLogger('dev:clean')

export default class DevClean extends Command {
	static override description = 'Remove build artifacts and dependencies'

	static override examples = [
		'<%= config.bin %> <%= command.id %>',
		'<%= config.bin %> <%= command.id %> --all',
	]

	static override flags = {
		all: Flags.boolean({
			char: 'a',
			description: 'Remove all artifacts including node_modules',
			default: false,
		}),
	}

	async run(): Promise<void> {
		const { flags } = await this.parse(DevClean)

		try {
			const makeTarget = flags.all ? 'clean-all' : 'clean'

			logger.info(`Cleaning ${flags.all ? 'all artifacts' : 'build artifacts'}...`)

			// Execute make target with streaming output
			await execa('make', [makeTarget], {
				stdio: 'inherit',
				cwd: process.cwd(),
			})

			logger.info('Cleanup completed successfully')
		} catch (error: unknown) {
			const err = error instanceof Error ? error : new Error('Unknown error')
			logger.error({ error: err.message }, 'Cleanup failed')
			this.error(`Cleanup failed: ${err.message}`, { exit: 1 })
		}
	}
}
