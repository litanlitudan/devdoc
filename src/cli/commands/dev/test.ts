/**
 * dev:test command - Run tests
 * Wraps the Makefile test target with TypeScript orchestration
 */

import { Command, Flags } from '@oclif/core'
import { execa } from 'execa'
import { createLogger } from '../../services/logger.js'

const logger = createLogger('dev:test')

export default class DevTest extends Command {
	static override description = 'Run the test suite'

	static override examples = [
		'<%= config.bin %> <%= command.id %>',
		'<%= config.bin %> <%= command.id %> --watch',
		'<%= config.bin %> <%= command.id %> --coverage',
		'<%= config.bin %> <%= command.id %> --ui',
	]

	static override flags = {
		watch: Flags.boolean({
			char: 'w',
			description: 'Run tests in watch mode',
			default: false,
		}),
		coverage: Flags.boolean({
			char: 'c',
			description: 'Run tests with coverage report',
			default: false,
		}),
		ui: Flags.boolean({
			description: 'Run tests with UI',
			default: false,
		}),
	}

	async run(): Promise<void> {
		const { flags } = await this.parse(DevTest)

		try {
			logger.info('Starting test suite...')

			let makeTarget = 'test'
			if (flags.watch) {
				makeTarget = 'test-watch'
			} else if (flags.coverage) {
				makeTarget = 'cover'
			} else if (flags.ui) {
				makeTarget = 'test-ui'
			}

			// Execute make target with streaming output
			await execa('make', [makeTarget], {
				stdio: 'inherit',
				cwd: process.cwd(),
			})

			if (!flags.watch) {
				logger.info('Tests completed successfully')
			}
		} catch (error: unknown) {
			const err = error instanceof Error ? error : new Error('Unknown error')
			logger.error({ error: err.message }, 'Tests failed')
			this.error(`Tests failed: ${err.message}`, { exit: 1 })
		}
	}
}
