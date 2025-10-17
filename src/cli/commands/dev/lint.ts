/**
 * dev:lint command - Run linter
 * Wraps the Makefile lint target with TypeScript orchestration
 */

import { Command, Flags } from '@oclif/core'
import { execa } from 'execa'
import { createLogger } from '../../services/logger.js'

const logger = createLogger('dev:lint')

export default class DevLint extends Command {
	static override description = 'Run linter and code quality checks'

	static override examples = [
		'<%= config.bin %> <%= command.id %>',
		'<%= config.bin %> <%= command.id %> --format',
		'<%= config.bin %> <%= command.id %> --typecheck',
	]

	static override flags = {
		format: Flags.boolean({
			char: 'f',
			description: 'Format code with prettier',
			default: false,
		}),
		typecheck: Flags.boolean({
			char: 't',
			description: 'Run TypeScript type checking',
			default: false,
		}),
	}

	async run(): Promise<void> {
		const { flags } = await this.parse(DevLint)

		try {
			// Run type checking if requested
			if (flags.typecheck) {
				logger.info('Running type checker...')
				await execa('make', ['typecheck'], {
					stdio: 'inherit',
					cwd: process.cwd(),
				})
			}

			// Run formatter if requested
			if (flags.format) {
				logger.info('Formatting code...')
				await execa('make', ['format'], {
					stdio: 'inherit',
					cwd: process.cwd(),
				})
			}

			// Always run linter
			logger.info('Running linter...')
			await execa('make', ['lint'], {
				stdio: 'inherit',
				cwd: process.cwd(),
			})

			logger.info('Linting completed successfully')
		} catch (error: unknown) {
			const err = error instanceof Error ? error : new Error('Unknown error')
			logger.error({ error: err.message }, 'Linting failed')
			this.error(`Linting failed: ${err.message}`, { exit: 1 })
		}
	}
}
