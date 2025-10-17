/**
 * Serve command - Start the devdoc development server
 *
 * This command wraps the existing server implementation with oclif,
 * maintaining full backward compatibility while providing a cleaner CLI interface.
 */

import { Command, Flags, Args } from '@oclif/core'
import path from 'node:path'
import devdoc from '../../../lib/server.js'
import splash from '../../../lib/splash.js'
import type { Flags as ServerFlags } from '../../../lib/types.js'
import { logger, setLogLevel } from '../services/logger.js'
import { loadConfig } from '../config.js'

export default class Serve extends Command {
	static override description = 'Start the devdoc development server'

	static override examples = [
		'<%= config.bin %> <%= command.id %>',
		'<%= config.bin %> <%= command.id %> ./docs',
		'<%= config.bin %> <%= command.id %> --port 3000 --watch',
		'<%= config.bin %> <%= command.id %> -p 8080 -a 0.0.0.0',
	]

	static override flags = {
		port: Flags.integer({
			char: 'p',
			description: 'Port to run server on',
			default: 8642,
		}),
		address: Flags.string({
			char: 'a',
			description: 'Address to bind server to',
			default: 'localhost',
		}),
		watch: Flags.boolean({
			char: 'w',
			description: 'Enable file watching and live reload',
			default: false,
		}),
		verbose: Flags.boolean({
			char: 'v',
			description: 'Verbose output',
			default: false,
		}),
		silent: Flags.boolean({
			char: 's',
			description: 'Silent mode',
			default: false,
		}),
		livereloadport: Flags.integer({
			char: 'b',
			description: 'Port for LiveReload server',
			default: 35729,
		}),
	}

	static override args = {
		path: Args.string({
			name: 'path',
			description: 'Directory or file to serve',
			required: false,
		}),
	}

	public async run(): Promise<void> {
		const { args, flags } = await this.parse(Serve)

		// Display splash screen (unless silent)
		splash({ silent: flags.silent })

		// Configure logger based on verbosity
		if (flags.verbose) {
			setLogLevel('debug')
		} else if (flags.silent) {
			setLogLevel('silent')
		}

		logger.info('Starting devdoc server...')

		// Load configuration (merging config file with CLI flags)
		const config = loadConfig({
			server: {
				port: flags.port,
				address: flags.address,
				watch: flags.watch,
				verbose: flags.verbose,
				silent: flags.silent,
			},
		})

		logger.debug({ config }, 'Loaded configuration')

		// Determine server path
		const cwd = process.cwd()
		const dir = args.path ?? cwd
		const validatedServerPath = this.validateServerPath(dir, cwd)

		// Build flags object compatible with existing server
		const serverFlags: ServerFlags = {
			port: String(config.server.port),
			address: config.server.address,
			watch: config.server.watch,
			verbose: config.server.verbose,
			silent: config.server.silent,
			livereloadport: flags.livereloadport,
			dir: validatedServerPath,
			$pathProvided: true,
			$openLocation: true,
			browser: process.env.NODE_ENV === 'test' ? false : undefined,
		}

		try {
			// Delegate to existing server implementation
			await devdoc.init(serverFlags)
		} catch (error: unknown) {
			if (error instanceof Error) {
				logger.error({ error }, 'Failed to start server')
				this.error(error.message, { exit: 1 })
			}

			logger.error({ error }, 'Failed to start server')
			this.error('Unknown error starting server', { exit: 1 })
		}
	}

	/**
	 * Validate and normalize server path
	 */
	private validateServerPath(serverPath: string, cwd: string): string {
		if (!serverPath || serverPath === cwd) {
			return cwd
		}

		// If it's already an absolute path, just normalize it
		if (path.isAbsolute(serverPath)) {
			return path.normalize(serverPath)
		}

		// Otherwise, resolve it relative to cwd
		return path.resolve(cwd, serverPath)
	}
}
