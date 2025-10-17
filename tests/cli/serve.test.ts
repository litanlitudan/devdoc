/**
 * Tests for the serve command
 * Focuses on command metadata and flag handling
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest'
import path from 'node:path'

vi.mock('../../lib/server.js', () => ({
	default: {
		init: vi.fn().mockResolvedValue(undefined),
	},
}))

vi.mock('../../src/cli/services/logger.js', () => {
	const logger = {
		info: vi.fn(),
		error: vi.fn(),
		debug: vi.fn(),
		child: vi.fn(() => ({
			info: vi.fn(),
			error: vi.fn(),
			debug: vi.fn(),
		})),
	}

	return {
		logger,
		createLogger: vi.fn(() => logger),
		setLogLevel: vi.fn(),
	}
})

import Serve from '../../src/cli/commands/serve.js'
import devdoc from '../../lib/server.js'

const devdocInit = (devdoc.init ?? devdoc.default?.init) as Mock

describe('Serve command metadata', () => {
	it('has expected description and examples', () => {
		expect(Serve.description).toBe('Start the devdoc development server')
		expect(Serve.examples).toContain('<%= config.bin %> <%= command.id %>')
	})

	it('exposes port flag defaults', () => {
		expect(Serve.flags.port.default).toBe(8642)
		expect(Serve.flags.port.description).toContain('Port')
	})

	it('defines address flag with localhost default', () => {
		expect(Serve.flags.address.default).toBe('localhost')
	})

	it('enables watch/verbose/silent flags', () => {
		expect(Serve.flags.watch.default).toBe(false)
		expect(Serve.flags.verbose.default).toBe(false)
		expect(Serve.flags.silent.default).toBe(false)
	})

	it('accepts optional path argument', () => {
		expect(Serve.args.path?.required).toBe(false)
		expect(Serve.args.path?.description).toContain('Directory or file')
	})
})

describe('Serve command execution', () => {
	beforeEach(() => {
		vi.clearAllMocks()
	})

	it('passes flags to devdoc.init', async () => {
		await Serve.run(['./docs', '--port', '3000', '--watch'])

		expect(devdocInit).toHaveBeenCalledTimes(1)
		expect(devdocInit).toHaveBeenCalledWith(
			expect.objectContaining({
				port: '3000',
				address: 'localhost',
				watch: true,
				dir: path.resolve(process.cwd(), './docs'),
				livereloadport: 35729,
				$pathProvided: true,
				$openLocation: true,
			}),
		)
	})

	it('normalizes paths when none provided', async () => {
		await Serve.run([])

		expect(devdocInit).toHaveBeenCalledWith(
			expect.objectContaining({
				dir: process.cwd(),
			}),
		)
	})

	it('surface errors from devdoc.init', async () => {
		devdocInit.mockRejectedValueOnce(new Error('Failed to start'))

		await expect(Serve.run(['--port', '9999'])).rejects.toThrow(
			'Failed to start',
		)
	})
})
