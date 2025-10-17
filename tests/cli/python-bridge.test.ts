/**
 * Tests for Python bridge service
 */

import {
	describe,
	it,
	expect,
	beforeEach,
	afterEach,
	beforeAll,
	type Mock,
	vi,
} from 'vitest'
import {
	checkPythonAvailable,
	executePythonScript,
	parseMLIR,
} from '../../src/cli/services/python-bridge.js'
import { z } from 'zod'

// Mock execa module
vi.mock('execa', () => ({
	execa: vi.fn(),
}))

describe('Python Bridge Service', () => {
	let execaModule: Awaited<typeof import('execa')>
	let execaMock: Mock

	beforeAll(async () => {
		execaModule = await import('execa')
		execaMock = vi.mocked(execaModule.execa)
	})

	beforeEach(() => {
		vi.clearAllMocks()
	})

	describe('checkPythonAvailable', () => {
		it('should detect available Python', async () => {
			execaMock.mockResolvedValueOnce({
				stdout: 'Python 3.9.0',
				stderr: '',
			} as any)

			const result = await checkPythonAvailable()

			expect(result.available).toBe(true)
			expect(result.version).toContain('Python')
		})

		it('should handle Python not found', async () => {
			const error = new Error('Python not found') as NodeJS.ErrnoException
			error.code = 'ENOENT'
			execaMock.mockRejectedValueOnce(error)

			const result = await checkPythonAvailable()

			expect(result.available).toBe(false)
			expect(result.error).toBeDefined()
		})

		it('should handle timeout', async () => {
			const error = new Error('Timed out') as Error & { timedOut: boolean }
			error.timedOut = true
			execaMock.mockRejectedValueOnce(error)

			const result = await checkPythonAvailable()

			expect(result.available).toBe(false)
			expect(result.error).toBeDefined()
		})
	})

	describe('executePythonScript', () => {
		it('should execute Python script successfully', async () => {
			const mockOutput = { success: true, data: { result: 'test' } }

			execaMock.mockResolvedValueOnce({
				stdout: JSON.stringify(mockOutput),
				stderr: '',
			} as any)

			const result = await executePythonScript('/path/to/script.py', ['arg1'])

			expect(result).toEqual(mockOutput)
			expect(execaMock).toHaveBeenCalledWith(
				expect.stringMatching(/python3?/),
				['/path/to/script.py', 'arg1'],
				expect.objectContaining({
					timeout: 30000,
					maxBuffer: 50 * 1024 * 1024,
				}),
			)
		})

		it('should pass stdin input to script', async () => {
			const mockOutput = { success: true }

			execaMock.mockResolvedValueOnce({
				stdout: JSON.stringify(mockOutput),
			} as any)

			await executePythonScript('/path/to/script.py', [], {
				input: 'test input',
			})

			expect(execaMock).toHaveBeenCalledWith(
				expect.any(String),
				expect.any(Array),
				expect.objectContaining({
					input: 'test input',
				}),
			)
		})

		it('should validate output with schema', async () => {
			const mockOutput = { name: 'test', value: 42 }
			const schema = z.object({
				name: z.string(),
				value: z.number(),
			})

			execaMock.mockResolvedValueOnce({
				stdout: JSON.stringify(mockOutput),
			} as any)

			const result = await executePythonScript('/path/to/script.py', [], {
				schema,
			})

			expect(result).toEqual(mockOutput)
		})

		it('should throw on schema validation failure', async () => {
			const mockOutput = { name: 'test', value: 'not-a-number' }
			const schema = z.object({
				name: z.string(),
				value: z.number(),
			})

			execaMock.mockResolvedValueOnce({
				stdout: JSON.stringify(mockOutput),
			} as any)

			await expect(
				executePythonScript('/path/to/script.py', [], { schema }),
			).rejects.toThrow()
		})

		it('should handle Python script errors', async () => {
			const mockOutput = {
				error: 'Script error',
				message: 'Something went wrong',
			}

			execaMock.mockResolvedValueOnce({
				stdout: JSON.stringify(mockOutput),
			} as any)

			await expect(
				executePythonScript('/path/to/script.py'),
			).rejects.toThrow('Python script error')
		})

		it('should handle Python not found', async () => {
			const error = new Error('spawn python3 ENOENT') as NodeJS.ErrnoException
			error.code = 'ENOENT'
			execaMock.mockRejectedValueOnce(error)

			await expect(
				executePythonScript('/path/to/script.py'),
			).rejects.toThrow('Python not found')
		})

		it('should handle timeout', async () => {
			const error = new Error('Timeout') as Error & { timedOut: boolean }
			error.timedOut = true
			execaMock.mockRejectedValueOnce(error)

			await expect(
				executePythonScript('/path/to/script.py'),
			).rejects.toThrow('Python script timed out')
		})

		it('should handle maxBuffer exceeded', async () => {
			const error = new Error('maxBuffer exceeded')
			execaMock.mockRejectedValueOnce(error)

			await expect(
				executePythonScript('/path/to/script.py'),
			).rejects.toThrow('Python script output exceeded')
		})

		it('should use custom timeout', async () => {
			const mockOutput = { success: true }

			execaMock.mockResolvedValueOnce({
				stdout: JSON.stringify(mockOutput),
			} as any)

			await executePythonScript('/path/to/script.py', [], { timeout: 60000 })

			expect(execaMock).toHaveBeenCalledWith(
				expect.any(String),
				expect.any(Array),
				expect.objectContaining({
					timeout: 60000,
				}),
			)
		})
	})

	describe('parseMLIR', () => {
		it('should parse MLIR content successfully', async () => {
			const mockGraph = {
				id: 'test.mlir',
				nodes: [
					{
						id: 'node1',
						label: 'func.func',
						namespace: 'main',
						attrs: [],
						incomingEdges: [],
					},
				],
			}

			execaMock.mockResolvedValueOnce({
				stdout: JSON.stringify(mockGraph),
			} as any)

			const mlirContent = 'func.func @main() { return }'
			const result = await parseMLIR(
				mlirContent,
				'test.mlir',
				'/path/to/parse_mlir.py',
			)

			expect(result).toEqual(mockGraph)
			expect(execaMock).toHaveBeenCalledWith(
				expect.stringMatching(/python3?/),
				['/path/to/parse_mlir.py', 'test.mlir'],
				expect.objectContaining({
					input: mlirContent,
				}),
			)
		})

		it('should validate MLIR graph schema', async () => {
			const invalidGraph = {
				id: 'test.mlir',
				nodes: 'invalid', // Should be array
			}

			execaMock.mockResolvedValueOnce({
				stdout: JSON.stringify(invalidGraph),
			} as any)

			await expect(
				parseMLIR('content', 'test.mlir', '/path/to/script.py'),
			).rejects.toThrow()
		})

		it('should handle empty MLIR content', async () => {
			const mockError = {
				error: 'Empty input',
				message: 'No MLIR content provided',
			}

			execaMock.mockResolvedValueOnce({
				stdout: JSON.stringify(mockError),
			} as any)

			await expect(
				parseMLIR('', 'test.mlir', '/path/to/script.py'),
			).rejects.toThrow()
		})
	})

	describe('Python command selection', () => {
		const originalEnv = process.env.CONDA_DEFAULT_ENV

		afterEach(() => {
			if (originalEnv) {
				process.env.CONDA_DEFAULT_ENV = originalEnv
			} else {
				delete process.env.CONDA_DEFAULT_ENV
			}
		})

		it('should use python3 by default', async () => {
			delete process.env.CONDA_DEFAULT_ENV

			execaMock.mockResolvedValueOnce({
				stdout: '{"success": true}',
			} as any)

			await executePythonScript('/path/to/script.py')

			expect(execaMock).toHaveBeenCalledWith(
				'python3',
				expect.any(Array),
				expect.any(Object),
			)
		})

		it('should use python in conda environment', async () => {
			process.env.CONDA_DEFAULT_ENV = 'test-env'

			execaMock.mockResolvedValueOnce({
				stdout: '{"success": true}',
			} as any)

			await executePythonScript('/path/to/script.py')

			expect(execaMock).toHaveBeenCalledWith(
				'python',
				expect.any(Array),
				expect.any(Object),
			)
		})
	})
})
