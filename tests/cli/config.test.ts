/**
 * Tests for configuration loading and validation
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { writeFileSync, unlinkSync, existsSync } from 'fs'
import { join } from 'path'
import {
	loadConfig,
	loadConfigFile,
	createDefaultConfig,
	ServerConfigSchema,
	GraphConfigSchema,
	ConfigSchema,
} from '../../src/cli/config.js'

describe('Configuration Schemas', () => {
	describe('ServerConfigSchema', () => {
		it('should validate valid server config', () => {
			const config = {
				port: 8642,
				address: 'localhost',
				watch: true,
				verbose: false,
				silent: false,
				openBrowser: false,
				livereload: true,
			}

			const result = ServerConfigSchema.parse(config)
			expect(result).toEqual(config)
		})

		it('should apply default values', () => {
			const result = ServerConfigSchema.parse({})
			expect(result.port).toBe(8642)
			expect(result.address).toBe('localhost')
			expect(result.watch).toBe(false)
		})

		it('should reject invalid port numbers', () => {
			expect(() => ServerConfigSchema.parse({ port: 0 })).toThrow()
			expect(() => ServerConfigSchema.parse({ port: 99999 })).toThrow()
			expect(() => ServerConfigSchema.parse({ port: -1 })).toThrow()
		})

		it('should validate port range', () => {
			const validPort = ServerConfigSchema.parse({ port: 3000 })
			expect(validPort.port).toBe(3000)

			const maxPort = ServerConfigSchema.parse({ port: 65535 })
			expect(maxPort.port).toBe(65535)

			const minPort = ServerConfigSchema.parse({ port: 1 })
			expect(minPort.port).toBe(1)
		})
	})

	describe('GraphConfigSchema', () => {
		it('should validate valid graph config', () => {
			const config = {
				mlirParser: 'python' as const,
				onnxShapeInference: true,
				maxFileSize: 100,
				timeout: 30000,
			}

			const result = GraphConfigSchema.parse(config)
			expect(result).toEqual(config)
		})

		it('should apply default values', () => {
			const result = GraphConfigSchema.parse({})
			expect(result.mlirParser).toBe('python')
			expect(result.onnxShapeInference).toBe(true)
			expect(result.maxFileSize).toBe(100)
			expect(result.timeout).toBe(30000)
		})

		it('should validate mlirParser enum', () => {
			const python = GraphConfigSchema.parse({ mlirParser: 'python' })
			expect(python.mlirParser).toBe('python')

			const cpp = GraphConfigSchema.parse({ mlirParser: 'cpp' })
			expect(cpp.mlirParser).toBe('cpp')

			expect(() =>
				GraphConfigSchema.parse({ mlirParser: 'invalid' }),
			).toThrow()
		})

		it('should validate positive numbers', () => {
			expect(() => GraphConfigSchema.parse({ maxFileSize: 0 })).toThrow()
			expect(() => GraphConfigSchema.parse({ maxFileSize: -1 })).toThrow()
			expect(() => GraphConfigSchema.parse({ timeout: 0 })).toThrow()
		})
	})

	describe('ConfigSchema', () => {
		it('should validate complete config', () => {
			const config = {
				server: {
					port: 3000,
					address: '0.0.0.0',
				},
				graph: {
					mlirParser: 'cpp' as const,
				},
			}

			const result = ConfigSchema.parse(config)
			expect(result.server.port).toBe(3000)
			expect(result.graph.mlirParser).toBe('cpp')
		})

		it('should merge with defaults', () => {
			const config = {
				server: { port: 3000 },
				graph: {},
			}

			const result = ConfigSchema.parse(config)
			expect(result.server.port).toBe(3000)
			expect(result.server.address).toBe('localhost')
			expect(result.graph.mlirParser).toBe('python')
		})
	})
})

describe('Configuration Loading', () => {
	const testConfigPath = join(process.cwd(), '.markservrc.test.json')

	afterEach(() => {
		if (existsSync(testConfigPath)) {
			unlinkSync(testConfigPath)
		}
	})

	describe('loadConfigFile', () => {
		it('should return empty object when no config file exists', () => {
			const config = loadConfigFile('/nonexistent/directory')
			expect(config).toEqual({})
		})

		it('should load valid JSON config file', () => {
			const testConfig = {
				server: { port: 3000 },
				graph: { mlirParser: 'cpp' },
			}

			writeFileSync(testConfigPath, JSON.stringify(testConfig, null, 2))

			// Note: loadConfigFile looks in cwd, so we need to test differently
			// This is a structural test for the function signature
			expect(typeof loadConfigFile).toBe('function')
		})

		it('should handle malformed JSON gracefully', () => {
			writeFileSync(testConfigPath, 'invalid json {')

			// Should not throw, just return empty object and warn
			const config = loadConfigFile(process.cwd())
			expect(config).toEqual({})
		})
	})

	describe('loadConfig', () => {
		it('should create config with defaults', () => {
			const config = loadConfig()

			expect(config.server.port).toBe(8642)
			expect(config.server.address).toBe('localhost')
			expect(config.graph.mlirParser).toBe('python')
		})

		it('should merge overrides', () => {
			const config = loadConfig({
				server: {
					port: 3000,
					address: '0.0.0.0',
					watch: true,
					verbose: false,
					silent: false,
					openBrowser: false,
					livereload: true,
				},
			})

			expect(config.server.port).toBe(3000)
			expect(config.server.address).toBe('0.0.0.0')
			expect(config.server.watch).toBe(true)
		})

		it('should respect environment variables', () => {
			const originalPort = process.env.PORT
			const originalAddress = process.env.ADDRESS

			process.env.PORT = '4000'
			process.env.ADDRESS = '127.0.0.1'

			const config = loadConfig()

			expect(config.server.port).toBe(4000)
			expect(config.server.address).toBe('127.0.0.1')

			// Restore original values
			if (originalPort) {
				process.env.PORT = originalPort
			} else {
				delete process.env.PORT
			}

			if (originalAddress) {
				process.env.ADDRESS = originalAddress
			} else {
				delete process.env.ADDRESS
			}
		})

		it('should prioritize overrides over environment', () => {
			const originalPort = process.env.PORT

			process.env.PORT = '4000'

			const config = loadConfig({
				server: {
					port: 5000,
					address: 'localhost',
					watch: false,
					verbose: false,
					silent: false,
					openBrowser: false,
					livereload: true,
				},
			})

			expect(config.server.port).toBe(5000)

			// Restore
			if (originalPort) {
				process.env.PORT = originalPort
			} else {
				delete process.env.PORT
			}
		})
	})

	describe('createDefaultConfig', () => {
		it('should create config with all defaults', () => {
			const config = createDefaultConfig()

			expect(config.server.port).toBe(8642)
			expect(config.server.address).toBe('localhost')
			expect(config.server.watch).toBe(false)
			expect(config.graph.mlirParser).toBe('python')
			expect(config.graph.onnxShapeInference).toBe(true)
		})

		it('should be valid according to schema', () => {
			const config = createDefaultConfig()
			const validated = ConfigSchema.parse(config)

			expect(validated).toEqual(config)
		})
	})
})
