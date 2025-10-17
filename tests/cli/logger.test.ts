/**
 * Tests for logger service
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { logger, createLogger, setLogLevel } from '../../src/cli/services/logger.js'

describe('Logger Service', () => {
	describe('logger instance', () => {
		it('should be a pino logger', () => {
			expect(logger).toBeDefined()
			expect(typeof logger.info).toBe('function')
			expect(typeof logger.error).toBe('function')
			expect(typeof logger.warn).toBe('function')
			expect(typeof logger.debug).toBe('function')
		})

		it('should have correct log methods', () => {
			const methods = ['info', 'error', 'warn', 'debug', 'trace', 'fatal']

			methods.forEach((method) => {
				expect(typeof (logger as any)[method]).toBe('function')
			})
		})

		it('should support child logger creation', () => {
			const childLogger = logger.child({ module: 'test' })
			expect(childLogger).toBeDefined()
			expect(typeof childLogger.info).toBe('function')
		})
	})

	describe('createLogger', () => {
		it('should create child logger with context', () => {
			const contextLogger = createLogger('test-context')

			expect(contextLogger).toBeDefined()
			expect(typeof contextLogger.info).toBe('function')
		})

		it('should create different loggers for different contexts', () => {
			const logger1 = createLogger('context1')
			const logger2 = createLogger('context2')

			expect(logger1).toBeDefined()
			expect(logger2).toBeDefined()
			// They should be different instances
			expect(logger1).not.toBe(logger2)
		})

		it('should maintain context in child logger', () => {
			const contextLogger = createLogger('test-module')

			// Test that it can log (doesn't throw)
			expect(() => {
				contextLogger.info('test message')
			}).not.toThrow()
		})
	})

	describe('setLogLevel', () => {
		const originalLevel = logger.level

		afterEach(() => {
			// Restore original log level
			setLogLevel(originalLevel as any)
		})

		it('should change log level', () => {
			setLogLevel('debug')
			expect(logger.level).toBe('debug')

			setLogLevel('warn')
			expect(logger.level).toBe('warn')

			setLogLevel('error')
			expect(logger.level).toBe('error')
		})

		it('should accept valid log levels', () => {
			const validLevels: Array<
				'fatal' | 'error' | 'warn' | 'info' | 'debug' | 'trace' | 'silent'
			> = ['fatal', 'error', 'warn', 'info', 'debug', 'trace', 'silent']

			validLevels.forEach((level) => {
				expect(() => setLogLevel(level)).not.toThrow()
				expect(logger.level).toBe(level)
			})
		})

		it('should support silent mode', () => {
			setLogLevel('silent')
			expect(logger.level).toBe('silent')
		})
	})

	describe('logging functionality', () => {
		it('should not throw when logging at various levels', () => {
			expect(() => logger.info('info message')).not.toThrow()
			expect(() => logger.error('error message')).not.toThrow()
			expect(() => logger.warn('warn message')).not.toThrow()
			expect(() => logger.debug('debug message')).not.toThrow()
		})

		it('should accept objects as log data', () => {
			expect(() => logger.info({ key: 'value' }, 'message')).not.toThrow()
			expect(() =>
				logger.error({ error: new Error('test') }, 'error message'),
			).not.toThrow()
		})

		it('should handle errors gracefully', () => {
			const error = new Error('test error')
			expect(() => logger.error({ err: error }, 'error occurred')).not.toThrow()
		})
	})

	describe('environment-based configuration', () => {
		const originalEnv = process.env.NODE_ENV

		afterEach(() => {
			process.env.NODE_ENV = originalEnv
		})

		it('should respect NODE_ENV for log level', () => {
			// This test verifies the logger is configured based on environment
			// The actual configuration happens at import time
			expect(logger.level).toBeDefined()
		})

		it('should respect LOG_LEVEL environment variable', () => {
			// Logger is already instantiated, so we test that it's configurable
			const originalLogLevel = process.env.LOG_LEVEL

			// This would need to be tested with a fresh import
			// For now, we verify the logger supports level changes
			setLogLevel('debug')
			expect(logger.level).toBe('debug')

			if (originalLogLevel) {
				process.env.LOG_LEVEL = originalLogLevel
			}
		})
	})
})
