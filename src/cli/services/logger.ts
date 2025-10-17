/**
 * Logger service for CLI
 * Uses pino for structured logging with pretty-printing in development
 */

import pino from 'pino'
import type { Logger } from 'pino'

// Determine if we're in development mode
const isDev = process.env.NODE_ENV !== 'production'

// Configure pino with pretty-printing in dev, JSON in production
const logger: Logger = pino({
	level: process.env.LOG_LEVEL || (isDev ? 'info' : 'warn'),
	transport: isDev
		? {
				target: 'pino-pretty',
				options: {
					colorize: true,
					translateTime: 'HH:MM:ss',
					ignore: 'pid,hostname',
					singleLine: false,
				},
			}
		: undefined,
})

/**
 * Create a child logger with a specific context
 */
export function createLogger(context: string): Logger {
	return logger.child({ context })
}

/**
 * Set the log level dynamically
 */
export function setLogLevel(level: pino.LevelWithSilent): void {
	logger.level = level
}

export { logger }
export type { Logger }
