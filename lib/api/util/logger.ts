/**
 * Shared logger for API
 * Re-exports the CLI logger for consistency across the codebase
 */

export { logger, createLogger, setLogLevel } from '../../../src/cli/services/logger.js'
export type { Logger } from 'pino'
