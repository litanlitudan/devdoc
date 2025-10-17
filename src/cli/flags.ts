/**
 * Shared CLI Flag Definitions
 *
 * This module defines the canonical CLI flags for devdoc.
 * Both the legacy (yargs) and new (oclif) CLIs should implement these flags
 * to maintain backward compatibility.
 */

/**
 * Common flag definitions shared between CLI implementations
 */
export const COMMON_FLAGS = {
	/** HTTP server port */
	port: {
		type: 'integer' as const,
		alias: 'p',
		default: 8642,
		description: 'Port to run server on',
	},
	/** Server bind address */
	address: {
		type: 'string' as const,
		alias: 'a',
		default: 'localhost',
		description: 'Address to bind server to',
	},
	/** Enable file watching and live reload */
	watch: {
		type: 'boolean' as const,
		alias: 'w',
		default: false,
		description: 'Enable file watching and live reload',
	},
	/** Verbose logging */
	verbose: {
		type: 'boolean' as const,
		alias: 'v',
		default: false,
		description: 'Verbose output',
	},
	/** Silent mode (suppress output) */
	silent: {
		type: 'boolean' as const,
		alias: 's',
		default: false,
		description: 'Silent mode',
	},
	/** LiveReload server port */
	livereloadport: {
		type: 'integer' as const,
		alias: 'b',
		default: 35729,
		description: 'Port for LiveReload server',
	},
} as const

/**
 * Type-safe flag definitions
 */
export type FlagDefinition = {
	type: 'string' | 'integer' | 'boolean'
	alias: string
	default: string | number | boolean
	description: string
}

/**
 * Flag names for reference
 */
export type FlagName = keyof typeof COMMON_FLAGS

/**
 * Validates that flag implementations match the canonical definitions
 */
export function validateFlagParity(
	implementation: Record<string, any>,
): boolean {
	for (const [name, def] of Object.entries(COMMON_FLAGS)) {
		if (!(name in implementation)) {
			console.warn(`Missing flag: ${name}`)
			return false
		}

		const impl = implementation[name]

		// Check alias
		if (impl.alias !== def.alias) {
			console.warn(
				`Flag ${name}: alias mismatch (expected ${def.alias}, got ${impl.alias})`,
			)
			return false
		}

		// Check default value
		if (impl.default !== def.default) {
			console.warn(
				`Flag ${name}: default mismatch (expected ${def.default}, got ${impl.default})`,
			)
			return false
		}
	}

	return true
}

/**
 * Migration guide for flag implementations
 */
export const MIGRATION_NOTES = {
	port: 'Port must be integer type for consistency',
	livereloadport: 'Maintains -b alias for backward compatibility',
	watch: 'Controls both file watching and LiveReload activation',
	verbose: 'Conflicts with silent - silent takes precedence',
	silent: 'Suppresses all output including splash screen',
} as const
