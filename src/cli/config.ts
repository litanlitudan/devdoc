/**
 * Configuration schemas and loaders for devdoc CLI
 * Uses zod for validation and type inference
 */

import { z } from 'zod'
import { existsSync, readFileSync } from 'fs'
import { join } from 'path'

/**
 * Server configuration schema
 */
export const ServerConfigSchema = z.object({
	port: z.number().int().min(1).max(65535).default(8642),
	address: z.string().default('localhost'),
	watch: z.boolean().default(false),
	verbose: z.boolean().default(false),
	silent: z.boolean().default(false),
	openBrowser: z.boolean().default(false),
	livereload: z.boolean().default(true),
})

/**
 * MLIR/ONNX processing configuration
 */
export const GraphConfigSchema = z.object({
	mlirParser: z.enum(['python', 'cpp']).default('python'),
	onnxShapeInference: z.boolean().default(true),
	maxFileSize: z.number().int().positive().default(100), // MB
	timeout: z.number().int().positive().default(30000), // ms
})

/**
 * Complete configuration schema
 */
export const ConfigSchema = z.object({
	server: ServerConfigSchema,
	graph: GraphConfigSchema,
})

export type ServerConfig = z.infer<typeof ServerConfigSchema>
export type GraphConfig = z.infer<typeof GraphConfigSchema>
export type Config = z.infer<typeof ConfigSchema>

export interface ConfigOverrides {
	server?: Partial<ServerConfig>
	graph?: Partial<GraphConfig>
}

/**
 * Load configuration from .devdocrc file (JSON or JS)
 */
export function loadConfigFile(cwd: string = process.cwd()): Partial<Config> {
	const configPaths = [
		join(cwd, '.devdocrc.json'),
		join(cwd, '.devdocrc'),
	]

	for (const configPath of configPaths) {
		if (existsSync(configPath)) {
			try {
				const content = readFileSync(configPath, 'utf-8')
				return JSON.parse(content)
			} catch (error) {
				console.warn(`Failed to parse config file ${configPath}:`, error)
			}
		}
	}

	return {}
}

/**
 * Load and validate complete configuration
 * Merges defaults, config file, and environment variables
 */
export function loadConfig(overrides?: ConfigOverrides): Config {
	const fileConfig = loadConfigFile()

	// Merge configurations with priority: overrides > env > file > defaults
	const rawConfig = {
		server: {
			...fileConfig.server,
			port:
				overrides?.server?.port ??
				(process.env.PORT ? parseInt(process.env.PORT) : undefined) ??
				fileConfig.server?.port,
			address:
				overrides?.server?.address ??
				process.env.ADDRESS ??
				fileConfig.server?.address,
			watch: overrides?.server?.watch ?? fileConfig.server?.watch,
			verbose: overrides?.server?.verbose ?? fileConfig.server?.verbose,
			silent: overrides?.server?.silent ?? fileConfig.server?.silent,
			openBrowser:
				overrides?.server?.openBrowser ?? fileConfig.server?.openBrowser,
			livereload:
				overrides?.server?.livereload ?? fileConfig.server?.livereload,
		},
		graph: {
			...fileConfig.graph,
			...overrides?.graph,
		},
	}

	// Validate and return with defaults applied
	return ConfigSchema.parse(rawConfig)
}

/**
 * Create default configuration (useful for testing)
 */
export function createDefaultConfig(): Config {
	return ConfigSchema.parse({
		server: {},
		graph: {},
	})
}
