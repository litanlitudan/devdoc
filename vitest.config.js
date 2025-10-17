import { defineConfig } from 'vitest/config'

export default defineConfig({
	test: {
		globals: true,
		environment: 'node',
		setupFiles: ['./tests/setup.ts'],
		coverage: {
			provider: 'v8',
			reporter: ['text', 'lcov', 'html'],
			exclude: [
				'node_modules/',
				'tests/',
				'scripts/',
				'*.config.js',
				'dist/',
				'third_party/',
			],
		},
		include: [
			'tests/**/*.test.js',
			'tests/**/*.test.ts',
			'tests/**/*.test.modern.js',
		],
		// Migrate from AVA configuration
		threads: true,
		isolate: true,
	},
})