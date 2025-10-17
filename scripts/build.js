#!/usr/bin/env node

/**
 * Build script for devdoc
 * Handles compilation, bundling, and optimization
 */

import { mkdir, rm, copyFile, readFile, writeFile, readdir } from 'node:fs/promises'
import { join, dirname, basename } from 'node:path'
import { fileURLToPath } from 'node:url'
import { existsSync } from 'node:fs'
import { exec } from 'node:child_process'
import { promisify } from 'node:util'
import chalk from 'chalk'
import ora from 'ora'

const execAsync = promisify(exec)
const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)
const rootDir = join(__dirname, '..')

/**
 * Build steps
 */
const buildSteps = {
	/**
	 * Clean build directory
	 */
	async clean() {
		const distDir = join(rootDir, 'dist')
		if (existsSync(distDir)) {
			await rm(distDir, { recursive: true, force: true })
		}
		await mkdir(distDir, { recursive: true })
		await mkdir(join(distDir, 'lib'), { recursive: true })
		await mkdir(join(distDir, 'lib/templates'), { recursive: true })
		await mkdir(join(distDir, 'lib/icons'), { recursive: true })
	},

	/**
	 * Copy source files
	 */
	async copySource() {
		const filesToCopy = [
			// Main library files
			'lib/cli.js',
			'lib/server.js',
			'lib/readme.js',
			'lib/splash.js',
			'lib/cli-defs.js',
			'lib/cli-help.txt',

			// Template files
			'lib/templates/markdown.html',
			'lib/templates/directory.html',
			'lib/templates/error.html',
			'lib/templates/devdoc.css',
			'lib/templates/highlight-js-github-gist.css',
			'lib/templates/github.less',

			// Icon files
			'lib/icons/material-icons.json',
			'lib/icons/icons.css',
			'lib/icons/devdoc.svg',

			// Root files
			'package.json',
			'README.md',
			'LICENSE',
			'CHANGELOG.md',
		]

		for (const file of filesToCopy) {
			const src = join(rootDir, file)
			const dest = join(rootDir, 'dist', file)

			if (existsSync(src)) {
				// Ensure destination directory exists
				await mkdir(dirname(dest), { recursive: true })
				await copyFile(src, dest)
			}
		}

		// Copy media directory
		const mediaDir = join(rootDir, 'media')
		const distMediaDir = join(rootDir, 'dist', 'media')
		if (existsSync(mediaDir)) {
			await mkdir(distMediaDir, { recursive: true })
			const files = await readdir(mediaDir)
			for (const file of files) {
				await copyFile(join(mediaDir, file), join(distMediaDir, file))
			}
		}
	},

	/**
	 * Transform CommonJS to ES modules (if needed)
	 */
	async transformModules() {
		// Check if files are already ES modules
		const serverPath = join(rootDir, 'dist', 'lib', 'server.js')
		const content = await readFile(serverPath, 'utf-8')

		if (content.includes('require(') && !content.includes('import ')) {
			console.log(chalk.yellow('  Converting CommonJS to ES modules...'))
			// Run migration script on dist files
			const { stdout, stderr } = await execAsync(
				`node ${join(rootDir, 'scripts', 'migrate-to-esm.js')} ${join(rootDir, 'dist')}`,
				{ cwd: rootDir }
			)
			if (stderr) console.error(stderr)
		}
	},

	/**
	 * Generate TypeScript declarations
	 */
	async generateTypes() {
		// Copy TypeScript declarations
		const typesDir = join(rootDir, 'types')
		const distTypesDir = join(rootDir, 'dist', 'types')

		if (existsSync(typesDir)) {
			await mkdir(distTypesDir, { recursive: true })
			const files = await readdir(typesDir)
			for (const file of files) {
				if (file.endsWith('.d.ts')) {
					await copyFile(join(typesDir, file), join(distTypesDir, file))
				}
			}
		}

		// Generate types from JSDoc comments
		try {
			await execAsync('npx tsc --allowJs --declaration --emitDeclarationOnly --outDir dist/types lib/*.js', {
				cwd: rootDir,
			})
		} catch (error) {
			// TypeScript might not be configured, that's okay
			console.log(chalk.gray('  Skipping auto-generated types (TypeScript not configured)'))
		}
	},

	/**
	 * Optimize for production
	 */
	async optimize() {
		// Update package.json in dist
		const pkgPath = join(rootDir, 'dist', 'package.json')
		const pkg = JSON.parse(await readFile(pkgPath, 'utf-8'))

		// Remove dev dependencies and scripts not needed in production
		delete pkg.devDependencies
		delete pkg.scripts.prepare
		delete pkg.scripts.prepublishOnly
		delete pkg.husky
		delete pkg['lint-staged']

		// Add dist-specific fields
		pkg.main = 'lib/server.js'
		pkg.bin = {
			devdoc: 'lib/cli.js',
			readme: 'lib/readme.js',
		}

		// Write optimized package.json
		await writeFile(pkgPath, JSON.stringify(pkg, null, 2))
	},

	/**
	 * Create executable scripts
	 */
	async createExecutables() {
		// Ensure CLI files have shebang and are executable
		const cliFiles = ['lib/cli.js', 'lib/readme.js']

		for (const file of cliFiles) {
			const filePath = join(rootDir, 'dist', file)
			if (existsSync(filePath)) {
				const content = await readFile(filePath, 'utf-8')
				if (!content.startsWith('#!/usr/bin/env node')) {
					await writeFile(filePath, `#!/usr/bin/env node\n\n${content}`)
				}
			}
		}
	},

	/**
	 * Run tests on built version
	 */
	async runTests() {
		try {
			const { stdout } = await execAsync('npm test', { cwd: rootDir })
			return true
		} catch (error) {
			console.error(chalk.red('Tests failed!'))
			return false
		}
	},

	/**
	 * Create tarball for npm
	 */
	async createTarball() {
		const { stdout } = await execAsync('npm pack', { cwd: join(rootDir, 'dist') })
		const tarballName = stdout.trim()
		console.log(chalk.green(`  Created ${tarballName}`))
	},
}

/**
 * Main build function
 */
async function build() {
	console.log(chalk.cyan.bold('\n🔨 Building devdoc...\n'))

	const steps = [
		{ name: 'Cleaning build directory', fn: buildSteps.clean },
		{ name: 'Copying source files', fn: buildSteps.copySource },
		{ name: 'Transforming modules', fn: buildSteps.transformModules },
		{ name: 'Generating TypeScript declarations', fn: buildSteps.generateTypes },
		{ name: 'Optimizing for production', fn: buildSteps.optimize },
		{ name: 'Creating executable scripts', fn: buildSteps.createExecutables },
		{ name: 'Running tests', fn: buildSteps.runTests },
	]

	let success = true

	for (const step of steps) {
		const spinner = ora(step.name).start()
		try {
			const result = await step.fn()
			if (result === false) {
				spinner.fail(step.name)
				success = false
				break
			}
			spinner.succeed(step.name)
		} catch (error) {
			spinner.fail(`${step.name}: ${error.message}`)
			console.error(chalk.red(error.stack))
			success = false
			break
		}
	}

	if (success) {
		console.log(chalk.green.bold('\n✅ Build completed successfully!\n'))
		console.log(chalk.cyan('Output directory: ') + chalk.white(join(rootDir, 'dist')))
		console.log(chalk.cyan('\nNext steps:'))
		console.log(chalk.white('  1. Install locally: ') + chalk.yellow.bold('npm link'))
		console.log(chalk.gray('     This will install devdoc globally for local development'))
		console.log(chalk.white('  2. Test the build: ') + chalk.gray('devdoc ./README.md'))
		console.log(chalk.white('  3. Publish to npm: ') + chalk.gray('cd dist && npm publish'))
		console.log(chalk.white('  4. Create tarball: ') + chalk.gray('cd dist && npm pack'))
	} else {
		console.log(chalk.red.bold('\n❌ Build failed!\n'))
		process.exit(1)
	}
}

// Handle errors
process.on('unhandledRejection', (error) => {
	console.error(chalk.red('Unhandled error:'), error)
	process.exit(1)
})

// Run build
build().catch((error) => {
	console.error(chalk.red('Build error:'), error)
	process.exit(1)
})