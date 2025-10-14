#!/usr/bin/env node

/**
 * Migration script to convert CommonJS to ES modules
 * Run with: node scripts/migrate-to-esm.js
 */

import { readFile, writeFile, readdir } from 'fs/promises'
import { join, dirname, extname } from 'path'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

/**
 * Transform CommonJS require() to ES import
 */
function transformRequires(content) {
	// Handle require statements
	// const x = require('y') → import x from 'y'
	content = content.replace(
		/const\s+(\w+)\s*=\s*require\s*\(\s*['"]([^'"]+)['"]\s*\)/g,
		"import $1 from '$2'"
	)

	// const {x, y} = require('z') → import {x, y} from 'z'
	content = content.replace(
		/const\s+\{([^}]+)\}\s*=\s*require\s*\(\s*['"]([^'"]+)['"]\s*\)/g,
		"import {$1} from '$2'"
	)

	// Handle dynamic path.join(__dirname, ...) patterns
	content = content.replace(
		/require\s*\(\s*path\.join\s*\(\s*__dirname\s*,\s*['"]([^'"]+)['"]\s*\)\s*\)/g,
		"await import('./$1')"
	)

	// Update Node.js built-in imports to use 'node:' prefix
	const nodeBuiltins = [
		'fs',
		'path',
		'http',
		'https',
		'crypto',
		'util',
		'stream',
		'os',
		'url',
		'querystring',
		'child_process',
	]

	nodeBuiltins.forEach((builtin) => {
		const regex = new RegExp(`from ['"]${builtin}['"]`, 'g')
		content = content.replace(regex, `from 'node:${builtin}'`)
	})

	return content
}

/**
 * Transform module.exports to ES export
 */
function transformExports(content) {
	// module.exports = x → export default x
	content = content.replace(/module\.exports\s*=\s*([^;]+);?/g, 'export default $1')

	// module.exports = { → export {
	content = content.replace(/module\.exports\s*=\s*\{/g, 'export {')

	// exports.x = y → export const x = y
	content = content.replace(/exports\.(\w+)\s*=\s*([^;]+);?/g, 'export const $1 = $2')

	return content
}

/**
 * Add __dirname and __filename equivalents for ES modules
 */
function addDirnameFilename(content) {
	// Check if file uses __dirname or __filename
	if (content.includes('__dirname') || content.includes('__filename')) {
		// Add imports at the top of the file after other imports
		const importLines = [
			"import { fileURLToPath } from 'node:url'",
			"import { dirname } from 'node:path'",
			'',
			'const __filename = fileURLToPath(import.meta.url)',
			'const __dirname = dirname(__filename)',
			'',
		]

		// Find the last import statement
		const importMatches = content.match(/^import .*/gm)
		if (importMatches && importMatches.length > 0) {
			const lastImport = importMatches[importMatches.length - 1]
			const insertPosition = content.indexOf(lastImport) + lastImport.length
			content =
				`${content.slice(0, insertPosition) 
				}\n${ 
				importLines.join('\n') 
				}${content.slice(insertPosition)}`
		} else {
			// No imports found, add at the beginning
			content = `${importLines.join('\n')  }\n${  content}`
		}
	}

	return content
}

/**
 * Update file extensions in import statements
 */
function updateImportExtensions(content) {
	// Add .js extension to relative imports that don't have an extension
	content = content.replace(
		/from\s+['"](\.\.?\/[^'"]+)(?<!\.js)(?<!\.json)(?<!\.css)(?<!\.html)['"]/g,
		"from '$1.js'"
	)

	return content
}

/**
 * Process a single JavaScript file
 */
async function processFile(filePath) {
	try {
		const content = await readFile(filePath, 'utf-8')
		let transformed = content

		// Skip if already using ES modules
		if (content.includes('import ') || content.includes('export ')) {
			console.log(`⏭️  Skipping ${filePath} (already ES module)`)
			return
		}

		// Apply transformations
		transformed = transformRequires(transformed)
		transformed = transformExports(transformed)
		transformed = addDirnameFilename(transformed)
		transformed = updateImportExtensions(transformed)

		// Write back if changed
		if (transformed !== content) {
			await writeFile(filePath, transformed)
			console.log(`✅ Converted ${filePath}`)
		} else {
			console.log(`⏭️  No changes needed for ${filePath}`)
		}
	} catch (error) {
		console.error(`❌ Error processing ${filePath}:`, error.message)
	}
}

/**
 * Process all JavaScript files in a directory recursively
 */
async function processDirectory(dirPath) {
	const entries = await readdir(dirPath, { withFileTypes: true })

	for (const entry of entries) {
		const fullPath = join(dirPath, entry.name)

		// Skip node_modules and .git
		if (entry.name === 'node_modules' || entry.name === '.git') {
			continue
		}

		if (entry.isDirectory()) {
			await processDirectory(fullPath)
		} else if (entry.isFile() && extname(entry.name) === '.js') {
			await processFile(fullPath)
		}
	}
}

/**
 * Main migration function
 */
async function migrate() {
	console.log('🚀 Starting CommonJS to ES modules migration...\n')

	const projectRoot = join(__dirname, '..')

	// Process lib directory
	console.log('📂 Processing lib directory...')
	await processDirectory(join(projectRoot, 'lib'))

	// Process test directory
	console.log('\n📂 Processing tests directory...')
	await processDirectory(join(projectRoot, 'tests'))

	// Update package.json type field (already done manually)
	console.log('\n📦 Package.json already updated with "type": "module"')

	console.log('\n✨ Migration complete!')
	console.log('\n📝 Next steps:')
	console.log('1. Review the changes')
	console.log('2. Run npm install to update dependencies')
	console.log('3. Test the application')
	console.log('4. Fix any runtime issues')
}

// Run migration
migrate().catch(console.error)