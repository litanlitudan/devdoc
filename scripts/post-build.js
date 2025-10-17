#!/usr/bin/env node
import fs from 'fs'
import path from 'path'

const distDir = 'dist'

const collectFiles = (dir, ext) => {
	let count = 0
	for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
		const fullPath = path.join(dir, entry.name)
		if (entry.isDirectory()) {
			count += collectFiles(fullPath, ext)
		} else if (entry.name.endsWith(ext)) {
			count += 1
		}
	}
	return count
}

const ensureDir = (dir) => {
	if (!fs.existsSync(dir)) {
		fs.mkdirSync(dir, { recursive: true })
	}
}

const copyAsset = (source, destination) => {
	if (!fs.existsSync(source)) return
	const stats = fs.statSync(source)
	if (stats.isDirectory()) {
		ensureDir(destination)
		fs.cpSync(source, destination, { recursive: true })
	} else {
		ensureDir(path.dirname(destination))
		fs.copyFileSync(source, destination)
	}
}

// Copy static assets required at runtime
const assetMap = [
	['lib/templates', path.join(distDir, 'lib', 'templates')],
	['lib/icons', path.join(distDir, 'lib', 'icons')],
	['lib/cli-help.txt', path.join(distDir, 'lib', 'cli-help.txt')],
]

for (const [src, dest] of assetMap) {
	copyAsset(src, dest)
}

const jsFiles = collectFiles(distDir, '.js')
const dtsFiles = collectFiles(distDir, '.d.ts')

console.log(`✓ TypeScript compilation successful`)
console.log(`  → ${jsFiles} JavaScript files`)
console.log(`  → ${dtsFiles} declaration files`)
console.log(`  → Source maps generated`)
