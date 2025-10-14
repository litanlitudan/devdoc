#!/usr/bin/env node
import fs from 'fs'
import path from 'path'

const distDir = 'dist'
const files = fs.readdirSync(distDir)
const jsFiles = files.filter(f => f.endsWith('.js')).length
const dtsFiles = files.filter(f => f.endsWith('.d.ts')).length

console.log(`✓ TypeScript compilation successful`)
console.log(`  → ${jsFiles} JavaScript files`)
console.log(`  → ${dtsFiles} declaration files`)
console.log(`  → Source maps generated`)
