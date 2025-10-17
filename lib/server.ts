'use strict'

import http, { Server as HttpServer } from 'node:http'
import path from 'node:path'
import fs from 'node:fs'

import chalk from 'chalk'
import open from 'open'
import express, { Request, Response, NextFunction, Application } from 'express'
import compression from 'compression'
import less from 'less'
import { WebSocketServer } from 'ws'
import chokidar, { FSWatcher } from 'chokidar'
import handlebars from 'handlebars'
import MarkdownIt from 'markdown-it'
import type Token from 'markdown-it/lib/token.mjs'
import type Renderer from 'markdown-it/lib/renderer.mjs'
import type { Options, PluginSimple } from 'markdown-it'
import mdItAnchor from 'markdown-it-anchor'
// These are CommonJS modules that need special handling in ES modules
import { createRequire } from 'module'
const require = createRequire(import.meta.url)

const mdItTaskLists = require('markdown-it-task-lists')
const mdItEmojiAll = require('markdown-it-emoji')
const mdItEmoji = mdItEmojiAll.full || mdItEmojiAll
const mdItTOC = require('markdown-it-table-of-contents')

import mdItHLJS from 'markdown-it-highlightjs'
import mdItMathJax from 'markdown-it-mathjax3'
import emojiRegexCreator from 'emoji-regex'
import isOnline from 'is-online'
import hljs from 'highlight.js/lib/core'
import hljsMLIR from 'highlightjs-mlir'
import { fileURLToPath } from 'node:url'
import { convertMLIRToGraph } from './mlir-to-graph.js'
import { convertONNXToGraph } from './onnx-to-graph.js'
import { dirname } from 'node:path'
import mime from 'mime-types'
import multer from 'multer'

import type {
	Flags,
	MarkservService,
	FileInfo,
	DirectoryInfo,
	Breadcrumb,
	ImplantOptions,
	ImplantHandlers,
	FileTypes,
	HttpServerResult
} from './types.js'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

const emojiRegex = emojiRegexCreator()

// Register MLIR language for highlight.js
hljs.registerLanguage('mlir', hljsMLIR)

// Analytics tracking interfaces and storage
interface PageVisit {
	timestamp: Date
	path: string
	method: string
	userAgent: string
	referer: string
	ip: string
	statusCode?: number
	fileType?: string
}

interface AnalyticsStats {
	totalVisits: number
	uniquePaths: number
	uniqueVisitors: number
	topPaths: { path: string; count: number }[]
	recentVisits: PageVisit[]
	visitsByFileType: { type: string; count: number }[]
	visitsByHour: { hour: number; count: number }[]
	topVisitors: { identifier: string; visits: number; lastSeen: Date }[]
}

// In-memory analytics storage (will be cleared on server restart)
const analyticsData: PageVisit[] = []
const MAX_ANALYTICS_RECORDS = 10000 // Limit to prevent memory overflow

interface StyleColors {
	link: typeof chalk
	github: typeof chalk
	address: typeof chalk
	port: typeof chalk
	pid: typeof chalk
}

const style: StyleColors = {
	link: chalk.blueBright.underline.italic,
	github: chalk.blue.underline.italic,
	address: chalk.greenBright.underline.italic,
	port: chalk.reset.cyanBright,
	pid: chalk.reset.cyanBright
}

// Simple implant replacement for template variable substitution
const processTemplate = async (
	html: string,
	handlers: Record<string, (varName: string) => Promise<string | false>>
): Promise<string> => {
	let result = html
	// Match both {{varName}} and {varName} syntax
	const doubleRegex = /\{\{\s*(\w+)\s*\}\}/g
	const singleRegex = /\{(\w+)\}/g

	// Process double curly brace syntax {{varName}}
	const doubleMatches = [...result.matchAll(doubleRegex)]
	for (const match of doubleMatches) {
		const [fullMatch, varName] = match
		if (handlers[varName]) {
			const value = await handlers[varName](varName)
			if (value !== false) {
				result = result.replace(fullMatch, value)
			}
		}
	}

	// Process single curly brace syntax {varName}
	const singleMatches = [...result.matchAll(singleRegex)]
	for (const match of singleMatches) {
		const [fullMatch, varName] = match
		if (handlers[varName]) {
			const value = await handlers[varName](varName)
			if (value !== false) {
				result = result.replace(fullMatch, value)
			}
		}
	}

	return result
}

const slugify = (text: string): string => {
	return text.toLowerCase().replace(/\s/g, '-')
		// Remove punctuations other than hyphen and underscore
		.replace(/[`~!@#$%^&*()+=<>?,./:;"'|{}[\]\\\u2000-\u206F\u2E00-\u2E7F]/g, '')
		// Remove emojis
		.replace(emojiRegex, '')
		// Remove CJK punctuations
		.replace(/[\u3000。？！，、；：""【】（）〔〕［］﹃﹄""''﹁﹂—…－～《》〈〉「」]/g, '')
}

const formatFileSize = (bytes: number): string => {
	if (bytes === 0) return '0 Bytes'
	const k = 1024
	const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB']
	const i = Math.floor(Math.log(bytes) / Math.log(k))
	return `${Math.round((bytes / Math.pow(k, i)) * 100) / 100} ${sizes[i]}`
}

// Custom Mermaid and MLIR plugin for markdown-it
const customFencePlugin = (md: MarkdownIt): void => {
	const defaultRender = md.renderer.rules.fence || function(tokens: Token[], idx: number, options: Options, _env: any, renderer: Renderer): string {
		return renderer.renderToken(tokens, idx, options)
	}

	md.renderer.rules.fence = function(tokens: Token[], idx: number, options: Options, _env: any, renderer: Renderer): string {
		const token = tokens[idx]
		const info = token.info ? md.utils.unescapeAll(token.info).trim() : ''

		if (info === 'mermaid') {
			const content = token.content.trim()
			// Generate unique ID for each mermaid diagram
			const id = `mermaid-${Math.random().toString(36).substr(2, 9)}`
			return `<div class="mermaid" id="${id}">${md.utils.escapeHtml(content)}</div>\n`
		}

		// Handle MLIR code blocks
		if (info === 'mlir') {
			const content = token.content.trim()
			return mlirToHTML(content)
		}

		return defaultRender(tokens, idx, options, _env, renderer)
	}
}

// Text to HTML conversion function with syntax highlighting
const textToHTML = (text: string, filePath: string = ''): string => {
	// Escape HTML characters
	const escapeHtml = (str: string): string => str
		.replace(/&/g, '&amp;')
		.replace(/</g, '&lt;')
		.replace(/>/g, '&gt;')
		.replace(/"/g, '&quot;')
		.replace(/'/g, '&#039;')

	// Map file extensions to highlight.js language codes
	const extensionToLanguage: Record<string, string> = {
		'.js': 'javascript',
		'.mjs': 'javascript',
		'.jsx': 'javascript',
		'.ts': 'typescript',
		'.tsx': 'typescript',
		'.py': 'python',
		'.pyw': 'python',
		'.c': 'c',
		'.h': 'c',
		'.cpp': 'cpp',
		'.cc': 'cpp',
		'.cxx': 'cpp',
		'.hpp': 'cpp',
		'.hxx': 'cpp',
		'.java': 'java',
		'.cs': 'csharp',
		'.php': 'php',
		'.rb': 'ruby',
		'.go': 'go',
		'.rs': 'rust',
		'.kt': 'kotlin',
		'.swift': 'swift',
		'.m': 'objectivec',
		'.mm': 'objectivec',
		'.scala': 'scala',
		'.sh': 'bash',
		'.bash': 'bash',
		'.zsh': 'bash',
		'.fish': 'bash',
		'.ps1': 'powershell',
		'.r': 'r',
		'.R': 'r',
		'.sql': 'sql',
		'.html': 'html',
		'.htm': 'html',
		'.xml': 'xml',
		'.css': 'css',
		'.scss': 'scss',
		'.sass': 'scss',
		'.less': 'less',
		'.json': 'json',
		'.yaml': 'yaml',
		'.yml': 'yaml',
		'.toml': 'toml',
		'.ini': 'ini',
		'.cfg': 'ini',
		'.conf': 'apache',
		'.dockerfile': 'dockerfile',
		'.Dockerfile': 'dockerfile',
		'.makefile': 'makefile',
		'.Makefile': 'makefile',
		'.mk': 'makefile',
		'.lua': 'lua',
		'.pl': 'perl',
		'.pm': 'perl',
		'.dart': 'dart',
		'.elm': 'elm',
		'.ex': 'elixir',
		'.exs': 'elixir',
		'.erl': 'erlang',
		'.hrl': 'erlang',
		'.fs': 'fsharp',
		'.fsx': 'fsharp',
		'.fsi': 'fsharp',
		'.ml': 'ocaml',
		'.mli': 'ocaml',
		'.pas': 'pascal',
		'.pp': 'pascal',
		'.clj': 'clojure',
		'.cljs': 'clojure',
		'.cljc': 'clojure',
		'.lisp': 'lisp',
		'.lsp': 'lisp',
		'.l': 'lisp',
		'.cl': 'lisp',
		'.jl': 'julia',
		'.vim': 'vim',
		'.vimrc': 'vim',
		'.nix': 'nix',
		'.asm': 'x86asm',
		'.s': 'armasm',
		'.S': 'armasm',
		'.v': 'verilog',
		'.sv': 'verilog',
		'.vhd': 'vhdl',
		'.vhdl': 'vhdl',
		'.proto': 'protobuf',
		'.cmake': 'cmake',
		'.gradle': 'gradle',
		'.groovy': 'groovy',
		'.properties': 'properties',
		'.bat': 'dos',
		'.cmd': 'dos',
		'.awk': 'awk',
		'.sed': 'sed',
		'.j2': 'jinja',
		'.jinja': 'jinja',
		'.jinja2': 'jinja',
		'.fbs': 'flatbuffers',
		'.diff': 'diff',
		'.patch': 'diff'
	}

	// Determine the language based on file extension
	let language = 'plaintext'
	if (filePath) {
		const ext = path.extname(filePath).toLowerCase()
		language = extensionToLanguage[ext] || 'plaintext'

		// Special case for files without extension but with specific names
		const basename = path.basename(filePath)
		const basenameLower = basename.toLowerCase()

		// Match Dockerfile, dockerfile, and any Dockerfile.* variant
		if (basenameLower === 'dockerfile' || basenameLower.startsWith('dockerfile.')) language = 'dockerfile'
		// Match Makefile, makefile, GNUmakefile, and any Makefile.* variant
		if (basenameLower === 'makefile' || basenameLower.startsWith('makefile.') || basenameLower === 'gnumakefile') language = 'makefile'
		if (basenameLower === 'cmakelists.txt') language = 'cmake'
	}

	// Wrap text in pre/code tags with language class for syntax highlighting
	const escapedText = escapeHtml(text)
	return `<pre><code class="language-${language} hljs">${escapedText}</code></pre>`
}

// Log file to HTML conversion function with color highlighting
const logToHTML = (logText: string): string => {
	// Escape HTML characters
	const escapeHtml = (str: string): string => str
		.replace(/&/g, '&amp;')
		.replace(/</g, '&lt;')
		.replace(/>/g, '&gt;')
		.replace(/"/g, '&quot;')
		.replace(/'/g, '&#039;')

	// Process each line with enhanced formatting
	const lines = logText.split('\n')
	const highlightedLines = lines.map(line => {
		if (!line.trim()) return '' // Preserve empty lines

		const escapedLine = escapeHtml(line)

		// Parse structured log format: timestamp | LEVEL | message
		const structuredLogPattern = /^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d{3})?)\s*\|\s*(\w+)\s*\|\s*(.*)$/
		const structuredMatch = line.match(structuredLogPattern)

		if (structuredMatch) {
			const [, timestamp, level, message] = structuredMatch
			const escapedTimestamp = escapeHtml(timestamp)
			const escapedLevel = escapeHtml(level.toUpperCase())
			const escapedMessage = escapeHtml(message)

			// Style based on log level
			let levelColor: string
			let messageColor: string
			let bgStyle = ''

			switch(escapedLevel) {
				case 'ERROR':
				case 'FATAL':
				case 'CRITICAL':
					levelColor = '#ffffff'
					messageColor = '#ff6b6b'
					bgStyle = 'background: rgba(220, 53, 69, 0.3); padding: 2px 6px; border-radius: 3px;'
					break
				case 'WARNING':
				case 'WARN':
					levelColor = '#ffa94d'
					messageColor = '#ffa94d'
					break
				case 'INFO':
				case 'INFORMATION':
					levelColor = '#74c0fc'
					messageColor = '#a5d8ff'
					break
				case 'SUCCESS':
				case 'OK':
					levelColor = '#51cf66'
					messageColor = '#8ce99a'
					break
				case 'DEBUG':
				case 'TRACE':
					levelColor = '#868e96'
					messageColor = '#868e96'
					break
				default:
					levelColor = '#ced4da'
					messageColor = '#ced4da'
			}

			return `<span style="color: #495057;">${escapedTimestamp}</span> | <span style="color: ${levelColor}; font-weight: 600; ${bgStyle}">${escapedLevel.padEnd(8)}</span> | <span style="color: ${messageColor};">${escapedMessage}</span>`
		}

		// Alternative format: [timestamp] [LEVEL] message
		const bracketPattern = /^(\[[\d\-\:\.\s]+\])\s*\[(\w+)\]\s*(.*)$/
		const bracketMatch = line.match(bracketPattern)

		if (bracketMatch) {
			const [, timestamp, level, message] = bracketMatch
			const escapedTimestamp = escapeHtml(timestamp)
			const escapedLevel = escapeHtml(level.toUpperCase())
			const escapedMessage = escapeHtml(message)

			let levelColor: string
			let messageColor: string

			switch(escapedLevel) {
				case 'ERROR':
				case 'FATAL':
				case 'CRITICAL':
					levelColor = '#ff6b6b'
					messageColor = '#ff6b6b'
					break
				case 'WARNING':
				case 'WARN':
					levelColor = '#ffa94d'
					messageColor = '#ffa94d'
					break
				case 'INFO':
					levelColor = '#74c0fc'
					messageColor = '#a5d8ff'
					break
				case 'SUCCESS':
					levelColor = '#51cf66'
					messageColor = '#8ce99a'
					break
				case 'DEBUG':
				case 'TRACE':
					levelColor = '#868e96'
					messageColor = '#868e96'
					break
				default:
					levelColor = '#ced4da'
					messageColor = '#ced4da'
			}

			return `<span style="color: #495057;">${escapedTimestamp}</span> <span style="color: ${levelColor}; font-weight: 600;">[${escapedLevel}]</span> <span style="color: ${messageColor};">${escapedMessage}</span>`
		}

		// Fallback patterns for unstructured logs
		// ERROR/FATAL/CRITICAL - Red with background
		if (/\b(ERROR|FATAL|FAILURE|FAILED|CRITICAL)\b/i.test(line)) {
			return `<span style="color: #ff6b6b; background: rgba(220, 53, 69, 0.2); padding: 2px 4px; border-radius: 3px; display: inline-block; width: 100%;">${escapedLine}</span>`
		}
		// WARNING/WARN - Orange
		if (/\b(WARNING|WARN|CAUTION)\b/i.test(line)) {
			return `<span style="color: #ffa94d;">${escapedLine}</span>`
		}
		// INFO - Blue
		if (/\b(INFO|INFORMATION|NOTICE)\b/i.test(line)) {
			return `<span style="color: #74c0fc;">${escapedLine}</span>`
		}
		// SUCCESS - Green
		if (/\b(SUCCESS|SUCCESSFUL|OK|PASS|PASSED|COMPLETE|COMPLETED)\b/i.test(line)) {
			return `<span style="color: #51cf66;">${escapedLine}</span>`
		}
		// DEBUG/TRACE - Gray
		if (/\b(DEBUG|TRACE|VERBOSE)\b/i.test(line)) {
			return `<span style="color: #868e96;">${escapedLine}</span>`
		}
		// Stack traces - Purple/Pink
		if (/^\s+at\s+/.test(line)) {
			return `<span style="color: #e599f7;">${escapedLine}</span>`
		}
		// File paths and line numbers
		if (/\.(js|ts|py|java|cpp|c|go|rs|rb|php):\d+/.test(line)) {
			return `<span style="color: #fcc2d7;">${escapedLine}</span>`
		}

		// Default - subtle gray
		return `<span style="color: #adb5bd;">${escapedLine}</span>`
	})

	// Terminal-style container with enhanced styling
	// Use class-based styling for consistent sizing (like header elements in markdown.html)
	return `<div class="log-output-container">
		<div class="log-output-header">
			<span>📄 ${logText.split('\n').length} lines</span>
			<span class="log-output-title">Log Output</span>
		</div>
		<pre class="log-output-content"><code class="log-output-code">${highlightedLines.join('\n')}</code></pre>
	</div>`
}

// MLIR to HTML conversion function using highlight.js
const mlirToHTML = (mlirText: string): string => {
	try {
		// Use highlight.js to highlight MLIR code
		const result = hljs.highlight(mlirText, { language: 'mlir' })
		return `<pre><code class="language-mlir hljs">${result.value}</code></pre>`
	} catch (err) {
		// Fallback to escaped text if highlighting fails
		console.error('MLIR highlighting error:', err)
		const escaped = mlirText
			.replace(/&/g, '&amp;')
			.replace(/</g, '&lt;')
			.replace(/>/g, '&gt;')
			.replace(/"/g, '&quot;')
			.replace(/'/g, '&#039;')
		return `<pre><code class="language-mlir">${escaped}</code></pre>`
	}
}

const md = new MarkdownIt({
	linkify: false,
	html: true
})
	.use(mdItAnchor, {
		slugify,
		permalink: mdItAnchor.permalink.headerLink({
			safariReaderFix: true
		})
	})
	.use(mdItTaskLists)
	.use(mdItHLJS as unknown as PluginSimple)
	.use(mdItEmoji)
	.use(mdItMathJax)
	.use(customFencePlugin)
	.use(mdItTOC, {
		includeLevel: [1, 2, 3, 4, 5, 6],
		slugify
	})

// Markdown Extension Types
const fileTypes: FileTypes = {
	markdown: [
		'.markdown',
		'.mdown',
		'.mkdn',
		'.md',
		'.mkd',
		'.mdwn',
		'.mdtxt',
		'.mdtext',
		'.text'
	],

	html: [
		'.html',
		'.htm'
	],

	mlir: [
		'.mlir'
	],

	onnx: [
		'.onnx'
	],

	diff: [
		'.diff',
		'.patch'
	],

	text: [
		'.txt',
		'.log',
		'.conf',
		'.cfg',
		'.ini',
		'.yaml',
		'.yml',
		'.toml',
		'.json',
		'.xml',
		'.csv',
		'.tsv',
		'.sh',
		'.bash',
		'.zsh',
		'.fish',
		'.py',
		'.js',
		'.ts',
		'.jsx',
		'.tsx',
		'.java',
		'.c',
		'.cpp',
		'.h',
		'.hpp',
		'.cs',
		'.go',
		'.rs',
		'.swift',
		'.kt',
		'.rb',
		'.php',
		'.sql',
		'.r',
		'.m',
		'.mm',
		'.pl',
		'.pm',
		'.lua',
		'.vim',
		'.env',
		'.gitignore',
		'.dockerignore',
		'.editorconfig',
		'.eslintrc',
		'.prettierrc',
		'.j2',
		'.jinja',
		'.jinja2',
		'.fbs',
		'.makefile',
		'.mk',
		'Makefile',
		'makefile',
		'GNUmakefile',
		'.dockerfile',
		'Dockerfile',
		'dockerfile',
		'Dockerfile.dev',
		'Dockerfile.prod',
		'Dockerfile.test'
	],

	watch: [
		'.sass',
		'.less',
		'.js',
		'.css',
		'.json',
		'.gif',
		'.png',
		'.jpg',
		'.jpeg',
		'.mlir',
		'.onnx'
	],

	// Directories and patterns to exclude from file watching and processing
	// Ordered by frequency of occurrence for optimal performance
	exclusions: [
		// Package manager directories (most common)
		'node_modules/',
		'.pnpm/',
		'.yarn/',

		// Version control systems
		'.git/',
		'.svn/',
		'.hg/',

		// Build output directories
		'.dist/',
		'dist/',
		'build/',
		'.build/',
		'out/',
		'.out/',
		'target/',

		// Cache and temporary directories
		'.cache/',
		'.temp/',
		'.tmp/',
		'temp/',
		'tmp/',

		// IDE and editor directories
		'.vscode/',
		'.idea/',
		'.vs/',

		// OS-specific directories
		'.DS_Store',
		'Thumbs.db',
		'__pycache__/',
		'.pytest_cache/',

		// Log directories
		'logs/',
		'.log/',

		// Coverage and test output
		'coverage/',
		'.nyc_output/',
		'.coverage/',

		// Dependency lock files and configs that shouldn't be watched
		'*.lock',
		'*.log'
	]
}

// Enhanced exclusion pattern matching with error handling
// Note: Currently unused but kept for future use in file filtering
// @ts-expect-error - Function is currently unused but kept for future implementation
const isExcluded = (filePath: string, exclusions: string[] = fileTypes.exclusions): boolean => {
	if (!filePath || typeof filePath !== 'string') {
		return false
	}

	// Normalize path separators for cross-platform compatibility
	const normalizedPath = filePath.replace(/\\/g, '/')

	return exclusions.some(pattern => {
		try {
			// Handle glob patterns
			if (pattern.includes('*')) {
				// Simple glob matching for *.ext patterns
				if (pattern.startsWith('*.')) {
					const ext = pattern.slice(1) // Remove the *
					return normalizedPath.endsWith(ext)
				}
			}

			// Handle directory patterns (ending with /)
			if (pattern.endsWith('/')) {
				return normalizedPath.includes(`/${pattern}`) ||
				       normalizedPath.startsWith(pattern) ||
				       normalizedPath.includes(`/${pattern.slice(0, -1)}/`)
			}

			// Handle exact file matches
			return normalizedPath.endsWith(`/${pattern}`) ||
			       normalizedPath === pattern ||
			       normalizedPath.includes(`/${pattern}`)

		} catch (_error) {
			const err = _error as Error
			console.warn(`Error matching exclusion pattern "${pattern}" against "${filePath}":`, err.message)
			return false
		}
	})
}

// Validate exclusion patterns on startup
const validateExclusions = (exclusions: string[]): string[] => {
	const invalid: any[] = []
	const valid: string[] = []

	exclusions.forEach(pattern => {
		if (typeof pattern !== 'string' || pattern.length === 0) {
			invalid.push(pattern)
		} else if (pattern.includes('..') || pattern.includes('//')) {
			// Prevent path traversal patterns
			invalid.push(pattern)
		} else {
			valid.push(pattern)
		}
	})

	if (invalid.length > 0) {
		console.warn('Invalid exclusion patterns detected:', invalid)
	}

	return valid
}

// Initialize validated exclusions
fileTypes.exclusions = validateExclusions(fileTypes.exclusions)

fileTypes.watch = fileTypes.watch
	.concat(fileTypes.markdown)
	.concat(fileTypes.html)
	.concat(fileTypes.diff)
	.concat(fileTypes.text)

// In compiled dist/server.js, __dirname is 'dist/', so we need to go up to find lib/
const libPath = path.join(__dirname, '..', 'lib')

const faviconPath = path.join(libPath, 'icons', 'markserv.svg')
const faviconData = fs.readFileSync(faviconPath)

const log = (str: string | null, flags: Flags, err?: Error): void => {
	if (flags.silent) {
		return
	}

	if (str) {
		console.log(str)
	}

	if (err) {
		console.error(err)
	}
}

const msg = (type: string, msg: string, flags: Flags): void => {
	if (type === 'github') {
		return log(`${chalk.bgYellow.black('    GitHub  ')} ${msg}`, flags)
	}

	log(chalk.bgGreen.black('  Markserv  ') + chalk.white(` ${type}: `) + msg, flags)
}

const errormsg = (type: string, msg: string, flags: Flags, err?: Error): void =>
	log(chalk.bgRed.white('  Markserv  ') + chalk.red(` ${type}: `) + msg, flags, err)

const warnmsg = (type: string, msg: string, flags: Flags): void =>
	log(chalk.bgYellow.black('  Markserv  ') + chalk.yellow(` ${type}: `) + msg, flags)

const isType = (exts: string[], filePath: string): boolean => {
	const fileExt = path.parse(filePath).ext
	const basename = path.basename(filePath)
	const basenameLower = basename.toLowerCase()

	// Check both extension and basename (for files like "Makefile" without extensions)
	if (exts.includes(fileExt) || exts.includes(basename)) {
		return true
	}

	// Special pattern matching for Dockerfile.* and Makefile.* variants
	// Check if any entry in exts array is a pattern that the basename matches
	for (const ext of exts) {
		// Match any Dockerfile variant (Dockerfile, Dockerfile.dev, etc.)
		if (ext === 'Dockerfile' && basenameLower.startsWith('dockerfile')) {
			return true
		}
		// Match any Makefile variant (Makefile, Makefile.build, GNUmakefile, etc.)
		if ((ext === 'Makefile' || ext === 'GNUmakefile') &&
		    (basenameLower.startsWith('makefile') || basenameLower === 'gnumakefile')) {
			return true
		}
	}

	return false
}

// MarkdownToHTML: turns a Markdown file into HTML content
const markdownToHTML = (markdownText: string): Promise<string> => new Promise((resolve, reject) => {
	let result: string

	try {
		result = md.render(markdownText)
	} catch (error) {
		return reject(error)
	}

	resolve(result)
})

// GetFile: reads utf8 content from a file
const getFile = (path: string): Promise<string> => new Promise((resolve, reject) => {
	fs.readFile(path, 'utf8', (err, data) => {
		if (err) {
			return reject(err)
		}

		resolve(data)
	})
})

// Get Custom Less CSS to use in all Markdown files
const buildLessStyleSheet = (cssPath: string): Promise<string> =>
	new Promise(resolve =>
		getFile(cssPath).then(data =>
			less.render(data).then(data =>
				resolve(data.css)
			)
		)
	)

interface HandlebarData {
	[key: string]: any
}

const baseTemplate = (templateUrl: string, handlebarData: HandlebarData): Promise<string> => new Promise((resolve, reject) => {
	getFile(templateUrl).then(source => {
		const template = handlebars.compile(source)
		const output = template(handlebarData)
		resolve(output)
	}).catch(reject)
})

const dirToHtml = (filePath: string): DirectoryInfo => {
	const urls = fs.readdirSync(filePath)

	let fileCount = 0
	let folderCount = 0

	let prettyPath = `/${path.relative(process.cwd(), filePath)}`
	if (prettyPath[prettyPath.length] !== '/') {
		prettyPath += '/'
	}

	if (prettyPath.slice(prettyPath.length - 2, 2) === '//') {
		prettyPath = prettyPath.slice(0, prettyPath.length - 1)
	}

	// For display, we want to show the actual filename with special characters
	// but escape HTML to prevent injection
	const escapeHtml = (str: string): string => str
		.replace(/&/g, '&amp;')
		.replace(/</g, '&lt;')
		.replace(/>/g, '&gt;')
		.replace(/"/g, '&quot;')
		.replace(/'/g, '&#039;')

	// Collect items with their metadata
	const items: FileInfo[] = []
	urls.forEach(subPath => {
		if (subPath.charAt(0) === '.') {
			return
		}

		const fullPath = path.join(filePath, subPath)
		const stats = fs.statSync(fullPath)
		const isDir = stats.isDirectory()

		items.push({
			name: subPath,
			fullPath,
			isDirectory: isDir,
			mtime: stats.mtime,
			birthtime: stats.birthtime,
			size: stats.size
		})
	})

	// Sort: directories first, then by modification time (newest first)
	// If mtime is the same as birthtime (never modified), use birthtime
	items.sort((a, b) => {
		// First, sort by type (directories before files)
		if (a.isDirectory !== b.isDirectory) {
			return a.isDirectory ? -1 : 1
		}

		// Then sort by modification time (newest first)
		// Use birthtime if file was never modified (mtime === birthtime)
		const aTime = a.mtime.getTime() === a.birthtime.getTime() ? a.birthtime : a.mtime
		const bTime = b.mtime.getTime() === b.birthtime.getTime() ? b.birthtime : b.mtime

		return bTime.getTime() - aTime.getTime()
	})

	// Generate HTML from sorted items
	let list = '<ul>\n'
	items.forEach(item => {
		const encodedPath = encodeURIComponent(item.name)

		if (item.isDirectory) {
			const href = `${encodedPath}/`
			const displayName = `🗂️ ${escapeHtml(item.name)}/`
			list += `\t<li class="icon folder isfolder" title="Directory"><a href="${href}">${displayName}</a></li>\n`
			folderCount++
		} else {
			const href = encodedPath
			const displayName = `📝 ${escapeHtml(item.name)}`
			const fileSize = formatFileSize(item.size)
			list += `\t<li class="isfile" title="File"><a href="${href}">${displayName}</a><span class="file-size">${fileSize}</span></li>\n`
			fileCount++
		}
	})

	list += '</ul>\n'

	// Return an object with both HTML and counts
	return {
		html: list,
		fileCount,
		folderCount
	}
}

// Remove URL params from file being fetched
const getPathFromUrl = (url: string): string => {
	return url.split(/[?#]/)[0]
}

// Removed markservPageObject - no longer needed since we directly
// return the relative path in the markserv handler

// Helper function to encode URL paths while preserving path structure
// Encodes each path segment separately to handle Chinese/Unicode characters
// while keeping forward slashes as path separators
const encodeUrlPath = (urlPath: string): string => {
	// Split by /, encode each segment, then rejoin with /
	return urlPath.split('/').map(segment => encodeURIComponent(segment)).join('/')
}

const secureUrl = (url: string): string => {
	// Use encodeURIComponent to properly encode all special characters including Chinese
	// This ensures Chinese characters and other Unicode characters are properly handled
	const encodedUrl = encodeURIComponent(url)
	return encodedUrl
}

// Create breadcrumb trail tracks
const createBreadcrumbs = (path: string): Breadcrumb[] => {
	const crumbs: Breadcrumb[] = [{
		href: '/',
		text: './'
	}]

	const dirParts = path.replace(/(^\/+|\/+$)/g, '').split('/')
	const urlParts = dirParts.map(secureUrl)

	if (path.length === 0) {
		return crumbs
	}

	let collectPath = '/'

	dirParts.forEach((dirName, i) => {
		const fullLink = `${collectPath + urlParts[i]}/`

		const crumb: Breadcrumb = {
			href: fullLink,
			text: `${dirName}/`
		}

		crumbs.push(crumb)
		collectPath = fullLink
	})

	return crumbs
}

// Analytics helper functions
const logPageVisit = (req: Request, _filePath: string, fileType?: string): void => {
	const visitPath = getPathFromUrl(req.originalUrl)

	// Exclude only markserv internal library files from analytics
	// Track user content including Model Explorer visualizations and API downloads
	if (
		visitPath.startsWith('/lib/') ||           // Library files (CSS, icons, etc.)
		visitPath.startsWith('/{markserv}') ||     // Markserv URL placeholders
		visitPath === '/tracking'                   // Analytics page itself
	) {
		return
	}

	const refererHeader = req.headers.referer || req.headers.referrer
	const referer = Array.isArray(refererHeader) ? refererHeader[0] : (refererHeader || 'Direct')

	const visit: PageVisit = {
		timestamp: new Date(),
		path: visitPath,
		method: req.method,
		userAgent: req.headers['user-agent'] || 'Unknown',
		referer,
		ip: req.ip || req.socket.remoteAddress || 'Unknown',
		fileType
	}

	// Add to analytics data with size limit
	analyticsData.push(visit)
	if (analyticsData.length > MAX_ANALYTICS_RECORDS) {
		analyticsData.shift() // Remove oldest record
	}
}

const generateAnalyticsStats = (): AnalyticsStats => {
	const pathCounts = new Map<string, number>()
	const fileTypeCounts = new Map<string, number>()
	const hourCounts = new Map<number, number>()
	const visitorCounts = new Map<string, { visits: number; lastSeen: Date }>()

	analyticsData.forEach(visit => {
		// Count paths
		pathCounts.set(visit.path, (pathCounts.get(visit.path) || 0) + 1)

		// Count file types
		if (visit.fileType) {
			fileTypeCounts.set(visit.fileType, (fileTypeCounts.get(visit.fileType) || 0) + 1)
		}

		// Count by hour
		const hour = visit.timestamp.getHours()
		hourCounts.set(hour, (hourCounts.get(hour) || 0) + 1)

		// Count unique visitors (using IP + User Agent as identifier)
		const visitorId = `${visit.ip}|${visit.userAgent}`
		const existing = visitorCounts.get(visitorId)
		if (existing) {
			existing.visits++
			if (visit.timestamp > existing.lastSeen) {
				existing.lastSeen = visit.timestamp
			}
		} else {
			visitorCounts.set(visitorId, { visits: 1, lastSeen: visit.timestamp })
		}
	})

	// Get top paths
	const topPaths = Array.from(pathCounts.entries())
		.map(([path, count]) => ({ path, count }))
		.sort((a, b) => b.count - a.count)
		.slice(0, 20)

	// Get visits by file type
	const visitsByFileType = Array.from(fileTypeCounts.entries())
		.map(([type, count]) => ({ type, count }))
		.sort((a, b) => b.count - a.count)

	// Get visits by hour (all 24 hours)
	const visitsByHour = Array.from({ length: 24 }, (_, hour) => ({
		hour,
		count: hourCounts.get(hour) || 0
	}))

	// Get top visitors
	const topVisitors = Array.from(visitorCounts.entries())
		.map(([identifier, data]) => ({
			identifier: identifier.split('|')[0], // Show only IP for privacy
			visits: data.visits,
			lastSeen: data.lastSeen
		}))
		.sort((a, b) => b.visits - a.visits)
		.slice(0, 10)

	// Get recent visits (last 100)
	const recentVisits = analyticsData.slice(-100).reverse()

	return {
		totalVisits: analyticsData.length,
		uniquePaths: pathCounts.size,
		uniqueVisitors: visitorCounts.size,
		topPaths,
		recentVisits,
		visitsByFileType,
		visitsByHour,
		topVisitors
	}
}

// Http_request_handler: handles all the browser requests
const createRequestHandler = (flags: Flags) => {
	let {dir} = flags
	const isDir = fs.statSync(dir).isDirectory()
	if (!isDir) {
		dir = path.parse(flags.dir).dir
	}

	flags.$openLocation = path.relative(dir, flags.dir)

	const implantOpts: ImplantOptions = {
		maxDepth: 10
	}

	const markservUrlLead = '%7Bmarkserv%7D'

	return (req: Request, res: Response): void => {
		// Properly decode the URL - decodeURIComponent handles special characters better than unescape
		const decodedUrl = getPathFromUrl(decodeURIComponent(req.originalUrl))

		// Handle Analytics Tracking route (/tracking)
		if (decodedUrl === '/tracking') {
			const stats = generateAnalyticsStats()
			const templateUrl = path.join(libPath, 'templates/tracking.html')

			// Get current visitor's IP
			const currentVisitorIP = req.ip || req.socket.remoteAddress || 'Unknown'

			// Get timezone information
			const now = new Date()
			const timezoneOffset = -now.getTimezoneOffset() / 60
			const timezoneSign = timezoneOffset >= 0 ? '+' : '-'
			const timezoneString = `UTC${timezoneSign}${Math.abs(timezoneOffset)}`

			// Format timestamps without timezone for display
			const formatTimestampWithoutTZ = (date: Date): string => {
				return date.toLocaleString('en-US', {
					year: 'numeric',
					month: 'short',
					day: 'numeric',
					hour: '2-digit',
					minute: '2-digit',
					second: '2-digit'
				})
			}

			// Format recent visits with formatted timestamps and decoded paths for display
			const formattedRecentVisits = stats.recentVisits.map(visit => ({
				...visit,
				timestamp: formatTimestampWithoutTZ(visit.timestamp),
				displayPath: decodeURIComponent(visit.path) // Decode for human-readable display
			}))

			// Format top visitors with formatted lastSeen
			const formattedTopVisitors = stats.topVisitors.map(visitor => ({
				...visitor,
				lastSeen: formatTimestampWithoutTZ(visitor.lastSeen)
			}))

			// Format top paths with decoded display names
			const formattedTopPaths = stats.topPaths.map(pathItem => ({
				...pathItem,
				displayPath: decodeURIComponent(pathItem.path) // Decode for human-readable display
			}))

			const handlebarData: HandlebarData = {
				totalVisits: stats.totalVisits,
				uniquePaths: stats.uniquePaths,
				uniqueVisitors: stats.uniqueVisitors,
				topPaths: formattedTopPaths,
				recentVisits: formattedRecentVisits,
				visitsByFileType: stats.visitsByFileType,
				visitsByHour: stats.visitsByHour,
				topVisitors: formattedTopVisitors,
				currentVisitorIP,
				pid: process.pid || 'N/A',
				timezone: timezoneString
			}

			baseTemplate(templateUrl, handlebarData).then(html => {
				res.writeHead(200, {
					'content-type': 'text/html'
				})
				res.end(html)
			}).catch((error: Error) => {
				console.error('Error rendering tracking page:', error)
				res.status(500).send('Error generating analytics report')
			})
			return
		}

		// Handle JSON Editor route (/json)
		if (decodedUrl.startsWith('/json/') || decodedUrl === '/json') {
			const jsonFilePath = decodedUrl === '/json' ? '' : decodedUrl.substring(6) // Remove '/json/'

			if (!jsonFilePath || jsonFilePath === '') {
				res.status(400).send('Please specify a JSON file path, e.g., /json/data.json')
				return
			}

			// Resolve the actual file path
			const actualFilePath = path.normalize(path.join(dir, jsonFilePath))

			// Check if file exists and has .json extension
			if (!actualFilePath.endsWith('.json')) {
				res.status(400).send('Only JSON (.json) files are supported')
				return
			}

			fs.stat(actualFilePath, (err, stats) => {
				if (err || !stats.isFile()) {
					res.status(404).send(`JSON file not found: ${jsonFilePath}`)
					return
				}

				// Log the JSON viewer visit
				logPageVisit(req, actualFilePath, 'json-editor')

				// Read the JSON file content
				fs.readFile(actualFilePath, 'utf8', (readErr, jsonContent) => {
					if (readErr) {
						console.error('Error reading JSON file:', readErr)
						res.status(500).send('Failed to read JSON file')
						return
					}

					// Parse and validate JSON
					let jsonData
					try {
						jsonData = JSON.parse(jsonContent)
					} catch (parseError) {
						console.error('Error parsing JSON:', parseError)
						res.status(500).send(`Invalid JSON: ${(parseError as Error).message}`)
						return
					}

					// Load and render the JSON editor template
					const templatePath = path.join(libPath, 'templates/json-editor.html')
					fs.readFile(templatePath, 'utf8', (templateErr, template) => {
						if (templateErr) {
							console.error('Error loading JSON editor template:', templateErr)
							res.status(500).send('Failed to load JSON editor template')
							return
						}

						const filename = path.basename(jsonFilePath)
						// Calculate parent directory, ensuring it's a proper path without /json/ prefix
						let backDir = path.dirname(jsonFilePath)
						if (backDir === '.') {
							backDir = '/'
						} else if (!backDir.startsWith('/')) {
							backDir = '/' + backDir
						}
						const lastModified = stats.mtime.toLocaleString('en-US', {
							year: 'numeric',
							month: 'short',
							day: 'numeric',
							hour: '2-digit',
							minute: '2-digit'
						})

						// Convert JSON data to string for embedding
						const jsonDataString = JSON.stringify(jsonData)

						// Replace template variables
						const html = template
							.replace(/\{\{title\}\}/g, filename)
							.replace(/\{\{fileName\}\}/g, filename)
							.replace(/\{\{lastModified\}\}/g, lastModified)
							.replace(/\{\{parentDir\}\}/g, backDir)
							.replace(/\{\{filePath\}\}/g, jsonFilePath)
							.replace(/\{\{\{jsonData\}\}\}/g, jsonDataString)

						res.writeHead(200, { 'Content-Type': 'text/html' })
						res.end(html)
					})
				})
			})
			return
		}

		// Handle Model Explorer route (/model-explorer)
		if (decodedUrl.startsWith('/model-explorer/') || decodedUrl === '/model-explorer') {
			const modelFilePath = decodedUrl === '/model-explorer' ? '' : decodedUrl.substring(16) // Remove '/model-explorer/'

			if (!modelFilePath || modelFilePath === '') {
				res.status(400).send('Please specify a model file path, e.g., /model-explorer/model.mlir or /model-explorer/model.onnx')
				return
			}

			// Resolve the actual file path
			const actualFilePath = path.normalize(path.join(dir, modelFilePath))

			// Check if file exists and has .mlir or .onnx extension
			const isMLIR = actualFilePath.endsWith('.mlir')
			const isONNX = actualFilePath.endsWith('.onnx')

			if (!isMLIR && !isONNX) {
				res.status(400).send('Only MLIR (.mlir) and ONNX (.onnx) files are supported')
				return
			}

			fs.stat(actualFilePath, (err, stats) => {
				if (err || !stats.isFile()) {
					res.status(404).send(`Model file not found: ${modelFilePath}`)
					return
				}

				// Log the Model Explorer visit
				logPageVisit(req, actualFilePath, 'model-explorer')

				// Handle MLIR files (text-based)
				if (isMLIR) {
					// Read the MLIR file content as UTF-8 text
					fs.readFile(actualFilePath, 'utf8', (readErr, mlirContent) => {
						if (readErr) {
							console.error('Error reading MLIR file:', readErr)
							res.status(500).send('Failed to read MLIR file')
							return
						}

						// Convert MLIR to graph format
						let graphData
						try {
							graphData = convertMLIRToGraph(mlirContent, path.basename(modelFilePath))
						} catch (conversionErr) {
							console.error('Error converting MLIR to graph:', conversionErr)
							res.status(500).send('Failed to convert MLIR to graph format')
							return
						}

						// Load and render the Model Explorer template
						const templatePath = path.join(libPath, 'templates/model-explorer.html')
						fs.readFile(templatePath, 'utf8', (templateErr, template) => {
							if (templateErr) {
								console.error('Error loading Model Explorer template:', templateErr)
								res.status(500).send('Failed to load Model Explorer template')
								return
							}

							const filename = path.basename(modelFilePath)
							const backUrl = path.dirname(modelFilePath) || '/'

							// Convert graph data to JSON - no need to escape since it's in a JSON script tag
							const graphDataJson = JSON.stringify(graphData, null, 2)

							// Replace template variables
							const html = template
								.replace(/\{\{filename\}\}/g, filename)
								.replace(/\{\{graphData\}\}/g, graphDataJson)
								.replace(/\{\{backUrl\}\}/g, backUrl)

							res.setHeader('Content-Type', 'text/html; charset=utf-8')
							res.status(200).send(html)
						})
					})
				} else if (isONNX) {
					// Read the ONNX file content as binary buffer
					fs.readFile(actualFilePath, (readErr, onnxBuffer) => {
						if (readErr) {
							console.error('Error reading ONNX file:', readErr)
							res.status(500).send('Failed to read ONNX file')
							return
						}

						// Convert ONNX to graph format (async)
						convertONNXToGraph(onnxBuffer, path.basename(modelFilePath))
							.then(graphData => {
								// Load and render the Model Explorer template
								const templatePath = path.join(libPath, 'templates/model-explorer.html')
								fs.readFile(templatePath, 'utf8', (templateErr, template) => {
									if (templateErr) {
										console.error('Error loading Model Explorer template:', templateErr)
										res.status(500).send('Failed to load Model Explorer template')
										return
									}

									const filename = path.basename(modelFilePath)
									const backUrl = path.dirname(modelFilePath) || '/'

									// Convert graph data to JSON - no need to escape since it's in a JSON script tag
									const graphDataJson = JSON.stringify(graphData, null, 2)

									// Replace template variables
									const html = template
										.replace(/\{\{filename\}\}/g, filename)
										.replace(/\{\{graphData\}\}/g, graphDataJson)
										.replace(/\{\{backUrl\}\}/g, backUrl)

									res.setHeader('Content-Type', 'text/html; charset=utf-8')
									res.status(200).send(html)
								})
							})
							.catch(conversionErr => {
								console.error('Error converting ONNX to graph:', conversionErr)
								res.status(500).send('Failed to convert ONNX to graph format: ' + (conversionErr as Error).message)
							})
					})
				}
			})
			return
		}

		// Handle MLIR conversion API endpoint
		if (req.url === '/api/convert-mlir' && req.method === 'POST') {
			let body = ''

			req.on('data', chunk => {
				body += chunk.toString()
			})

			req.on('end', () => {
				try {
					const { mlirContent, filename } = JSON.parse(body)

					// Parse MLIR content and convert to Model Explorer graph format
					const graphData = convertMLIRToGraph(mlirContent, filename)

					res.setHeader('Content-Type', 'application/json')
					res.status(200).send(JSON.stringify(graphData))
				} catch (error) {
					console.error('MLIR conversion error:', error)
					res.status(500).send(JSON.stringify({
						error: 'Failed to convert MLIR to graph format',
						message: (error as Error).message
					}))
				}
			})
			return
		}

		// Handle dev3000 specific routes (Next.js paths)
		// dev3000 tries to communicate using Next.js-specific paths like /_next/mcp
		// Silently return 404 for these paths since markserv is not a Next.js app
		if (decodedUrl.startsWith('/_next/')) {
			res.status(404).end()
			return
		}

		// Special handling for lib resources (CSS, icons, etc.)
		// These should always be served from the actual lib directory, not relative to serving directory
		let filePath: string
		if (decodedUrl.startsWith('/lib/')) {
			// Remove the leading /lib/ and serve from the actual lib directory
			const libResource = decodedUrl.substring(5)  // Remove '/lib/'
			filePath = path.normalize(path.join(libPath, libResource))
		} else {
			// Don't use unescape as it's deprecated and doesn't handle special characters well
			filePath = path.normalize(path.join(dir, decodedUrl))
		}
		const baseDir = path.parse(filePath).dir
		implantOpts.baseDir = baseDir

		// Since we're using absolute URLs starting with /lib/, we always use '/lib'
		// This works from any directory depth
		const relativePath = '/lib'

		// Create request-specific handlers with the correct relative path
		const implantHandlers: ImplantHandlers = {
			markserv: (_prop: string): Promise<string | false> => new Promise(resolve => {
				// Return the relative path from the current location to the lib directory
				// This ensures CSS paths work correctly in nested directories
				resolve(relativePath)
			}),

			file: (url: string, opts?: ImplantOptions): Promise<string | false> => new Promise(resolve => {
				const absUrl = path.join(opts?.baseDir || '', url)
				getFile(absUrl)
					.then(data => {
						msg('implant', style.link(absUrl), flags)
						resolve(data)
					})
					.catch((_error: Error) => {
						warnmsg('implant 404', style.link(absUrl), flags)
						resolve(false)
					})
			}),

			less: (url: string, opts?: ImplantOptions): Promise<string | false> => new Promise(resolve => {
				const absUrl = path.join(opts?.baseDir || '', url)
				buildLessStyleSheet(absUrl)
					.then(data => {
						msg('implant', style.link(absUrl), flags)
						resolve(data)
					})
					.catch((_error: Error) => {
						warnmsg('implant 404', style.link(absUrl), flags)
						resolve(false)
					})
			}),

			markdown: (url: string, opts?: ImplantOptions): Promise<string | false> => new Promise(resolve => {
				const absUrl = path.join(opts?.baseDir || '', url)
				getFile(absUrl).then(markdownToHTML)
					.then(data => {
						msg('implant', style.link(absUrl), flags)
						resolve(data)
					})
					.catch((_error: Error) => {
						warnmsg('implant 404', style.link(absUrl), flags)
						resolve(false)
					})
			}),

			html: (url: string, opts?: ImplantOptions): Promise<string | false> => new Promise(resolve => {
				const absUrl = path.join(opts?.baseDir || '', url)
				getFile(absUrl)
					.then(data => {
						msg('implant', style.link(absUrl), flags)
						resolve(data)
					})
					.catch((_error: Error) => {
						warnmsg('implant 404', style.link(absUrl), flags)
						resolve(false)
					})
			})
		}

		const errorPage = (code: number, filePath: string, err: Error): Promise<void> => {
			errormsg(String(code), filePath, flags, err)

			const templateUrl = path.join(libPath, 'templates/error.html')
			const fileName = path.parse(filePath).base
			// Use decodeURIComponent instead of deprecated unescape
			// Ensure the fallback referer (parent directory) always has a trailing slash
			const parentPath = path.parse(decodedUrl).dir
			const referer = req.headers.referer ?
				decodeURIComponent(req.headers.referer) :
				(parentPath === '' || parentPath === '/' ? '/' : parentPath + '/')
			const errorMsg = md.utils.escapeHtml(err.message)
			const errorStack = md.utils.escapeHtml(String(err.stack))

			const handlebarData: HandlebarData = {
				pid: process.pid || 'N/A',
				code,
				fileName,
				filePath,
				errorMsg,
				errorStack,
				referer
			}

			return baseTemplate(templateUrl, handlebarData).then(final => {
				res.writeHead(200, {
					'content-type': 'text/html; charset=utf-8'
				})
				res.end(final)
			})
		}

		if (flags.verbose) {
			msg('request', filePath, flags)
		}

		const isMarkservUrl = req.url.includes(markservUrlLead)
		if (isMarkservUrl) {
			const markservFilePath = req.url.split(markservUrlLead)[1]
			const markservRelFilePath = path.join(__dirname, markservFilePath)
			if (flags.verbose) {
				msg('{markserv url}', style.link(markservRelFilePath), flags)
			}

			// Send static file
			const stream = fs.createReadStream(markservRelFilePath)
			stream.on('error', () => {
				res.status(404).end()
			})
			stream.pipe(res)
			return
		}

		// Serve Model Explorer static files
		if (decodedUrl.startsWith('/lib/model-explorer/')) {
			const modelExplorerFile = decodedUrl.substring(20) // Remove '/lib/model-explorer/' prefix
			const modelExplorerPath = path.join(libPath, 'model-explorer', modelExplorerFile)

			if (flags.verbose) {
				msg('model-explorer', style.link(modelExplorerPath), flags)
			}

			// Check if file exists
			if (!fs.existsSync(modelExplorerPath)) {
				console.error('Model Explorer file not found:', modelExplorerPath)
				res.status(404).send('Model Explorer file not found: ' + modelExplorerFile)
				return
			}

			// Check if it's a directory (shouldn't happen, but handle it)
			const stat = fs.statSync(modelExplorerPath)
			if (stat.isDirectory()) {
				res.status(403).send('Cannot serve directory')
				return
			}

			// Set appropriate content type
			const mimeType = mime.lookup(modelExplorerPath)
			res.setHeader('Content-Type', mimeType || 'application/octet-stream')
			res.setHeader('Cache-Control', 'public, max-age=3600') // Cache for 1 hour

			// Send the file
			const stream = fs.createReadStream(modelExplorerPath)
			stream.on('error', (err) => {
				console.error('Error streaming model-explorer file:', err)
				res.status(404).end()
			})
			stream.pipe(res)
			return
		}

		// API route for direct file downloads (e.g., /api/demo.js downloads demo.js)
		if (decodedUrl.startsWith('/api/')) {
			const apiPath = decodedUrl.substring(5) // Remove '/api/' prefix
			const apiFilePath = path.normalize(path.join(dir, getPathFromUrl(decodeURIComponent(apiPath))))

			if (flags.verbose) {
				msg('api', style.link(apiFilePath), flags)
			}

			try {
				const stat = fs.statSync(apiFilePath)

				if (stat.isDirectory()) {
					res.status(400).send('API route does not support directories')
					return
				}

				// Log the API download visit
				logPageVisit(req, apiFilePath, 'api')

				const mimeType = mime.lookup(apiFilePath)
				const fileName = path.basename(apiFilePath)

				// Set headers for direct download
				res.setHeader('Content-Type', mimeType || 'application/octet-stream')
				res.setHeader('Content-Length', stat.size)
				res.setHeader('Content-Disposition', `attachment; filename="${fileName}"`)

				// Stream the file
				const stream = fs.createReadStream(apiFilePath)
				stream.on('error', (err) => {
					errormsg('api error', apiFilePath, flags, err as Error)
					res.status(500).end()
				})
				stream.pipe(res)
				return
			} catch (error) {
				errormsg('api 404', apiFilePath, flags, error as Error)
				res.status(404).send('File not found')
				return
			}
		}

		// Make prettyPath relative to the base directory for URL generation
		const prettyPath = `/${path.relative(flags.dir || process.cwd(), filePath)}`

		let stat: fs.Stats
		let isDir: boolean
		let isMarkdown = false
		let isHtml = false
		let isMLIR = false
		let isONNX = false
		let isDiff = false
		let isText = false

		try {
			stat = fs.statSync(filePath)
			isDir = stat.isDirectory()
			if (!isDir) {
				isMarkdown = isType(fileTypes.markdown, filePath)
				isHtml = isType(fileTypes.html, filePath)
				isMLIR = isType(fileTypes.mlir, filePath)
				isONNX = isType(fileTypes.onnx, filePath)
				isDiff = isType(fileTypes.diff, filePath)
				isText = isType(fileTypes.text, filePath)
			}
		} catch (error) {
			const fileName = path.parse(filePath).base
			if (fileName === 'favicon.ico') {
				res.writeHead(200, {'Content-Type': 'image/x-icon'})
				res.write(faviconData)
				res.end()
				return
			}

			errormsg('404', filePath, flags, error as Error)
			errorPage(404, filePath, error as Error)
			return
		}

		// Check if this is a download request for any file type
		if (req.url.includes('download=true') && !isDir) {
			const fileName = path.basename(filePath)
			const mimeType = mime.lookup(filePath)

			res.setHeader('Content-Disposition', `attachment; filename="${fileName}"`)
			if (mimeType) {
				res.setHeader('Content-Type', mimeType)
			}

			const stream = fs.createReadStream(filePath)
			stream.on('error', () => {
				res.status(404).end()
			})
			stream.pipe(res)
			return
		}

		// Markdown: Browser is requesting a Markdown file
		if (isMarkdown) {
			msg('markdown', style.link(prettyPath), flags)
			logPageVisit(req, filePath, 'markdown')
			getFile(filePath).then(markdownToHTML).then((html: string) => {
				return processTemplate(html, implantHandlers).then(output => {
					const templateUrl = path.join(libPath, 'templates/markdown.html')

					const stats = fs.statSync(filePath)
					const lastModified = stats.mtime.toLocaleString('en-US', {
						year: 'numeric',
						month: 'short',
						day: 'numeric',
						hour: '2-digit',
						minute: '2-digit'
					})

					const handlebarData: HandlebarData = {
						title: path.parse(filePath).base,
						content: output,
						pid: process.pid || 'N/A',
						filePath: prettyPath,
						fileName: path.basename(filePath),
						lastModified,
						parentDir: path.dirname(prettyPath) === '/' ? '/' : path.dirname(prettyPath) + '/'
					}

					return baseTemplate(templateUrl, handlebarData).then(final => {
						return processTemplate(final, implantHandlers)
							.then(output => {
								res.writeHead(200, {
									'content-type': 'text/html'
								})
								res.end(output)
							})
					})
				})
			}).catch((error: Error) => {
				console.error(error)
			})
		} else if (isHtml) {
			msg('html', style.link(prettyPath), flags)
			logPageVisit(req, filePath, 'html')
			getFile(filePath).then(html => {
				return processTemplate(html, implantHandlers).then(output => {
					res.writeHead(200, {
						'content-type': 'text/html'
					})
					res.end(output)
				})
			}).catch((error: Error) => {
				console.error(error)
			})
		} else if (isDiff) {
			// Diff: Browser is requesting a diff or patch file
			msg('diff', style.link(prettyPath), flags)
			logPageVisit(req, filePath, 'diff')
			getFile(filePath).then(diffContent => {
				const htmlContent = textToHTML(diffContent, filePath)
				const templateUrl = path.join(libPath, 'templates/markdown.html')

				const stats = fs.statSync(filePath)
				const lastModified = stats.mtime.toLocaleString('en-US', {
					year: 'numeric',
					month: 'short',
					day: 'numeric',
					hour: '2-digit',
					minute: '2-digit'
				})

				const handlebarData: HandlebarData = {
					title: path.parse(filePath).base,
					content: htmlContent,
					pid: process.pid || 'N/A',
					filePath: prettyPath,
					fileName: path.basename(filePath),
					lastModified,
					parentDir: path.dirname(prettyPath) || '/'
				}

				return baseTemplate(templateUrl, handlebarData).then(final => {
					return processTemplate(final, implantHandlers)
						.then(output => {
							res.writeHead(200, {
								'content-type': 'text/html'
							})
							res.end(output)
						})
				})
			}).catch((error: Error) => {
				errorPage(500, filePath, error)
			})
		} else if (isMLIR) {
			// MLIR: Browser is requesting an MLIR file
			msg('mlir', style.link(prettyPath), flags)
			logPageVisit(req, filePath, 'mlir')
			getFile(filePath).then(mlirContent => {
				const htmlContent = mlirToHTML(mlirContent)
				const templateUrl = path.join(libPath, 'templates/markdown.html')

				const stats = fs.statSync(filePath)
				const lastModified = stats.mtime.toLocaleString('en-US', {
					year: 'numeric',
					month: 'short',
					day: 'numeric',
					hour: '2-digit',
					minute: '2-digit'
				})

				const handlebarData: HandlebarData = {
					title: path.parse(filePath).base,
					content: htmlContent,
					pid: process.pid || 'N/A',
					filePath: prettyPath,
					fileName: path.basename(filePath),
					lastModified,
					parentDir: path.dirname(prettyPath) || '/'
				}

				return baseTemplate(templateUrl, handlebarData).then(final => {
					return processTemplate(final, implantHandlers)
						.then(output => {
							res.writeHead(200, {
								'content-type': 'text/html'
							})
							res.end(output)
						})
				})
			}).catch((error: Error) => {
				errorPage(500, filePath, error)
			})
		} else if (isDir) {
			// Redirect to URL with trailing slash if accessing directory without one
			// This ensures relative links work correctly
			if (!decodedUrl.endsWith('/')) {
				// Properly encode URL path for Location header to handle Chinese/Unicode characters
				const redirectUrl = encodeUrlPath(decodedUrl) + '/'
				res.writeHead(301, { 'Location': redirectUrl })
				res.end()
				return
			}

			// Handle file upload for directories
			if (req.method === 'POST' && req.url.includes('upload=true')) {
				// First, parse the multipart form to get all fields
				const upload = multer().any()

				upload(req, res, function (_err) {
					if (_err) {
						console.error('Upload error:', _err)
						res.status(500).send('Upload failed: ' + _err.message)
						return
					}

					// Find the file and relativePath from the parsed fields
					const fileField = (req as any).files && (req as any).files.find((f: any) => f.fieldname === 'file')
					const relativePath = (req as any).body && (req as any).body.relativePath ? (req as any).body.relativePath : ''

					if (!fileField) {
						res.status(400).send('No file uploaded')
						return
					}

					// Get the original filename - it may contain special characters
					const originalFilename = fileField.originalname

					// For safety, we should validate that the path doesn't try to escape the upload directory
					// But we should preserve special characters that are valid in filenames
					const isPathSafe = (pathStr: string): boolean => {
						// Check for path traversal attempts
						const normalized = path.normalize(pathStr)
						return !normalized.includes('..') && !path.isAbsolute(normalized)
					}

					// Validate paths
					if (relativePath && !isPathSafe(relativePath)) {
						console.error('Invalid relative path:', relativePath)
						res.status(400).send('Invalid path')
						return
					}

					if (!isPathSafe(originalFilename)) {
						console.error('Invalid filename:', originalFilename)
						res.status(400).send('Invalid filename')
						return
					}

					// Determine target directory
					let targetDir = filePath
					let targetPath: string

					// Check write permissions on the base directory first
					try {
						fs.accessSync(filePath, fs.constants.W_OK)
					} catch (accessErr) {
						console.error('No write permission for directory:', filePath)
						console.error('Error:', accessErr)
						res.status(403).send('Permission denied: Cannot write to this directory. Please check directory permissions.')
						return
					}

					try {
						if (relativePath) {
							// Create the subdirectory structure
							// Note: path.join handles special characters properly
							targetDir = path.join(filePath, relativePath)

							// Create all parent directories if they don't exist
							fs.mkdirSync(targetDir, { recursive: true })
						}

						// Save the file to the target directory
						// path.join properly handles special characters in filenames
						targetPath = path.join(targetDir, originalFilename)

						// Write the file
						fs.writeFile(targetPath, fileField.buffer, function(writeErr) {
							if (writeErr) {
								console.error('File write error:', writeErr)
								console.error('Target path was:', targetPath)

								// Provide more specific error message for permission errors
								if ((writeErr as NodeJS.ErrnoException).code === 'EACCES') {
									res.status(403).send('Permission denied: Cannot write file. Please check directory permissions.')
								} else {
									res.status(500).send('Failed to save file: ' + writeErr.message)
								}
								return
							}

							const displayPath = relativePath ?
								`${relativePath}/${originalFilename}` :
								originalFilename

							msg('upload', `${displayPath} -> ${style.link(prettyPath)}`, flags)
							res.status(200).send('File uploaded successfully')
						})
					} catch (mkdirError) {
						console.error('Directory creation error:', mkdirError)
						console.error('Target directory was:', targetDir)

						// Provide more specific error message for permission errors
						if ((mkdirError as NodeJS.ErrnoException).code === 'EACCES') {
							res.status(403).send('Permission denied: Cannot create directory. Please check parent directory permissions.')
						} else {
							res.status(500).send('Failed to create directory: ' + (mkdirError as Error).message)
						}
						return
					}
				})
				return
			}

			try {
				// Index: Browser is requesting a Directory Index
				msg('dir', style.link(prettyPath), flags)

				const templateUrl = path.join(libPath, 'templates/directory.html')

				const dirInfo = dirToHtml(filePath)

				// Format the counts text
				let countsText = ''
				if (dirInfo.fileCount > 0) {
					countsText += `${dirInfo.fileCount} file${dirInfo.fileCount !== 1 ? 's' : ''}`
				}
				if (dirInfo.folderCount > 0) {
					if (countsText) countsText += ', '
					countsText += `${dirInfo.folderCount} folder${dirInfo.folderCount !== 1 ? 's' : ''}`
				}
				if (!countsText) {
					countsText = 'Empty folder'
				}

				const handlebarData: HandlebarData = {
					dirname: path.parse(filePath).dir,
					content: dirInfo.html,
					title: path.parse(filePath).base,
					pid: process.pid || 'N/A',
					breadcrumbs: createBreadcrumbs(path.relative(dir, filePath)),
					countsText
				}

				baseTemplate(templateUrl, handlebarData).then(final => {
					return processTemplate(final, implantHandlers).then(output => {
						res.writeHead(200, {
							'content-type': 'text/html'
						})
						res.end(output)
					}).catch((_error: Error) => {
						console.error(_error)
					})
				}).catch((_error: Error) => {
					console.error(_error)
				})
			} catch (error) {
				errorPage(500, filePath, error as Error)
			}
		} else if (isText) {
			// Text: Browser is requesting a text file
			msg('text', style.link(prettyPath), flags)
			logPageVisit(req, filePath, 'text')
			getFile(filePath).then(textContent => {
				// Use logToHTML for .log files, textToHTML for others
				const isLogFile = path.extname(filePath).toLowerCase() === '.log'
				const htmlContent = isLogFile ? logToHTML(textContent) : textToHTML(textContent, filePath)
				const templateUrl = path.join(libPath, 'templates/markdown.html')

				const stats = fs.statSync(filePath)
				const lastModified = stats.mtime.toLocaleString('en-US', {
					year: 'numeric',
					month: 'short',
					day: 'numeric',
					hour: '2-digit',
					minute: '2-digit'
				})

				const handlebarData: HandlebarData = {
					title: path.parse(filePath).base,
					content: htmlContent,
					pid: process.pid || 'N/A',
					filePath: prettyPath,
					fileName: path.basename(filePath),
					lastModified,
					parentDir: path.dirname(prettyPath) || '/'
				}

				return baseTemplate(templateUrl, handlebarData).then(final => {
					return processTemplate(final, implantHandlers)
						.then(output => {
							res.writeHead(200, {
								'content-type': 'text/html'
							})
							res.end(output)
						})
				})
			}).catch((error: Error) => {
				errorPage(500, filePath, error)
			})
		} else {
			// Check if file has text MIME type and should be rendered as text
			const mimeType = mime.contentType(path.extname(filePath))
			const isTextMime = mimeType && (mimeType.startsWith('text/') ||
				mimeType.includes('application/json') ||
				mimeType.includes('application/xml') ||
				mimeType.includes('application/javascript') ||
				mimeType.includes('application/x-sh'))

			if (isTextMime && req.headers.accept && req.headers.accept.includes('text/html')) {
				// Render any text MIME type file as formatted text
				msg('text (mime)', style.link(prettyPath), flags)
				getFile(filePath).then(textContent => {
					const htmlContent = textToHTML(textContent, filePath)
					const templateUrl = path.join(libPath, 'templates/markdown.html')

					const stats = fs.statSync(filePath)
					const lastModified = stats.mtime.toLocaleString('en-US', {
						year: 'numeric',
						month: 'short',
						day: 'numeric',
						hour: '2-digit',
						minute: '2-digit'
					})

					const handlebarData: HandlebarData = {
						title: path.parse(filePath).base,
						content: htmlContent,
						pid: process.pid || 'N/A',
						filePath: prettyPath,
						fileName: path.basename(filePath),
						lastModified
					}

					return baseTemplate(templateUrl, handlebarData).then(final => {
						return processTemplate(final, implantHandlers)
							.then(output => {
								res.writeHead(200, {
									'content-type': 'text/html'
								})
								res.end(output)
							})
					})
				}).catch((error: Error) => {
					errorPage(500, filePath, error)
				})
				return
			}

			// Other: Browser requests other MIME typed file
			msg('file', style.link(prettyPath), flags)
			logPageVisit(req, filePath, 'file')

			// Check if we should show a download page or serve directly
			const isDownloadable = !mimeType || !mimeType.startsWith('image/') && !mimeType.startsWith('video/') && !mimeType.startsWith('audio/')

			// For non-media files, show a download page
			if (isDownloadable && req.headers.accept && req.headers.accept.includes('text/html')) {
				const fileName = path.basename(filePath)
				const stats = fs.statSync(filePath)
				const fileSize = stats.size
				const fileSizeFormatted = formatFileSize(fileSize)
				const lastModified = stats.mtime.toLocaleString('en-US', {
					year: 'numeric',
					month: 'short',
					day: 'numeric',
					hour: '2-digit',
					minute: '2-digit'
				})

				// Check if this is an ONNX file to add visualization button
				const visualizeButton = isONNX
					? `<a href="/model-explorer${prettyPath}" class="download-button" style="background: #28a745; margin-left: 10px;">📊 Visualize in Model Explorer</a>`
					: ''

				const downloadPageHtml = `
<!DOCTYPE html>
<html>
<head>
	<meta charset="UTF-8">
	<meta name="viewport" content="width=device-width, initial-scale=1.0">
	<title>${fileName}</title>
	<link rel="stylesheet" href="{markserv}/templates/markserv.css">
	<style>
		.download-container {
			max-width: 800px;
			margin: 50px auto;
			padding: 30px;
			text-align: center;
		}
		.file-info {
			background: #f5f5f5;
			border-radius: 8px;
			padding: 20px;
			margin: 20px 0;
		}
		.download-button {
			display: inline-block;
			padding: 12px 30px;
			background: #0366d6;
			color: white;
			text-decoration: none;
			border-radius: 6px;
			font-size: 16px;
			transition: background 0.2s;
		}
		.download-button:hover {
			background: #0256c7;
		}
		.file-icon {
			font-size: 64px;
			margin: 20px 0;
		}
	</style>
</head>
<body>
	<article class="markdown-body">
		<div class="download-container">
			<div class="file-icon">📄</div>
			<h1>${fileName}</h1>
			<div class="file-info">
				<p><strong>File Type:</strong> ${mimeType || 'Unknown'}</p>
				<p><strong>File Size:</strong> ${fileSizeFormatted}</p>
				<p><strong>Last Modified:</strong> ${lastModified}</p>
			</div>
			<a href="${req.url}?download=true" class="download-button" download="${fileName}">⬇ Download File</a>
			${visualizeButton}
		</div>
		<footer><sup><hr> PID: ${process.pid}</sup></footer>
	</article>
</body>
</html>`
				res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' })
				res.end(downloadPageHtml)
			} else {
				// Serve the file directly (for images, videos, audio, or when download=true is in query)
				if (mimeType) {
					res.setHeader('Content-Type', mimeType)
				}
				// Add download header if requested
				if (req.url.includes('download=true')) {
					res.setHeader('Content-Disposition', `attachment; filename="${path.basename(filePath)}"`)
				}
				const stream = fs.createReadStream(filePath)
				stream.on('error', () => {
					res.status(404).end()
				})
				stream.pipe(res)
			}
		}
	}
}

const startExpressApp = (liveReloadPort: number | 'false', httpRequestHandler: (req: Request, res: Response) => void, watchEnabled: boolean): Application => {
	const app = express()
	app.use(compression())
	if (liveReloadPort && liveReloadPort !== 'false' && watchEnabled) {
		// LiveReload script injection handled in templates
		app.use((req: Request, res: Response, next: NextFunction) => {
			if (req.headers.accept && req.headers.accept.includes('text/html')) {
				res.locals.liveReloadPort = liveReloadPort
			}
			next()
		})
	}
	app.use('/', httpRequestHandler)
	return app
}

// Helper function to check if a port is available
const checkPortAvailable = (port: number, address: string): Promise<boolean> => {
	return new Promise((resolve) => {
		const tester = http.createServer()
		tester.once('error', (err: NodeJS.ErrnoException) => {
			if (err.code === 'EADDRINUSE') {
				resolve(false)
			} else {
				resolve(false)
			}
		})
		tester.once('listening', () => {
			tester.close(() => resolve(true))
		})
		tester.listen(port, address)
	})
}

// Helper function to find an available port
const findAvailablePort = async (startPort: number, address: string, maxAttempts: number = 10): Promise<number> => {
	for (let i = 0; i < maxAttempts; i++) {
		const port = startPort + i
		const available = await checkPortAvailable(port, address)
		if (available) {
			return port
		}
	}
	throw new Error(`Could not find available port after ${maxAttempts} attempts starting from ${startPort}`)
}

const startHTTPServer = async (expressApp: Application | null, port: number | string, flags: Flags): Promise<HttpServerResult> => {
	let httpServer: HttpServer

	if (expressApp) {
		httpServer = http.createServer(expressApp)
	} else {
		httpServer = http.createServer()
	}

	const portNum = typeof port === 'string' ? parseInt(port, 10) : port

	// Try to listen on the specified port
	return new Promise((resolve, reject) => {
		httpServer.once('error', async (err: NodeJS.ErrnoException) => {
			if (err.code === 'EADDRINUSE') {
				msg('port', chalk.yellow(`Port ${portNum} is already in use, finding an available port...`), flags)
				try {
					const newPort = await findAvailablePort(portNum + 1, flags.address)
					msg('port', chalk.green(`Using port ${newPort} instead`), flags)
					httpServer.listen(newPort, flags.address, () => {
						resolve({ server: httpServer, port: newPort })
					})
				} catch (findErr) {
					reject(findErr)
				}
			} else {
				reject(err)
			}
		})

		httpServer.once('listening', () => {
			const address = httpServer.address()
			const actualPort = typeof address === 'object' && address !== null ? address.port : portNum
			resolve({ server: httpServer, port: actualPort })
		})

		httpServer.listen(portNum, flags.address)
	})
}

interface LiveReloadResult {
	wss: WebSocketServer | null
	watcher: FSWatcher | null
	port: number
}

const startLiveReloadServer = async (liveReloadPort: number, flags: Flags): Promise<LiveReloadResult> => {
	let {dir} = flags
	const isDir = fs.statSync(dir).isDirectory()
	if (!isDir) {
		dir = path.parse(flags.dir).dir
	}

	msg('watch', path.dirname(dir), flags)

	// Try to create WebSocket server for live reload with port conflict handling
	let actualPort = liveReloadPort
	let wss: WebSocketServer

	// First check if the port is available
	const portAvailable = await checkPortAvailable(liveReloadPort, '::')

	if (!portAvailable) {
		msg('livereload', chalk.yellow(`LiveReload port ${liveReloadPort} is already in use, finding an available port...`), flags)
		try {
			actualPort = await findAvailablePort(liveReloadPort + 1, '::', 10)
			msg('livereload', chalk.green(`Using LiveReload port ${actualPort} instead`), flags)
		} catch (findErr) {
			errormsg('livereload', `Could not find available port for LiveReload: ${(findErr as Error).message}`, flags, findErr as Error)
			// Return a dummy object so the server can still start without LiveReload
			return { wss: null, watcher: null, port: 0 }
		}
	}

	// Create the WebSocket server with the available port
	wss = new WebSocketServer({ port: actualPort })

	// Handle any runtime errors
	wss.on('error', (err: Error) => {
		errormsg('livereload', `LiveReload server error: ${err.message}`, flags, err)
	})

	// Track connected clients
	const clients = new Set<any>()

	wss.on('connection', (ws) => {
		clients.add(ws)
		ws.on('close', () => {
			clients.delete(ws)
		})
	})

	// Watch for file changes
	const watcher = chokidar.watch(dir, {
		ignored: [
			/(^|[\/\\])\../, // Hidden files
			/node_modules/,
			/__pycache__/,
			/.git/
		],
		persistent: true
	})

	watcher.on('change', (filepath: string) => {
		msg('reload', filepath, flags)
		// Send reload message to all connected clients
		clients.forEach(ws => {
			if (ws.readyState === ws.OPEN) {
				ws.send(JSON.stringify({ type: 'reload' }))
			}
		})
	})

	return { wss, watcher, port: actualPort }
}

const logActiveServerInfo = async (serveURL: string, _actualHttpPort: number, liveReloadPort: number, flags: Flags): Promise<void> => {
	const dir = path.resolve(flags.dir)

	const githubLink = 'github.com/litanlitudan/markserv'

	msg('address', style.address(serveURL), flags)
	msg('path', chalk.grey(style.address(dir)), flags)
	msg('livereload', chalk.grey(`communicating on port: ${style.port(liveReloadPort)}`), flags)

	if (process.pid) {
		msg('process', chalk.grey(`your pid is: ${style.pid(process.pid)}`), flags)
		msg('stop', chalk.grey(`press ${chalk.magenta('[Ctrl + C]')} or type ${chalk.magenta(`"sudo kill -9 ${process.pid}"`)}`), flags)
	}

	msg('github', `Contribute on Github - ${chalk.yellow.underline(githubLink)}`, flags)
}

const checkForUpgrade = (): Promise<boolean> => new Promise((resolve) => {
	// For now, skip upgrade check since analyze-deps is not available
	// This can be replaced with a simpler npm API check
	resolve(false)
})

const optionalUpgrade = async (flags: Flags): Promise<void> => {
	if (flags.silent) {
		return
	}

	msg('upgrade', 'checking for upgrade...', flags)

	return checkForUpgrade().then(async version => {
		if (version === false) {
			msg('upgrade', 'no upgrade available', flags)
			return
		}

		msg(chalk.bgRed('✨UPGRADE✨'), `Markserv version: ${version} is available!`, flags)

		const logInstallNotes = (): void => {
			msg(chalk.bgRed('✨UPGRADE✨'), 'Upgrade cancelled. To upgrade manually:', flags)
			msg(chalk.bgRed('✨UPGRADE✨'), chalk.bgYellow.black.bold(` npm i -g markserv@${version} `), flags)
			msg(chalk.bgRed('✨UPGRADE✨'), chalk.bgYellow.black.bold(` yarn global add markserv@${version} `), flags)
		}

		// For now, just log the install notes since promptly is not available
		logInstallNotes()
	}).catch((error: Error) => {
		console.error(error)
	})
}

const init = async (flags: Flags): Promise<MarkservService> => {
	const liveReloadPort = flags.livereloadport
	const httpPort = flags.port
	const watchEnabled = flags.watch || false

	const httpRequestHandler = createRequestHandler(flags)

	// First, determine the actual LiveReload port if needed
	let actualLiveReloadPort: number = typeof liveReloadPort === 'number' ? liveReloadPort : 0
	let liveReloadServer: LiveReloadResult | undefined
	if (liveReloadPort && liveReloadPort !== 'false' && watchEnabled) {
		const lrResult = await startLiveReloadServer(typeof liveReloadPort === 'number' ? liveReloadPort : 35729, flags)
		liveReloadServer = lrResult
		actualLiveReloadPort = lrResult.port
	}

	// Create Express app with the actual LiveReload port
	const expressApp = startExpressApp(actualLiveReloadPort, httpRequestHandler, watchEnabled)

	// Start HTTP server with automatic port finding
	const httpResult = await startHTTPServer(expressApp, httpPort, flags)
	const actualHttpPort = httpResult.port
	const httpServer = httpResult.server

	const serveURL = `http://${flags.address}:${actualHttpPort}`

	// Log server info to CLI with actual ports
	if (watchEnabled) {
		logActiveServerInfo(serveURL, actualHttpPort, actualLiveReloadPort, flags)
	} else {
		// Log without LiveReload info
		const dir = path.resolve(flags.dir)
		const githubLink = 'github.com/litanlitudan/markserv'

		msg('address', style.address(serveURL), flags)
		msg('path', chalk.grey(style.address(dir)), flags)
		msg('watch', chalk.grey('file watching disabled (use --watch to enable)'), flags)

		if (process.pid) {
			msg('process', chalk.grey(`your pid is: ${style.pid(process.pid)}`), flags)
			msg('stop', chalk.grey(`press ${chalk.magenta('[Ctrl + C]')} or type ${chalk.magenta(`"sudo kill -9 ${process.pid}"`)}`), flags)
		}

		msg('github', `Contribute on Github - ${chalk.yellow.underline(githubLink)}`, flags)
	}

	let launchUrl: string | false = false
	if (flags.$openLocation || flags.$pathProvided) {
		launchUrl = `${serveURL}/${flags.$openLocation}`
	}

	// Update flags with actual ports for other functions to use
	flags.port = actualHttpPort
	if (liveReloadPort && liveReloadPort !== 'false' && watchEnabled) {
		flags.livereloadport = actualLiveReloadPort
	}

	const service: MarkservService = {
		pid: process.pid,
		httpServer,
		liveReloadServer,
		expressApp: httpServer, // Express app is embedded in the httpServer
		launchUrl
	}

	const launchBrowser = (): void => {
		if (flags.browser === false ||
			flags.browser === 'false') {
			return
		}

		if (launchUrl) {
			msg('browser', `Opening browser at: ${launchUrl}`, flags)
			open(launchUrl)
		}
	}

	// Only check for upgrades when online
	isOnline({timeout: 5000}).then(() => {
		optionalUpgrade(flags)
	})
	launchBrowser()

	return service
}

export const createMarkservApp = (flags: Flags): Application => {
	const normalizedFlags: Flags = {
		...flags,
		port: typeof flags.port === 'undefined' ? 0 : flags.port,
		watch: false,
		livereloadport: 'false',
		verbose: flags.verbose ?? false,
		silent: flags.silent ?? true,
	}

	const httpRequestHandler = createRequestHandler(normalizedFlags)
	return startExpressApp(0, httpRequestHandler, false)
}

export default {
	getFile,
	markdownToHTML,
	init,
	createMarkservApp
}
