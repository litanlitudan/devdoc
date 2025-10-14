import {describe, it, expect, beforeAll, afterAll} from 'vitest'
import getPort from 'get-port'
import serverModule from '../../dist/server.js'
import fs from 'node:fs'
import path from 'node:path'
import {fileURLToPath} from 'node:url'

const {init} = serverModule
const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

describe('API route for direct downloads', () => {
	let server
	let port

	beforeAll(async () => {
		port = await getPort()
		const flags = {
			dir: path.join(__dirname, '..', '..'),
			port,
			address: 'localhost',
			livereloadport: false,
			silent: true,
			verbose: false
		}

		server = await init(flags)
	})

	afterAll(async () => {
		if (server && server.httpServer) {
			await new Promise((resolve) => server.httpServer.close(resolve))
		}
	})

	it('should download files directly via /api/ route', async () => {
		// Test downloading package.json via API route
		const response = await fetch(`http://localhost:${port}/api/package.json`)
		const content = await response.text()

		// Check response headers
		expect(response.headers.get('content-type')).toMatch(/application\/json/)
		expect(response.headers.get('content-disposition')).toBe('attachment; filename="package.json"')

		// Verify content matches actual file
		const actualContent = fs.readFileSync(path.join(__dirname, '..', '..', 'package.json'), 'utf8')
		expect(content).toBe(actualContent)
	})

	it('should download JavaScript files via /api/ route', async () => {
		// Test downloading a JS file
		const response = await fetch(`http://localhost:${port}/api/dist/cli.js`)

		// Check response headers
		expect(response.headers.get('content-type')).toMatch(/application\/javascript/)
		expect(response.headers.get('content-disposition')).toBe('attachment; filename="cli.js"')

		// Verify content
		const content = await response.text()
		expect(content.length).toBeGreaterThan(0)
		expect(content).toContain('#!/usr/bin/env node')
	})

	it('should return 404 for non-existent files via /api/ route', async () => {
		const response = await fetch(`http://localhost:${port}/api/nonexistent.js`)

		expect(response.status).toBe(404)
		const content = await response.text()
		expect(content).toBe('File not found')
	})

	it('should return 400 for directories via /api/ route', async () => {
		const response = await fetch(`http://localhost:${port}/api/lib/`)

		expect(response.status).toBe(400)
		const content = await response.text()
		expect(content).toBe('API route does not support directories')
	})

	it('should work with curl-like user agents', async () => {
		// Simulate curl request
		const response = await fetch(`http://localhost:${port}/api/package.json`, {
			headers: {
				'User-Agent': 'curl/7.64.1'
			}
		})

		expect(response.status).toBe(200)
		expect(response.headers.get('content-disposition')).toBe('attachment; filename="package.json"')
	})

	it('should handle files with special characters in names', async () => {
		// Create a test file with special characters
		const testFile = path.join(__dirname, 'test file with spaces.txt')
		fs.writeFileSync(testFile, 'test content')

		try {
			const response = await fetch(`http://localhost:${port}/api/tests/unit/test%20file%20with%20spaces.txt`)

			expect(response.status).toBe(200)
			expect(response.headers.get('content-disposition')).toBe('attachment; filename="test file with spaces.txt"')

			const content = await response.text()
			expect(content).toBe('test content')
		} finally {
			// Cleanup
			if (fs.existsSync(testFile)) {
				fs.unlinkSync(testFile)
			}
		}
	})
})
