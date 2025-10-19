import { describe, it, expect } from 'vitest'
import request from 'supertest'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { createDevdocApp } from '../../dist/lib/server.js'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const baseFlags = {
	dir: path.join(__dirname, '..', '..'),
	port: 0,
	address: '127.0.0.1',
	livereloadport: 'false',
	watch: false,
	silent: true,
	verbose: false,
	browser: false,
}

const app = createDevdocApp({ ...baseFlags })

const supertestAvailable = process.env.DEVDOC_ENABLE_SUPERTEST === '1'

const describeIf = supertestAvailable ? describe : describe.skip
const itIf = supertestAvailable ? it : it.skip

describeIf('Raw file route for direct downloads', () => {
	itIf('should download files directly via /raw/ route', async () => {
		const response = await request(app).get('/raw/package.json')

		expect(response.status).toBe(200)
		expect(response.headers['content-type']).toMatch(/application\/json/)
		expect(response.headers['content-disposition']).toBe(
			'attachment; filename="package.json"',
		)

		const actualContent = fs.readFileSync(
			path.join(__dirname, '..', '..', 'package.json'),
			'utf8',
		)
		expect(response.text).toBe(actualContent)
	})

	itIf('should download JavaScript files via /raw/ route', async () => {
		const response = await request(app).get('/raw/dist/lib/server.js')

		expect(response.status).toBe(200)
		expect(response.headers['content-type']).toMatch(/javascript/)
		expect(response.headers['content-disposition']).toBe(
			'attachment; filename="server.js"',
		)

		expect(response.text.length).toBeGreaterThan(0)
	})

	itIf('should return 404 for non-existent files via /raw/ route', async () => {
		const response = await request(app).get('/raw/nonexistent.js')

		expect(response.status).toBe(404)
		expect(response.text).toBe('File not found')
	})

	itIf('should return 400 for directories via /raw/ route', async () => {
		const response = await request(app).get('/raw/lib/')

		expect(response.status).toBe(400)
		expect(response.text).toBe('Raw file route does not support directories')
	})

	itIf('should work with curl-like user agents', async () => {
		const response = await request(app)
			.get('/raw/package.json')
			.set('User-Agent', 'curl/7.64.1')

		expect(response.status).toBe(200)
		expect(response.headers['content-disposition']).toBe(
			'attachment; filename="package.json"',
		)
	})

	itIf('should handle files with special characters in names', async () => {
		const testFile = path.join(__dirname, 'test file with spaces.txt')
		fs.writeFileSync(testFile, 'test content')

		try {
			const response = await request(app).get(
				'/raw/tests/unit/test%20file%20with%20spaces.txt',
			)

			expect(response.status).toBe(200)
			expect(response.headers['content-disposition']).toBe(
				'attachment; filename="test file with spaces.txt"',
			)
			expect(response.text).toBe('test content')
		} finally {
			if (fs.existsSync(testFile)) {
				fs.unlinkSync(testFile)
			}
		}
	})
})
