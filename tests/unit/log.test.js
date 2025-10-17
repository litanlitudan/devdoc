import { describe, it, expect } from 'vitest'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import request from 'supertest'
import { createDevdocApp } from '../../dist/lib/server.js'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const app = createDevdocApp({
	dir: path.join(__dirname, '..', '..'),
	port: 0,
	address: '127.0.0.1',
	livereloadport: 'false',
	watch: false,
	silent: true,
	verbose: false,
	browser: false,
})

const supertestAvailable = process.env.MARKSERV_ENABLE_SUPERTEST === '1'

const describeIf = supertestAvailable ? describe : describe.skip
const itIf = supertestAvailable ? it : it.skip

describeIf('Log file rendering', () => {
	itIf('should render .log files with consistent sizing styles', async () => {
		const response = await request(app).get('/tests/fixtures/test.log')

		expect(response.status).toBe(200)
		expect(response.headers['content-type']).toMatch(/text\/html/)

		const html = response.text
		expect(html).toContain('class="log-output-container"')
		expect(html).toContain('class="log-output-header"')
		expect(html).toContain('class="log-output-content"')
		expect(html).toContain('class="log-output-code"')

		expect(html).toContain('.log-output-container')
		expect(html).toContain('font-size: 13px !important')
		expect(html).toContain('font-size: 11px !important')
		expect(html).toContain('body.expanded .markdown-body .log-output-container')
		expect(html).toContain('body.expanded .markdown-body .log-output-header')
		expect(html).toContain('body.expanded .markdown-body .log-output-code')

		expect(html).toContain('INFO')
		expect(html).toContain('WARNING')
		expect(html).toContain('ERROR')
		expect(html).toContain('CRITICAL')

		expect(html).toContain('color: #74c0fc')
		expect(html).toContain('color: #ffa94d')
		expect(html).toContain('color: #ff6b6b')
	})
})
