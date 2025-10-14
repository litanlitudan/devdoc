import {describe, it, expect, beforeAll, afterAll} from 'vitest'
import getPort from 'get-port'
import serverModule from '../../dist/server.js'
import fs from 'node:fs'
import path from 'node:path'
import {fileURLToPath} from 'node:url'

const {init} = serverModule
const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

describe('Log file rendering', () => {
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

	it('should render .log files with consistent sizing styles', async () => {
		const response = await fetch(`http://localhost:${port}/tests/fixtures/test.log`)
		const html = await response.text()

		// Check that the response is HTML
		expect(response.headers.get('content-type')).toMatch(/text\/html/)

		// Check for class-based styling (not inline styles)
		expect(html).toContain('class="log-output-container"')
		expect(html).toContain('class="log-output-header"')
		expect(html).toContain('class="log-output-content"')
		expect(html).toContain('class="log-output-code"')

		// Check for CSS with !important rules for consistent sizing
		expect(html).toContain('.log-output-container')
		expect(html).toContain('font-size: 13px !important')
		expect(html).toContain('font-size: 11px !important')
		expect(html).toContain('body.expanded .markdown-body .log-output-container')
		expect(html).toContain('body.expanded .markdown-body .log-output-header')
		expect(html).toContain('body.expanded .markdown-body .log-output-code')

		// Check for log level highlighting
		expect(html).toContain('INFO')
		expect(html).toContain('WARNING')
		expect(html).toContain('ERROR')
		expect(html).toContain('CRITICAL')

		// Check for colored spans
		expect(html).toContain('color: #74c0fc') // INFO color
		expect(html).toContain('color: #ffa94d') // WARNING color
		expect(html).toContain('color: #ff6b6b') // ERROR color
	})
})
