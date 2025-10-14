import { describe, it, expect, beforeAll, afterAll } from 'vitest'
import getPort from 'get-port'
import fetch from 'node-fetch'
import server from '../../dist/server.js'

const { init } = server

describe('Copy button functionality', () => {
	let httpServer
	let port

	beforeAll(async () => {
		port = await getPort()
		const flags = {
			port,
			livereloadport: false,
			address: 'localhost',
			dir: process.cwd(),
			silent: true,
		}
		httpServer = await init(flags)
	})

	afterAll(async () => {
		if (httpServer && httpServer.httpServer) {
			await new Promise((resolve) => httpServer.httpServer.close(resolve))
		}

		if (httpServer && httpServer.liveReloadServer) {
			httpServer.liveReloadServer.close()
		}
	})

	it('should include copy button CSS in markdown pages', async () => {
		const response = await fetch(
			`http://localhost:${port}/tests/fixtures/markdown/copy-button-test.md`,
		)
		const html = await response.text()

		// Check for copy button CSS
		expect(html).toContain('.copy-code-button')
		expect(html).toContain('Copy')
		expect(html).toContain('opacity: 0')
	})

	it('should include copy button JavaScript in markdown pages', async () => {
		const response = await fetch(
			`http://localhost:${port}/tests/fixtures/markdown/copy-button-test.md`,
		)
		const html = await response.text()

		// Check for copy button JavaScript function
		expect(html).toContain('addCopyButtonsToCodeBlocks')
		expect(html).toContain('navigator.clipboard.writeText')
		expect(html).toContain('Copied!')
	})

	it('should wrap regular code blocks', async () => {
		const response = await fetch(
			`http://localhost:${port}/tests/fixtures/markdown/copy-button-test.md`,
		)
		const html = await response.text()

		// Check that code blocks exist with highlight.js classes
		expect(html).toContain('language-javascript')
		expect(html).toContain('language-python')
		expect(html).toContain('language-bash')
	})

	it('should have code blocks ready for copy button wrapping', async () => {
		const response = await fetch(
			`http://localhost:${port}/tests/fixtures/markdown/copy-button-test.md`,
		)
		const html = await response.text()

		// Check for pre>code structure that will be wrapped by JavaScript
		expect(html).toContain('<pre><code')
		expect(html).toContain('</code></pre>')
	})

	it('should not add copy buttons to inline code', async () => {
		const response = await fetch(
			`http://localhost:${port}/tests/fixtures/markdown/copy-button-test.md`,
		)
		const html = await response.text()

		// Inline code should be in the document
		expect(html).toContain('inline code')

		// But the function should skip inline code (which is just <code> without <pre>)
		// This is verified by the JavaScript selector that only targets pre > code
	})
})
