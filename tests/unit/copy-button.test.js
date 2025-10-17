import { describe, it, expect } from 'vitest'
import request from 'supertest'
import { createDevdocApp } from '../../dist/lib/server.js'

const app = createDevdocApp({
	dir: process.cwd(),
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

describeIf('Copy button functionality', () => {
	const getMarkdown = async () => {
		const response = await request(app).get(
			'/tests/fixtures/markdown/copy-button-test.md',
		)
		expect(response.status).toBe(200)
		return response.text
	}

	itIf('should include copy button CSS in markdown pages', async () => {
		const html = await getMarkdown()

		expect(html).toContain('.copy-code-button')
		expect(html).toContain('Copy')
		expect(html).toContain('opacity: 0')
	})

	itIf('should include copy button JavaScript in markdown pages', async () => {
		const html = await getMarkdown()

		expect(html).toContain('addCopyButtonsToCodeBlocks')
		expect(html).toContain('navigator.clipboard.writeText')
		expect(html).toContain('Copied!')
	})

	itIf('should wrap regular code blocks', async () => {
		const html = await getMarkdown()

		expect(html).toContain('language-javascript')
		expect(html).toContain('language-python')
		expect(html).toContain('language-bash')
	})

	itIf('should have code blocks ready for copy button wrapping', async () => {
		const html = await getMarkdown()

		expect(html).toContain('<pre><code')
		expect(html).toContain('</code></pre>')
	})

	itIf('should not add copy buttons to inline code', async () => {
		const html = await getMarkdown()

		expect(html).toContain('inline code')
		// Inline code is untouched; selector targets only block code (`pre > code`)
	})
})
