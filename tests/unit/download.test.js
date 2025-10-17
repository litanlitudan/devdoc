import { describe, it, expect } from 'vitest'
import request from 'supertest'
import { join } from 'node:path'
import { createMarkservApp } from '../../dist/lib/server.js'

const buildApp = (overrides = {}) =>
	createMarkservApp({
		dir: process.cwd(),
		port: 0,
		address: '127.0.0.1',
		livereloadport: 'false',
		watch: false,
		silent: true,
		verbose: false,
		browser: false,
		...overrides,
	})

const supertestAvailable = process.env.MARKSERV_ENABLE_SUPERTEST === '1'

const describeIf = supertestAvailable ? describe : describe.skip
const itIf = supertestAvailable ? it : it.skip

describeIf('Download functionality', () => {
	itIf('should show download button on markdown files', async () => {
		const app = buildApp()
		const response = await request(app).get('/README.md')

		expect(response.status).toBe(200)
		const html = response.text
		expect(html).toContain('download-link')
		expect(html).toContain('Download')
		expect(html).toContain('download=true')
	})

	itIf('should render text files as formatted HTML with download link', async () => {
		const app = buildApp()
		const response = await request(app)
			.get('/package.json')
			.set('Accept', 'text/html')

		expect(response.status).toBe(200)
		const html = response.text
		expect(html).toContain('<pre><code class="language-json hljs">')
		expect(html).toContain('&quot;name&quot;: &quot;markserv&quot;')
		expect(html).toContain('download-link')
		expect(html).toContain('⬇ Download')
		expect(html).toContain('package.json?download=true')
	})

	itIf('should add Content-Disposition header with download=true parameter', async () => {
		const app = buildApp()
		const response = await request(app).get('/package.json?download=true')

		expect(response.status).toBe(200)
		expect(response.headers['content-disposition']).toContain('attachment')
		expect(response.headers['content-disposition']).toContain('package.json')
	})

	itIf('should serve images directly without download page', async () => {
		const app = buildApp({ dir: join(process.cwd(), 'media') })
		const response = await request(app)
			.get('/markserv-splash.png')
			.set('Accept', 'text/html')

		expect(response.status).toBe(200)
		expect(response.headers['content-type']).toContain('image/png')
		expect(response.headers['content-type']).not.toContain('text/html')
	}, 10000)
})
