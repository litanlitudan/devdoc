import { describe, it, expect } from 'vitest'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import request from 'supertest'
import { createMarkservApp } from '../../dist/lib/server.js'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const app = createMarkservApp({
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

describeIf('Diff file rendering', () => {
	itIf('should render .diff files with syntax highlighting', async () => {
		const response = await request(app).get('/tests/fixtures/test.diff')

		expect(response.status).toBe(200)
		expect(response.headers['content-type']).toMatch(/text\/html/)

		const html = response.text
		expect(html).toContain('<pre><code class="language-diff hljs">')
		expect(html).toContain('diff --git')
		expect(html).toContain('@@')
		expect(html).toContain('test.diff')
		expect(html).toContain('languages/diff.min.js')
	})

	itIf('should render diff code blocks in markdown', async () => {
		const mdPath = path.join(
			__dirname,
			'..',
			'fixtures',
			'markdown',
			'test-diff-block.md',
		)

		const mdContent = `# Test Diff Block

\`\`\`diff
diff --git a/file.js b/file.js
@@ -1,3 +1,4 @@
 function test() {
-  console.log('old');
+  console.log('new');
+  console.log('added');
 }
\`\`\`
`

		fs.writeFileSync(mdPath, mdContent)

		try {
			const response = await request(app).get(
				'/tests/fixtures/markdown/test-diff-block.md',
			)

			expect(response.status).toBe(200)
			const html = response.text
			expect(html).toContain('language-diff')
			expect(html).toContain('languages/diff.min.js')
		} finally {
			if (fs.existsSync(mdPath)) {
				fs.unlinkSync(mdPath)
			}
		}
	})

	itIf('should render .patch files with syntax highlighting', async () => {
		const patchPath = path.join(__dirname, '..', 'fixtures', 'test.patch')
		const patchContent = `From abc123 Mon Sep 17 00:00:00 2001
From: Test User <test@example.com>
Date: Thu, 28 Nov 2024 10:00:00 +0000
Subject: [PATCH] Test patch file

---
 file.js | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)

diff --git a/file.js b/file.js
index abc123..def456 100644
--- a/file.js
+++ b/file.js
@@ -1,3 +1,3 @@
 function test() {
-  return 'old value';
+  return 'new value';
 }
`

		fs.writeFileSync(patchPath, patchContent)

		try {
			const response = await request(app).get('/tests/fixtures/test.patch')

			expect(response.status).toBe(200)
			const html = response.text
			expect(html).toContain('language-diff')
			expect(html).toContain('test.patch')
		} finally {
			if (fs.existsSync(patchPath)) {
				fs.unlinkSync(patchPath)
			}
		}
	})
})
