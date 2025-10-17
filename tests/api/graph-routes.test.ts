import { describe, it, expect, beforeAll, beforeEach, vi } from 'vitest'
import request from 'supertest'
import path from 'node:path'
import { createDevdocApp } from '../../dist/lib/server.js'

vi.mock('../../lib/api/services/mlir.js', () => ({
  parseMlirToGraph: vi.fn(),
  validateMlirSyntax: vi.fn(),
}))

const supertestAvailable = process.env.MARKSERV_ENABLE_SUPERTEST === '1'
const describeIf = supertestAvailable ? describe : describe.skip
const itIf = supertestAvailable ? it : it.skip

describeIf('Graph API Routes', () => {
  const app = createDevdocApp({
    dir: path.join(process.cwd(), 'tests'),
    port: 0,
    address: '127.0.0.1',
    livereloadport: 'false',
    watch: false,
    silent: true,
    verbose: false,
    browser: false,
  })

  type VitestMock = ReturnType<typeof vi.fn>
  let parseMlirToGraph: VitestMock
  let validateMlirSyntax: VitestMock

  beforeAll(async () => {
    const module = await import('../../lib/api/services/mlir.js')
    parseMlirToGraph = vi.mocked(module.parseMlirToGraph) as unknown as VitestMock
    validateMlirSyntax = vi.mocked(module.validateMlirSyntax) as unknown as VitestMock
  })

  beforeEach(() => {
    vi.clearAllMocks()
  })

  itIf('should return health status', async () => {
    const response = await request(app).get('/api/graph/health')

    expect(response.status).toBe(200)
    expect(response.body).toMatchObject({
      success: true,
      message: 'Graph API is healthy',
    })
  })

  itIf('should parse valid MLIR content', async () => {
    validateMlirSyntax.mockReturnValue({ valid: true })
    const mockGraph = { id: 'test', nodes: [] }
    parseMlirToGraph.mockResolvedValue(mockGraph as any)

    const response = await request(app)
      .post('/api/graph/mlir')
      .send({ content: 'func.func @main() { return }', filename: 'test.mlir' })

    expect(response.status).toBe(200)
    expect(response.body.success).toBe(true)
    expect(response.body.data).toEqual(mockGraph)
    expect(parseMlirToGraph).toHaveBeenCalledWith('func.func @main() { return }', 'test.mlir')
  })

  itIf('should use default filename when not provided', async () => {
    validateMlirSyntax.mockReturnValue({ valid: true })
    parseMlirToGraph.mockResolvedValue({ id: 'input.mlir', nodes: [] } as any)

    const response = await request(app)
      .post('/api/graph/mlir')
      .send({ content: 'func.func @main() { return }' })

    expect(response.status).toBe(200)
    expect(parseMlirToGraph).toHaveBeenCalledWith('func.func @main() { return }', 'input.mlir')
  })

  itIf('should reject missing content field', async () => {
    const response = await request(app)
      .post('/api/graph/mlir')
      .send({ filename: 'test.mlir' })

    expect(response.status).toBe(400)
    expect(response.body.error).toBeDefined()
  })

  itIf('should reject empty content', async () => {
    const response = await request(app)
      .post('/api/graph/mlir')
      .send({ content: '' })

    expect(response.status).toBe(400)
    expect(response.body.error).toBeDefined()
  })

  itIf('should reject invalid MLIR syntax', async () => {
    validateMlirSyntax.mockReturnValue({ valid: false, error: 'Invalid MLIR structure' })

    const response = await request(app)
      .post('/api/graph/mlir')
      .send({ content: 'invalid mlir' })

    expect(response.status).toBe(400)
    expect(response.body.error).toMatchObject({ message: expect.stringContaining('Invalid MLIR') })
  })

  itIf('should handle parsing errors', async () => {
    validateMlirSyntax.mockReturnValue({ valid: true })
    parseMlirToGraph.mockRejectedValue(new Error('Parse error'))

    const response = await request(app)
      .post('/api/graph/mlir')
      .send({ content: 'func.func @main() { return }' })

    expect(response.status).toBe(500)
    expect(response.body.error).toBeDefined()
  })

  itIf('should handle malformed JSON', async () => {
    const response = await request(app)
      .post('/api/graph/mlir')
      .set('Content-Type', 'application/json')
      .send('invalid json')

    expect(response.status).toBe(400)
  })

  itIf('should handle 404 for unknown routes', async () => {
    const response = await request(app).get('/api/graph/unknown')

    expect(response.status).toBe(404)
  })

  itIf('should accept application/json payload', async () => {
    validateMlirSyntax.mockReturnValue({ valid: true })
    parseMlirToGraph.mockResolvedValue({ id: 'json', nodes: [] } as any)

    const response = await request(app)
      .post('/api/graph/mlir')
      .set('Content-Type', 'application/json')
      .send(JSON.stringify({ content: 'func.func @main() { return }' }))

    expect(response.status).toBe(200)
  })

  itIf('should reject non-JSON content', async () => {
    const response = await request(app)
      .post('/api/graph/mlir')
      .set('Content-Type', 'text/plain')
      .send('func.func @main() { return }')

    expect(response.status).toBe(400)
  })
})
