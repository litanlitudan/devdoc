/**
 * Graph API Routes - MLIR/ONNX parsing endpoints
 * Provides RESTful API for graph generation from MLIR/ONNX files
 */

import { Router } from 'express'
import { validateRequest, asyncHandler, createApiError } from '../util/errors.js'
import { MLIRParseRequestSchema } from '../schemas/graph.js'
import { parseMlirToGraph, validateMlirSyntax } from '../services/mlir.js'
import { createLogger } from '../util/logger.js'

const router = Router()
const logger = createLogger('graph-routes')

/**
 * POST /api/graph/mlir
 * Parse MLIR content and return Model Explorer graph
 */
router.post(
	'/mlir',
	validateRequest(MLIRParseRequestSchema, 'body'),
	asyncHandler(async (req, res) => {
		const { content, filename } = req.body

		logger.info({ filename }, 'Received MLIR parse request')

		// Validate MLIR syntax
		const validation = validateMlirSyntax(content)
		if (!validation.valid) {
			throw createApiError(400, validation.error || 'Invalid MLIR syntax')
		}

		// Parse MLIR to graph
		const graph = await parseMlirToGraph(content, filename)

		res.json({
			success: true,
			data: graph,
		})
	}),
)

/**
 * GET /api/graph/health
 * Health check endpoint
 */
router.get('/health', (_req, res) => {
	res.json({
		success: true,
		message: 'Graph API is healthy',
	})
})

export default router
