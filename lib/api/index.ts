/**
 * API Router Index - Registers all API routes
 * Provides centralized API route management for the Express server
 */

import { Router } from 'express'
import graphRoutes from './routes/graph.js'
import { errorHandler } from './util/errors.js'
import { createLogger } from './util/logger.js'

const logger = createLogger('api')

/**
 * Create and configure the API router
 */
export function createApiRouter(): Router {
	const router = Router()

	logger.info('Registering API routes...')

	// Register route modules
	router.use('/graph', graphRoutes)

	// Health check endpoint
	router.get('/health', (_req, res) => {
		res.json({
			success: true,
			message: 'API is healthy',
			timestamp: new Date().toISOString(),
		})
	})

	logger.info('API routes registered successfully')

	return router
}

/**
 * Apply error handling middleware
 * Should be called after all routes are registered
 */
export { errorHandler }
