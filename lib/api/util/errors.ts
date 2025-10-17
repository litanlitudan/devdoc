/**
 * Error handling utilities for API
 * Provides consistent error responses across all endpoints
 */

import createHttpError, { type HttpError } from 'http-errors'
import type { Request, Response, NextFunction } from 'express'
import { ZodError, type ZodIssue } from 'zod'
import { logger } from './logger.js'

/**
 * Standard error response format
 */
export interface ErrorResponse {
	error: {
		message: string
		code?: string
		status: number
		details?: any
	}
}

/**
 * Create an HTTP error with consistent format
 */
export function createApiError(
	status: number,
	message: string,
	code?: string,
	details?: any,
): HttpError {
	const error = createHttpError(status, message) as HttpError & {
		code?: string
		details?: any
	}
	if (code) error.code = code
	if (details) error.details = details
	return error as HttpError
}

/**
 * Format Zod validation errors
 */
export function formatZodError(
	error: ZodError,
): Array<{ path: string; message: string; code: string }> {
	return error.errors.map((issue: ZodIssue) => ({
		path: issue.path.join('.'),
		message: issue.message,
		code: issue.code,
	}))
}

/**
 * Express error handling middleware
 * Catches all errors and returns consistent JSON responses
 */
export function errorHandler(
	err: unknown,
	req: Request,
	res: Response,
	_next: NextFunction,
): void {
	const error = err as any

	// Log the error
	logger.error(
		{
			err: error,
			req: {
				method: req.method,
				url: req.url,
				headers: req.headers,
			},
		},
		'API error',
	)

	// Handle Zod validation errors
	if (error instanceof ZodError) {
		const response: ErrorResponse = {
			error: {
				message: 'Validation failed',
				code: 'VALIDATION_ERROR',
				status: 400,
				details: formatZodError(error),
			},
		}
		res.status(400).json(response)
		return
	}

	// Handle HTTP errors from http-errors
	if (error?.status && error?.message) {
		const response: ErrorResponse = {
			error: {
				message: error.message,
				code: error.code || 'HTTP_ERROR',
				status: error.status,
				details: error.details,
			},
		}
		res.status(error.status).json(response)
		return
	}

	// Handle generic errors
	const response: ErrorResponse = {
		error: {
			message:
				process.env.NODE_ENV === 'production'
					? 'Internal server error'
					: (error instanceof Error && error.message) || 'Unknown error',
			code: 'INTERNAL_ERROR',
			status: 500,
		},
	}

	res.status(500).json(response)
}

/**
 * Async handler wrapper to catch promise rejections
 */
export function asyncHandler(
	fn: (req: Request, res: Response, next: NextFunction) => Promise<any>,
) {
	return (req: Request, res: Response, next: NextFunction) => {
		Promise.resolve(fn(req, res, next)).catch((error: unknown) => next(error))
	}
}

/**
 * Validation middleware factory
 * Validates request body, query, or params against a Zod schema
 */
export function validateRequest<T>(
	schema: { parse: (data: any) => T },
	source: 'body' | 'query' | 'params' = 'body',
) {
	return (req: Request, _res: Response, next: NextFunction) => {
		try {
			req[source] = schema.parse(req[source])
			next()
		} catch (error: unknown) {
			next(error)
		}
	}
}
