/**
 * Vitest test setup file
 * Runs before all tests to configure the test environment
 */

import { beforeAll, afterAll, vi } from 'vitest'

// Set test environment
process.env.NODE_ENV = 'test'
process.env.LOG_LEVEL = 'silent' // Silence logs during tests

// Mock get-port to return a deterministic port during tests that still require it
vi.mock('get-port', () => {
	const getPortMock = vi.fn(async () => 54321)
	return {
		__esModule: true,
		default: getPortMock,
	}
})

// Mock timers can be enabled globally if needed
// vi.useFakeTimers()

beforeAll(() => {
	// Global test setup
	console.log('🧪 Running test suite...')
})

afterAll(() => {
	// Global test cleanup
	vi.clearAllMocks()
	vi.restoreAllMocks()
})

// Add custom matchers or global test utilities here if needed
