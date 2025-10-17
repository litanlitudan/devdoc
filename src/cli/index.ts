#!/usr/bin/env node

/**
 * devdoc CLI entry point (oclif)
 *
 * This is the new CLI architecture using oclif.
 * Enable with: MARKSERV_USE_NEW_CLI=1
 */

import { run } from '@oclif/core'

// Run oclif
run(process.argv.slice(2), import.meta.url).catch((error: unknown) => {
	// Handle oclif errors
	if (error instanceof Error) {
		console.error(error)
	} else {
		console.error('Unknown CLI error', error)
	}
	process.exit(1)
})
