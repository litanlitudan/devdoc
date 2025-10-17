#!/usr/bin/env node

import path from 'node:path'
import fs from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname } from 'node:path'
import yargs from 'yargs'
import { hideBin } from 'yargs/helpers'
import markserv from './server.js'
import splash from './splash.js'
import type { Flags } from './types.js'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

// In compiled dist/cli.js, __dirname is 'dist/', so we need to go up to find lib/
const cliHelp = String(
	fs.readFileSync(path.join(__dirname, '..', 'lib', 'cli-help.txt')),
)

const cliOpts = yargs(hideBin(process.argv))
	.usage(cliHelp)
	.option('port', {
		alias: 'p',
		default: '8642',
		describe: 'Port to run server on',
		type: 'string',
	})
	.option('livereloadport', {
		alias: 'b',
		default: 35729,
		describe: 'Port for LiveReload server',
		type: 'number',
	})
	.option('address', {
		alias: 'a',
		default: 'localhost',
		describe: 'Address to bind server to',
		type: 'string',
	})
	.option('silent', {
		alias: 's',
		default: false,
		describe: 'Silent mode',
		type: 'boolean',
	})
	.option('verbose', {
		alias: 'v',
		default: false,
		describe: 'Verbose output',
		type: 'boolean',
	})
	.option('watch', {
		alias: 'w',
		default: false,
		describe: 'Enable file watching and live reload',
		type: 'boolean',
	})
	.help().argv

const validateServerPath = (
	serverPath: string | undefined,
	cwd: string,
): string => {
	if (!serverPath || serverPath === cwd) {
		return cwd
	}

	// If it's already an absolute path, just normalize it
	if (path.isAbsolute(serverPath)) {
		return path.normalize(serverPath)
	}

	// Otherwise, resolve it relative to cwd
	return path.resolve(cwd, serverPath)
}

const run = async (opts: any): Promise<any> => {
	const flags: Flags = await Promise.resolve(opts)
	splash(flags)

	const cwd = process.cwd()

	let dir = (flags as any)._?.[0] as string | undefined
	if (dir === undefined) {
		dir = cwd
	}

	const validatedServerPath = validateServerPath(dir, cwd)
	flags.dir = validatedServerPath
	flags.$pathProvided = true
	flags.$openLocation = true

	return markserv.init(flags)
}

// Check if this module is being run directly
if (import.meta.url === `file://${process.argv[1]}`) {
	// Run without args (process.argv will be picked up)
	run(cliOpts)
}

export default { run }
