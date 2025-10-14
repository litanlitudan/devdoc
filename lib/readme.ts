#!/usr/bin/env node

import fs from 'node:fs'
import path from 'node:path'
import yargs from 'yargs'
import { hideBin } from 'yargs/helpers'
import { fileURLToPath } from 'node:url'
import { dirname } from 'node:path'
import markserv from './server.js'
import splash from './splash.js'
import type { Flags } from './types.js'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

// In compiled dist/readme.js, __dirname is 'dist/', so we need to go up to find lib/
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
	.help().argv

const fileExistsSync = (uri: string): boolean => {
	let exists: boolean

	try {
		const stat = fs.statSync(uri)
		if (stat.isFile()) {
			exists = true
		} else {
			exists = false
		}
	} catch (error) {
		exists = false
	}

	return exists
}

const findUpFile = (dir: string, fileToFind: string): string | false => {
	const filepath = path.join(dir, fileToFind)
	const existsHere = fileExistsSync(filepath)

	if (existsHere) {
		return filepath
	}

	const nextDirUp = path.dirname(dir)
	const areWeThereYet = nextDirUp === dir

	if (areWeThereYet) {
		// Reached root
		return false
	}

	return findUpFile(nextDirUp, fileToFind)
}

const run = async (opts: any): Promise<any> => {
	const flags: Flags = await Promise.resolve(opts)
	splash(flags)

	const cwd = process.cwd()
	const readmePath = findUpFile(cwd, 'README.md')

	if (!readmePath) {
		console.log('No README.md found')
		return
	}

	const { dir } = path.parse(readmePath)
	flags.dir = dir
	flags.$pathProvided = true
	flags.$openLocation = 'README.md'

	return markserv.init(flags)
}

// Check if this module is being run directly
if (import.meta.url === `file://${process.argv[1]}`) {
	run(cliOpts)
}

export default { run }
