import chalk from 'chalk'

interface Flags {
	silent?: boolean
	[key: string]: any
}

const msg = (type: string, msg: string): void => {
	console.log(
		chalk.bgGreen.black('  Markserv  ') + chalk.white(` ${type}: `) + msg,
	)
}

const splash = (flags?: Flags): void => {
	if (flags && flags.silent) {
		return
	}

	// Display ASCII art logo instead of image
	console.log(
		chalk.green(`
  ┌─┐┌─┐┌─┐┬  ┬┌─┐┌─┐┬  ┬
  │││├─┤├┬┘├┐┌┘└─┐├┤ ├┬┘└┐┌┘
  ┴ ┴┴ ┴┴└─┴ └ └─┘└─┘┴└─ └┘
  `),
	)

	msg('boot', 'starting Markserv...')
}

export default splash
