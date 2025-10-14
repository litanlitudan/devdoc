interface Flag {
	alias: string
	default: string | number | boolean
}

interface CliDefs {
	flags: {
		port: Flag
		livereloadport: Flag
		address: Flag
		silent: Flag
		verbose: Flag
	}
}

const cliDefs: CliDefs = {
	flags: {
		port: {
			alias: 'p',
			default: '8642',
		},

		livereloadport: {
			alias: 'b',
			default: 35729,
		},

		address: {
			alias: 'a',
			default: 'localhost',
		},

		silent: {
			alias: 's',
			default: false,
		},

		verbose: {
			alias: 'v',
			default: false,
		},
	},
}

export default cliDefs
