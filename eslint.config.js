import js from '@eslint/js'
import prettier from 'eslint-config-prettier'

export default [
	js.configs.recommended,
	prettier,
	{
		files: ['**/*.js'],
		languageOptions: {
			ecmaVersion: 2024,
			sourceType: 'module',
			globals: {
				console: 'readonly',
				process: 'readonly',
				Buffer: 'readonly',
				__dirname: 'readonly',
				__filename: 'readonly',
				URL: 'readonly',
			},
		},
		rules: {
			'no-unused-vars': [
				'error',
				{
					argsIgnorePattern: '^_',
					varsIgnorePattern: '^_',
				},
			],
			'no-console': 'off',
			'prefer-const': 'error',
			'no-var': 'error',
			'object-shorthand': 'error',
			'prefer-arrow-callback': 'error',
			'prefer-template': 'error',
			'prefer-destructuring': ['error', { object: true, array: false }],
			'no-param-reassign': 'error',
			'no-return-await': 'error',
			'require-await': 'error',
		},
	},
	{
		ignores: ['node_modules/', 'coverage/', 'dist/', '*.min.js', 'tests/**/*.expected.html'],
	},
]