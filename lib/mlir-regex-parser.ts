import {
	EdgeData,
	EdgeOverlay,
	EdgeOverlaysData,
	GraphNode,
	KeyValue,
	MetadataItem,
	ModelExplorerGraph,
	ModelExplorerGraphs,
	TasksData,
} from './mlir-graph-types.js'

type OperationAttributes = Record<string, string>

interface MlirOperation {
	outputs: string[]
	opType: string
	inputs: string[]
	attributes: OperationAttributes
	resultTypes: string[]
	namespace?: string
}

interface MlirFunction {
	name: string
	inputs: Array<{ name: string; type: string }>
	outputs: string[]
	operations: MlirOperation[]
}

// Supports both {attrs} and <{attrs}> syntax, multiple attribute blocks, and location annotations
const QUOTED_WITH_RESULT =
	/(%[\w]+(?:,\s*%[\w]+)*)\s*=\s*"([^"]+)"\s*\(([^)]*)\)\s*((?:<?\{[^}]*\}>?\s*)*)\s*:\s*(?:\([^)]+\)\s*->\s*)?(.+?)(?:\s+loc\(|$)/gms
const UNQUOTED_WITH_RESULT =
	/(%[\w]+(?:,\s*%[\w]+)*)\s*=\s*([\w]+\.[\w]+)\s+(?!(?:\{[^}]+\}\s+)?ins\()([^{:]+?)\s*(?:\{([^}]*)\})?\s*:\s*(.+?)(?:\s|$)/gms
// Supports both {attrs} and <{attrs}> syntax, multiple attribute blocks, and location annotations
const QUOTED_NO_RESULT =
	/^\s*"([\w]+\.[\w]+)"\s*\(([^)]*)\)\s*((?:<?\{[^}]*\}>?\s*)*)\s*(?::\s*(.+?))?(?:\s+loc\(|$)/gm
const UNQUOTED_NO_RESULT =
	/^\s*([\w]+\.[\w]+)\s+([^{:\n]+?)\s*(?:\{([^}]*)\})?\s*(?::\s*(.+?))?(?:\s|$)/gm
const NO_INPUT_PATTERN =
	/(%[\w]+(?:,\s*%[\w]+)*)\s*=\s*([\w]+\.[\w]+)\s*\(\)\s*(?:\{([^}]*)\})?\s*:\s*(.+?)(?:\s|$)/gms
const CALL_NO_RESULT_PATTERN =
	/^\s*(call)\s+(@\w+)\s*\(([^)]*)\)\s*:\s*(.+?)(?:\s|$)/gm
const CALL_WITH_RESULT_PATTERN =
	/(%[\w]+(?:,\s*%[\w]+)*)\s*=\s*(call)\s+(@\w+)\s*\(([^)]*)\)\s*:\s*(.+?)(?:\s|$)/gms
const INS_OUTS_PATTERN =
	/(%[\w]+(?:,\s*%[\w]+)*)\s*=\s*([\w]+\.[\w]+)\s+(?:\{[^}]+\}\s+)?ins\(([^)]+)\)\s+outs\(([^)]+)\)\s*(?:->\s*(.+?))?(?:\s|$)/gms

// Function pattern - we'll handle argument parsing separately to handle nested parens
const FUNCTION_PATTERN = /(?:\w+\.func)\s+@(\w+)\s*\(/gm
// Input pattern - stops at loc(...), comma, paren, or end to strip location annotations
const INPUT_PATTERN = /(%\w+)\s*:\s*([^,)]+?)(?:\s+loc\(|,|\)|$)/g
const RETURN_PATTERN = /func\.return\s+([^:]+)/m

const ATTRIBUTE_PATTERN = /(\w+)\s*=\s*([^,}]+)/g

const DENSE_PATTERN =
	/(.*?dense)<([^>]+(?:>[^:]*?<[^>]+)*)>(.*?tensor<[^>]+>.*?)(?=\s*(?:\n|$|\/\/|%|\}))/gms

const UNKNOWN_DIALECT_PALETTE: Array<[string, string]> = [
	['#E3F2FD', '#0D47A1'],
	['#E8F5E9', '#1B5E20'],
	['#FFF3E0', '#E65100'],
	['#F3E5F5', '#4A148C'],
	['#FCE4EC', '#880E4F'],
	['#F1F8E9', '#33691E'],
	['#E0F7FA', '#006064'],
	['#FFF9C4', '#F57F17'],
	['#ECEFF1', '#37474F'],
]

const unknownDialectColorCache = new Map<string, [string, string]>()

function hashDialect(dialect: string): number {
	let hash = 0
	for (let i = 0; i < dialect.length; i += 1) {
		hash = (hash * 31 + dialect.charCodeAt(i)) & 0xff_ff_ff_ff
	}
	return hash >>> 0
}

function getUnknownDialectColor(dialect: string): [string, string] {
	const cached = unknownDialectColorCache.get(dialect)
	if (cached) return cached

	const index = hashDialect(dialect) % UNKNOWN_DIALECT_PALETTE.length
	const color = UNKNOWN_DIALECT_PALETTE[index]
	unknownDialectColorCache.set(dialect, color)
	return color
}

const DIALECT_COLORS: Record<string, [string, string]> = {
	// Core
	arith: ['#E3F2FD', '#0D47A1'],
	func: ['#F3E5F5', '#4A148C'],
	tensor: ['#E8F5E9', '#1B5E20'],
	linalg: ['#FFF3E0', '#E65100'],
	scf: ['#FCE4EC', '#880E4F'],
	memref: ['#E0F2F1', '#004D40'],
	vector: ['#F1F8E9', '#33691E'],
	affine: ['#FFF8E1', '#F57F17'],
	cf: ['#EFEBE9', '#3E2723'],
	gpu: ['#E1F5FE', '#01579B'],

	// ML / DL dialects
	tosa: ['#F9FBE7', '#827717'],
	stablehlo: ['#EDE7F6', '#311B92'],
	mhlo: ['#E8EAF6', '#1A237E'],
	tf: ['#FFEBEE', '#B71C1C'],
	tfl: ['#FFCDD2', '#C62828'],

	// Transformation
	transform: ['#FBE9E7', '#BF360C'],
	pdl: ['#E0F7FA', '#006064'],

	// Other
	llvm: ['#ECEFF1', '#263238'],
	spirv: ['#F3E5F5', '#6A1B9A'],
	async: ['#E1BEE7', '#6A1B9A'],
	math: ['#C5E1A5', '#558B2F'],
	call: ['#FFCCBC', '#D84315'],
}

function getDialectColor(opType: string): [string, string] {
	const dialect = opType.includes('.') ? opType.split('.')[0] : opType
	return DIALECT_COLORS[dialect] ?? getUnknownDialectColor(dialect)
}

function removeDenseConstantValues(mlirContent: string): {
	content: string
	replaced: number
} {
	let replacedCount = 0

	const replaced = mlirContent.replace(
		DENSE_PATTERN,
		(_, prefix: string, denseContent: string, suffix: string) => {
			replacedCount += 1

			const typeMatch = suffix.match(/tensor<([^>]+)>/)
			const tensorType = typeMatch ? typeMatch[1] : 'unknown'

			const sizeKb = denseContent.length / 1024
			const sizeInfo =
				sizeKb < 1024
					? `${sizeKb.toFixed(1)}KB`
					: `${(sizeKb / 1024).toFixed(1)}MB`

			return `${prefix}dense<0.0>${suffix}  // VALUES_REMOVED (${sizeInfo}, shape: ${tensorType})`
		},
	)

	return { content: replaced, replaced: replacedCount }
}

function splitCommaSeparated(input: string): string[] {
	return input
		.split(',')
		.map((part) => part.trim())
		.filter((part) => part.length > 0)
}

function parseAttributes(attrString: string | undefined): OperationAttributes {
	const attrs: OperationAttributes = {}
	if (!attrString) return attrs

	for (const match of attrString.matchAll(ATTRIBUTE_PATTERN)) {
		const key = match[1]
		const value = match[2].trim()
		attrs[key] = value
	}

	return attrs
}

function lastResultType(resultTypes: string[]): string[] {
	if (resultTypes.length === 0) return []
	return [resultTypes[resultTypes.length - 1]]
}

function parseResultTypes(resultTypeString: string | undefined): string[] {
	if (!resultTypeString) return []
	return splitCommaSeparated(resultTypeString.split('->').pop() ?? '')
}

function parseMlirOperations(mlirContent: string): MlirOperation[] {
	type MatchEntry = {
		index: number
		type: string
		match: RegExpMatchArray
	}

	const matches: MatchEntry[] = []

	for (const match of mlirContent.matchAll(QUOTED_WITH_RESULT)) {
		matches.push({ index: match.index ?? 0, type: 'quoted_result', match })
	}
	for (const match of mlirContent.matchAll(UNQUOTED_WITH_RESULT)) {
		matches.push({ index: match.index ?? 0, type: 'unquoted_result', match })
	}
	for (const match of mlirContent.matchAll(QUOTED_NO_RESULT)) {
		matches.push({ index: match.index ?? 0, type: 'quoted_no_result', match })
	}
	for (const match of mlirContent.matchAll(UNQUOTED_NO_RESULT)) {
		matches.push({ index: match.index ?? 0, type: 'unquoted_no_result', match })
	}
	for (const match of mlirContent.matchAll(NO_INPUT_PATTERN)) {
		matches.push({ index: match.index ?? 0, type: 'no_input', match })
	}
	for (const match of mlirContent.matchAll(CALL_NO_RESULT_PATTERN)) {
		matches.push({ index: match.index ?? 0, type: 'call_no_result', match })
	}
	for (const match of mlirContent.matchAll(CALL_WITH_RESULT_PATTERN)) {
		matches.push({ index: match.index ?? 0, type: 'call_with_result', match })
	}
	for (const match of mlirContent.matchAll(INS_OUTS_PATTERN)) {
		matches.push({ index: match.index ?? 0, type: 'ins_outs', match })
	}

	matches.sort((a, b) => a.index - b.index)

	const operations: MlirOperation[] = []

	for (const entry of matches) {
		const { match, type } = entry

		let outputs: string[] = []
		let opType = ''
		let inputs: string[] = []
		let attrs: OperationAttributes = {}
		let resultTypes: string[] = []

		switch (type) {
			case 'quoted_result': {
				outputs = splitCommaSeparated(match[1])
				opType = match[2]
				inputs = splitCommaSeparated(match[3])
				attrs = parseAttributes(match[4])
				resultTypes = lastResultType(parseResultTypes(match[5]))
				break
			}
			case 'unquoted_result': {
				outputs = splitCommaSeparated(match[1])
				opType = match[2]
				const rawInputs = splitCommaSeparated(match[3])
				inputs = rawInputs.filter(
					(value) => value.startsWith('%') || value.startsWith('@'),
				)
				if (inputs.length === 0 && match[3].trim().startsWith('%')) {
					inputs = [match[3].trim()]
				}
				attrs = parseAttributes(match[4])
				resultTypes = lastResultType(parseResultTypes(match[5]))
				break
			}
			case 'quoted_no_result': {
				opType = match[1]
				inputs = splitCommaSeparated(match[2])
				attrs = parseAttributes(match[3])
				resultTypes = lastResultType(parseResultTypes(match[4]))
				break
			}
			case 'unquoted_no_result': {
				opType = match[1]
				const rawInputs = splitCommaSeparated(match[2])
				inputs = rawInputs.filter(
					(value) => value.startsWith('%') || value.startsWith('@'),
				)
				if (inputs.length === 0 && match[2].trim().startsWith('%')) {
					inputs = [match[2].trim()]
				}
				attrs = parseAttributes(match[3])
				resultTypes = lastResultType(parseResultTypes(match[4]))
				break
			}
			case 'no_input': {
				outputs = splitCommaSeparated(match[1])
				opType = match[2]
				attrs = parseAttributes(match[3])
				resultTypes = lastResultType(parseResultTypes(match[4]))
				break
			}
			case 'call_no_result': {
				opType = match[1]
				const funcRef = match[2]
				inputs = [funcRef]
				const args = splitCommaSeparated(match[3])
				inputs.push(...args)
				resultTypes = lastResultType(parseResultTypes(match[4]))
				break
			}
			case 'call_with_result': {
				outputs = splitCommaSeparated(match[1])
				opType = match[2]
				const funcRef = match[3]
				inputs = [funcRef]
				const args = splitCommaSeparated(match[4])
				inputs.push(...args)
				resultTypes = lastResultType(parseResultTypes(match[5]))
				break
			}
			case 'ins_outs': {
				outputs = splitCommaSeparated(match[1])
				opType = match[2]
				const insStr = match[3]
				const outsStr = match[4]
				// Extract inputs from both ins() and outs()
				const insInputs = insStr.match(/%[\w]+/g) ?? []
				const outsInputs = outsStr.match(/%[\w]+/g) ?? []
				inputs = [...insInputs, ...outsInputs]
				// Use result type from -> clause if present, otherwise parse from outs clause
				const resultTypeStr = match[5]
				if (resultTypeStr) {
					resultTypes = lastResultType(parseResultTypes(resultTypeStr))
				} else if (outsStr.includes(':')) {
					// Parse types from outs clause, removing trailing parenthesis
					const typesPart = outsStr.split(':')[1].trim().replace(/\)$/, '')
					resultTypes = splitCommaSeparated(typesPart)
				}
				break
			}
			default:
				continue
		}

		operations.push({
			outputs,
			opType,
			inputs,
			attributes: attrs,
			resultTypes,
		})
	}

	return operations
}

function assignNamespacesToOperations(
	mlirContent: string,
	operations: MlirOperation[],
	baseNamespace: string,
) {
	const valueToOp = new Map<string, MlirOperation>()
	for (const op of operations) {
		for (const output of op.outputs) {
			valueToOp.set(output, op)
		}
	}

	const unassigned = new Set(operations)
	const namespaceStack = [baseNamespace]
	let depth = 0
	const operationCounter: Record<string, number> = {}

	const lines = mlirContent.split('\n')
	for (const line of lines) {
		const prevDepth = depth
		const openBraces = (line.match(/{/g) ?? []).length
		const closeBraces = (line.match(/}/g) ?? []).length
		const depthChange = openBraces - closeBraces

		let matchedOp: MlirOperation | undefined
		let opType: string | undefined

		const assignmentMatch = line.match(
			/\s*(%[\w]+(?:,\s*%[\w]+)*)\s*=\s*([\w.]+)/,
		)
		if (assignmentMatch) {
			const firstOutput = assignmentMatch[1].split(',')[0].trim()
			opType = assignmentMatch[2]
			matchedOp = valueToOp.get(firstOutput)
		} else {
			const opMatch = line.match(/\b([\w]+\.[\w]+)\b/)
			if (opMatch) {
				opType = opMatch[1]
				for (const op of unassigned) {
					if (op.opType === opType && !op.namespace) {
						matchedOp = op
						break
					}
				}
			}
		}

		if (matchedOp) {
			const currentNamespace = namespaceStack[namespaceStack.length - 1]
			matchedOp.namespace = currentNamespace
			unassigned.delete(matchedOp)

			if (depthChange > 0 && opType) {
				const cleanType = opType.replace('.', '_')
				operationCounter[cleanType] = (operationCounter[cleanType] ?? 0) + 1
				const layerId = `${cleanType}_${operationCounter[cleanType]}`
				const newNamespace = currentNamespace
					? `${currentNamespace}/${layerId}`
					: layerId
				namespaceStack.push(newNamespace)
			}
		}

		depth += depthChange

		if (depth < prevDepth && namespaceStack.length > 1) {
			const popsNeeded = Math.min(prevDepth - depth, namespaceStack.length - 1)
			for (let i = 0; i < popsNeeded; i += 1) {
				namespaceStack.pop()
			}
		}
	}

	for (const op of operations) {
		if (!op.namespace) {
			op.namespace = baseNamespace
		}
	}
}

function parseFunctionOutputs(body: string): string[] {
	const match = body.match(RETURN_PATTERN)
	if (!match) return []

	return splitCommaSeparated(match[1])
}

function parseFunctions(content: string): MlirFunction[] {
	const functions: MlirFunction[] = []

	for (const match of content.matchAll(FUNCTION_PATTERN)) {
		const funcName = match[1]

		// Manually extract inputs by counting parentheses (handles nested loc(...))
		const parenStart = match.index! + match[0].length - 1 // Position of opening (
		let parenCount = 1
		let pos = match.index! + match[0].length

		while (pos < content.length && parenCount > 0) {
			if (content[pos] === '(') parenCount += 1
			else if (content[pos] === ')') parenCount -= 1
			pos += 1
		}

		const inputsStr =
			parenCount === 0 ? content.slice(parenStart + 1, pos - 1) : ''

		// Find the function body start by looking for { before first operation
		// This handles both "func @name() {" and "func @name() attributes {...} {"
		const searchStart = pos
		const remaining = content.slice(searchStart)

		// Look for the pattern: possible attributes block(s) followed by {
		const bodyStartMatch = remaining.match(/\{\s*(?=%|\s*func\.return|\s*$)/)
		const bodyStartOffset = bodyStartMatch
			? bodyStartMatch.index! + 1
			: remaining.indexOf('{') + 1

		if (bodyStartOffset === 0) continue // No function body found

		const bodyStart = searchStart + bodyStartOffset

		// Find the corresponding closing brace
		let braceCount = 1
		pos = bodyStart
		let bodyEnd = content.length

		while (pos < content.length && braceCount > 0) {
			if (content[pos] === '{') braceCount += 1
			else if (content[pos] === '}') {
				braceCount -= 1
				if (braceCount === 0) {
					bodyEnd = pos
					break
				}
			}
			pos += 1
		}

		const body = content.slice(bodyStart, bodyEnd)
		const inputs: Array<{ name: string; type: string }> = []

		if (inputsStr.trim()) {
			for (const inputMatch of inputsStr.matchAll(INPUT_PATTERN)) {
				const name = inputMatch[1]
				const type = inputMatch[2].trim()
				inputs.push({ name, type })
			}
		}

		const operations = parseMlirOperations(body)
		assignNamespacesToOperations(body, operations, funcName)

		functions.push({
			name: funcName,
			inputs,
			outputs: parseFunctionOutputs(body),
			operations,
		})
	}

	if (functions.length === 0) {
		const operations = parseMlirOperations(content)
		assignNamespacesToOperations(content, operations, 'main')

		const inputs: Array<{ name: string; type: string }> = []
		const funcMatch = content.match(/func\.func\s+@\w+\s*\(([^)]+)\)/m)
		if (funcMatch) {
			for (const inputMatch of funcMatch[1].matchAll(INPUT_PATTERN)) {
				inputs.push({
					name: inputMatch[1],
					type: inputMatch[2].trim(),
				})
			}
		}

		functions.push({
			name: 'main',
			inputs,
			outputs: parseFunctionOutputs(content),
			operations,
		})
	}

	return functions
}

function detectFunctionCalls(
	operations: MlirOperation[],
	functionName: string,
): Record<string, string[]> {
	const calls: Record<string, string[]> = {}
	operations.forEach((op, index) => {
		if (!op.opType.toLowerCase().includes('call')) {
			return
		}
		const called: string[] = []
		for (const input of op.inputs) {
			if (input.startsWith('@')) {
				called.push(input.slice(1))
			}
		}
		if (called.length > 0) {
			calls[`${functionName}_op_${index}`] = called
		}
	})
	return calls
}

function buildMetadataItem(id: string, attrs: KeyValue[]): MetadataItem {
	return { id, attrs }
}

function appendMetadataAttr(
	metadata: MetadataItem,
	key: string,
	value: string,
) {
	metadata.attrs.push({ key, value })
}

function createGraphForFunction(
	func: MlirFunction,
	availableFunctions: string[],
): ModelExplorerGraph {
	const nodes: GraphNode[] = []
	const valueToProducer = new Map<
		string,
		{ nodeId: string; outputId: string }
	>()

	func.inputs.forEach((input, idx) => {
		const nodeId = `${func.name}_input_${idx}`
		const metadata: MetadataItem = buildMetadataItem('0', [
			{ key: '__tensor_tag', value: input.name },
			{ key: 'tensor_shape', value: input.type },
		])

		const node: GraphNode = {
			id: nodeId,
			label: 'Input',
			namespace: `${func.name}/Inputs`,
			attrs: [
				{ key: 'name', value: input.name },
				{ key: 'index', value: String(idx) },
			],
			incomingEdges: [],
			outputsMetadata: [metadata],
		}

		nodes.push(node)
		valueToProducer.set(input.name, { nodeId, outputId: '0' })
	})

	const functionCalls = detectFunctionCalls(func.operations, func.name)

	func.operations.forEach((op, idx) => {
		const nodeId = `${func.name}_op_${idx}`
		const [backgroundColor, textColor] = getDialectColor(op.opType)

		const node: GraphNode = {
			id: nodeId,
			label: op.opType,
			namespace: op.namespace ?? func.name,
			attrs: [],
			incomingEdges: [],
			style: {
				backgroundColor,
				textColor,
			},
		}

		for (const [key, value] of Object.entries(op.attributes)) {
			node.attrs.push({ key, value })
		}

		const called = functionCalls[nodeId]
		if (called) {
			node.subgraphIds = called.filter((callee) =>
				availableFunctions.includes(callee),
			)
		}

		op.inputs.forEach((input, inputIdx) => {
			const producer = valueToProducer.get(input)
			if (!producer) return

			node.incomingEdges.push({
				sourceNodeId: producer.nodeId,
				sourceNodeOutputId: producer.outputId,
				targetNodeInputId: String(inputIdx),
			})

			if (!node.inputsMetadata) node.inputsMetadata = []
			const metadata = buildMetadataItem(String(inputIdx), [
				{ key: '__tensor_tag', value: input },
			])
			node.inputsMetadata.push(metadata)
		})

		if (op.outputs.length > 0) {
			node.outputsMetadata = []
			op.outputs.forEach((output, outputIdx) => {
				const metadata = buildMetadataItem(String(outputIdx), [
					{ key: '__tensor_tag', value: output },
				])
				if (op.resultTypes[outputIdx]) {
					appendMetadataAttr(
						metadata,
						'tensor_shape',
						op.resultTypes[outputIdx],
					)
				}
				node.outputsMetadata!.push(metadata)
				valueToProducer.set(output, {
					nodeId,
					outputId: String(outputIdx),
				})
			})
		}

		nodes.push(node)
	})

	func.outputs.forEach((output, idx) => {
		const nodeId = `${func.name}_output_${idx}`
		const node: GraphNode = {
			id: nodeId,
			label: 'Output',
			namespace: `${func.name}/Outputs`,
			attrs: [
				{ key: 'name', value: output },
				{ key: 'index', value: String(idx) },
			],
			incomingEdges: [],
		}

		const producer = valueToProducer.get(output)
		if (producer) {
			node.incomingEdges.push({
				sourceNodeId: producer.nodeId,
				sourceNodeOutputId: producer.outputId,
				targetNodeInputId: '0',
			})
		}

		nodes.push(node)
	})

	const graph: ModelExplorerGraph = {
		id: func.name,
		nodes,
		tasksData: generateEdgeOverlaysForGraph(nodes, func.name),
	}

	return graph
}

function generateEdgeOverlaysForGraph(
	nodes: GraphNode[],
	functionName: string,
): TasksData {
	const tensorEdges: EdgeData[] = []
	const scalarEdges: EdgeData[] = []
	const allEdges: EdgeData[] = []

	const nodeById = new Map(nodes.map((node) => [node.id, node]))

	nodes.forEach((node: GraphNode) => {
		node.incomingEdges.forEach((edge: GraphNode['incomingEdges'][number]) => {
			const sourceNode = nodeById.get(edge.sourceNodeId)
			if (!sourceNode?.outputsMetadata) return

			const outputMeta = sourceNode.outputsMetadata.find(
				(meta: MetadataItem) => meta.id === edge.sourceNodeOutputId,
			)
			if (!outputMeta) return

			let tensorShape = ''
			let tensorTag = ''

			outputMeta.attrs.forEach((attr: KeyValue) => {
				if (attr.key === 'tensor_shape') tensorShape = attr.value
				else if (attr.key === '__tensor_tag') tensorTag = attr.value
			})

			const label = tensorShape || tensorTag
			if (!label) return

			const overlayEdge: EdgeData = {
				sourceNodeId: edge.sourceNodeId,
				targetNodeId: node.id,
				label,
			}

			allEdges.push(overlayEdge)
			if (label.toLowerCase().includes('tensor<')) tensorEdges.push(overlayEdge)
			else scalarEdges.push(overlayEdge)
		})
	})

	const overlays: EdgeOverlay[] = []
	if (allEdges.length > 0) {
		overlays.push({
			name: 'Tensor Shapes',
			edges: allEdges,
			edgeColor: '#4285f4',
			edgeWidth: 2,
			edgeLabelFontSize: 7.5,
		})
	}
	if (tensorEdges.length > 0) {
		overlays.push({
			name: 'Tensor Data Flow',
			edges: tensorEdges,
			edgeColor: '#34a853',
			edgeWidth: 3,
			edgeLabelFontSize: 8,
		})
	}
	if (scalarEdges.length > 0) {
		overlays.push({
			name: 'Scalar Values',
			edges: scalarEdges,
			edgeColor: '#fbbc04',
			edgeWidth: 2,
			edgeLabelFontSize: 7,
		})
	}

	const tasksData: TasksData = {}
	if (overlays.length > 0) {
		const overlayData: EdgeOverlaysData = {
			type: 'EDGE_OVERLAYS',
			name: `Tensor Flow - ${functionName}`,
			overlays,
		}
		tasksData.edgeOverlaysDataListLeftPane = [overlayData]
	}

	return tasksData
}

export function parseMlirWithRegex(
	mlirContent: string,
	filename: string,
): ModelExplorerGraphs {
	if (!mlirContent.trim()) {
		return {
			graphs: [],
			_metadata: {
				parser: 'regex-ts',
				error: 'Empty input',
			},
		}
	}

	const { content: preprocessed, replaced } =
		removeDenseConstantValues(mlirContent)
	const functions = parseFunctions(preprocessed)

	const availableFunctions = functions.map((fn) => fn.name)
	const graphs = functions.map((fn) =>
		createGraphForFunction(fn, availableFunctions),
	)

	return {
		graphs,
		_metadata: {
			parser: 'regex-ts',
			source: filename,
			functions_parsed: graphs.length,
			preprocessing: {
				constants_removed: replaced,
			},
		},
	}
}
