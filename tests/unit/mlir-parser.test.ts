/**
 * Comprehensive tests for MLIR parser
 * Validates all operation patterns and graph generation
 */

import { describe, it, expect } from 'vitest'
import { execFileSync } from 'child_process'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

// Helper to parse MLIR using the Python parser (returns single graph from multi-graph format)
function parseMLIR(mlirContent: string, filename: string = 'test-graph') {
	const scriptPath = path.join(
		__dirname,
		'..',
		'..',
		'scripts',
		'parse_mlir_regex.py',
	)
	const pythonCmd = process.env.CONDA_DEFAULT_ENV ? 'python' : 'python3'

	const resultJson = execFileSync(pythonCmd, [scriptPath, filename], {
		input: mlirContent,
		encoding: 'utf-8',
		maxBuffer: 10 * 1024 * 1024,
	})

	const result = JSON.parse(resultJson)
	// Return the first graph for backward compatibility with tests
	return result.graphs && result.graphs[0] ? result.graphs[0] : result
}

describe('MLIR Parser - Operation Patterns', () => {
	it('should parse operations with results and inputs', () => {
		const mlir = `
module {
  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %0 = "custom.op"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
    func.return %0 : tensor<2x3xf32>
  }
}
`
		const result = parseMLIR(mlir)

		expect(result.id).toBe('main') // Graph ID is now the function name
		expect(result.nodes).toBeInstanceOf(Array)

		// Should have: Input, func.func, custom.op, func.return, Output
		expect(result.nodes.length).toBeGreaterThan(3)

		// Check input node
		const inputNode = result.nodes.find((n: any) => n.label === 'Input')
		expect(inputNode).toBeDefined()
		expect(inputNode.namespace).toBe('main/Inputs') // Namespace is function-scoped

		// Check operation node
		const opNode = result.nodes.find((n: any) => n.label === 'custom.op')
		expect(opNode).toBeDefined()
		expect(opNode.incomingEdges.length).toBeGreaterThan(0)
	})

	it('should parse operations without results', () => {
		const mlir = `
module {
  func.func @test(%arg0: tensor<2x2xf32>) {
    call @helper(%arg0) : (tensor<2x2xf32>) -> ()
    func.return
  }
}
`
		const result = parseMLIR(mlir)

		// Check that func.return is parsed (no outputs)
		const returnNode = result.nodes.find((n: any) => n.label === 'func.return')
		expect(returnNode).toBeDefined()
		expect(returnNode.outputsMetadata || []).toHaveLength(0)
	})

	it('should parse operations without inputs', () => {
		const mlir = `
module {
  func.func @constants() -> tensor<2x2xf32> {
    %cst = arith.constant dense<1.0> : tensor<2x2xf32>
    func.return %cst : tensor<2x2xf32>
  }
}
`
		const result = parseMLIR(mlir)

		// Check constant operation (no inputs)
		const constNode = result.nodes.find(
			(n: any) => n.label === 'arith.constant',
		)
		expect(constNode).toBeDefined()
		expect(constNode.incomingEdges).toHaveLength(0)
		expect(constNode.outputsMetadata.length).toBeGreaterThan(0)
	})

	it('should handle custom dialects', () => {
		const mlir = `
module {
  func.func @custom_ops(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = mycompany.custom_transform %arg0 {scale = 2.0 : f32} : tensor<2x2xf32>
    func.return %0 : tensor<2x2xf32>
  }
}
`
		const result = parseMLIR(mlir)

		const customOp = result.nodes.find(
			(n: any) => n.label === 'mycompany.custom_transform',
		)
		expect(customOp).toBeDefined()
		expect(customOp.attrs.length).toBeGreaterThan(0)

		// Check attribute
		const scaleAttr = customOp.attrs.find((a: any) => a.key === 'scale')
		expect(scaleAttr).toBeDefined()
	})

	it('should handle multiple inputs and outputs', () => {
		const mlir = `
module {
  func.func @multi_io(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>) {
    %0 = arith.addf %a, %b : tensor<2x2xf32>
    %1 = arith.mulf %a, %b : tensor<2x2xf32>
    func.return %0, %1 : tensor<2x2xf32>, tensor<2x2xf32>
  }
}
`
		const result = parseMLIR(mlir)

		// Should have 2 inputs
		const inputNodes = result.nodes.filter((n: any) => n.label === 'Input')
		expect(inputNodes.length).toBe(2)

		// Should have 2 outputs
		const outputNodes = result.nodes.filter((n: any) => n.label === 'Output')
		expect(outputNodes.length).toBe(2)
	})
})

describe('MLIR Parser - Graph Structure', () => {
	it('should create proper edges between nodes', () => {
		const mlir = `
module {
  func.func @chain(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = "op.a"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %1 = "op.b"(%0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %2 = "op.c"(%1) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    func.return %2 : tensor<2x2xf32>
  }
}
`
		const result = parseMLIR(mlir)

		const opA = result.nodes.find((n: any) => n.label === 'op.a')
		const opB = result.nodes.find((n: any) => n.label === 'op.b')
		const opC = result.nodes.find((n: any) => n.label === 'op.c')

		// Verify chain: op.a <- op.b <- op.c
		expect(opB.incomingEdges.length).toBe(1)
		expect(opB.incomingEdges[0].sourceNodeId).toBe(opA.id)

		expect(opC.incomingEdges.length).toBe(1)
		expect(opC.incomingEdges[0].sourceNodeId).toBe(opB.id)
	})

	it('should include tensor shape metadata', () => {
		const mlir = `
module {
  func.func @shapes(%arg0: tensor<128x256xf32>) -> tensor<256x512xf16> {
    %0 = "custom.reshape"(%arg0) : (tensor<128x256xf32>) -> tensor<256x512xf16>
    func.return %0 : tensor<256x512xf16>
  }
}
`
		const result = parseMLIR(mlir)

		const inputNode = result.nodes.find((n: any) => n.label === 'Input')
		expect(inputNode.outputsMetadata[0].attrs).toContainEqual({
			key: 'tensor_shape',
			value: 'tensor<128x256xf32>',
		})

		const reshapeNode = result.nodes.find(
			(n: any) => n.label === 'custom.reshape',
		)
		expect(reshapeNode.outputsMetadata[0].attrs).toContainEqual({
			key: 'tensor_shape',
			value: 'tensor<256x512xf16>',
		})
	})
})

describe('MLIR Parser - Real World Fixtures', () => {
	it('should parse custom dialect sample', () => {
		const fs = require('fs')
		const samplePath = path.join(__dirname, '..', 'sample-custom-dialect.mlir')

		if (!fs.existsSync(samplePath)) {
			console.warn('Sample file not found, skipping test')
			return
		}

		const mlirContent = fs.readFileSync(samplePath, 'utf-8')
		const result = parseMLIR(mlirContent, 'custom-dialect-test')

		expect(result.id).toBe('main') // Graph ID is now the function name
		expect(result.nodes.length).toBeGreaterThan(5)

		// Should contain custom dialects
		const hasCustom = result.nodes.some(
			(n: any) =>
				n.label.includes('custom') ||
				n.label.includes('mycompany') ||
				n.label.includes('acme'),
		)
		expect(hasCustom).toBe(true)
	})
})

describe('MLIR Parser - Error Handling', () => {
	it('should handle empty input', () => {
		expect(() => parseMLIR('')).toThrow()
	})

	it('should handle malformed MLIR', () => {
		const malformed = 'this is not valid MLIR'
		const result = parseMLIR(malformed)

		// Should return empty graph or minimal structure
		expect(result.nodes).toBeInstanceOf(Array)
	})
})

describe('MLIR Parser - Multi-Graph Format', () => {
	// Helper to parse MLIR with new multi-graph format
	function parseMLIRGraphs(
		mlirContent: string,
		filename: string = 'test-graphs',
	) {
		const scriptPath = path.join(
			__dirname,
			'..',
			'..',
			'scripts',
			'parse_mlir_regex.py',
		)
		const pythonCmd = process.env.CONDA_DEFAULT_ENV ? 'python' : 'python3'

		const resultJson = execFileSync(pythonCmd, [scriptPath, filename], {
			input: mlirContent,
			encoding: 'utf-8',
			maxBuffer: 10 * 1024 * 1024,
		})

		return JSON.parse(resultJson)
	}

	it('should return graphs array for multi-function MLIR', () => {
		const mlir = `
module {
  func.func @helper(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = "math.sqrt"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    func.return %0 : tensor<2x2xf32>
  }

  func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = call @helper(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    func.return %0 : tensor<2x2xf32>
  }
}
`
		const result = parseMLIRGraphs(mlir)

		expect(result.graphs).toBeDefined()
		expect(result.graphs).toBeInstanceOf(Array)
		expect(result.graphs.length).toBe(2)

		// Check graph IDs
		const graphIds = result.graphs.map((g: any) => g.id)
		expect(graphIds).toContain('helper')
		expect(graphIds).toContain('main')
	})

	it('should separate operations by function', () => {
		const mlir = `
module {
  func.func @func_a(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = arith.addf %arg0, %arg0 : tensor<2x2xf32>
    func.return %0 : tensor<2x2xf32>
  }

  func.func @func_b(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = arith.mulf %arg0, %arg0 : tensor<2x2xf32>
    func.return %0 : tensor<2x2xf32>
  }
}
`
		const result = parseMLIRGraphs(mlir)

		const funcAGraph = result.graphs.find((g: any) => g.id === 'func_a')
		const funcBGraph = result.graphs.find((g: any) => g.id === 'func_b')

		expect(funcAGraph).toBeDefined()
		expect(funcBGraph).toBeDefined()

		// Check that func_a has addf but not mulf
		const funcAOps = funcAGraph.nodes.map((n: any) => n.label)
		expect(funcAOps).toContain('arith.addf')
		expect(funcAOps).not.toContain('arith.mulf')

		// Check that func_b has mulf but not addf
		const funcBOps = funcBGraph.nodes.map((n: any) => n.label)
		expect(funcBOps).toContain('arith.mulf')
		expect(funcBOps).not.toContain('arith.addf')
	})

	it('should apply namespace scoping per function', () => {
		const mlir = `
module {
  func.func @my_function(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = arith.constant dense<1.0> : tensor<2x2xf32>
    func.return %0 : tensor<2x2xf32>
  }
}
`
		const result = parseMLIRGraphs(mlir)
		const graph = result.graphs[0]

		// Check that input nodes have function-scoped namespace
		const inputNodes = graph.nodes.filter((n: any) => n.label === 'Input')
		expect(inputNodes[0].namespace).toBe('my_function/Inputs')

		// Check that operation nodes have function-scoped namespace
		const constNode = graph.nodes.find((n: any) => n.label === 'arith.constant')
		expect(constNode.namespace).toBe('my_function')
	})

	it('should populate subgraphIds for function calls', () => {
		const mlir = `
module {
  func.func @callee(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = arith.addf %arg0, %arg0 : tensor<2x2xf32>
    func.return %0 : tensor<2x2xf32>
  }

  func.func @caller(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = call @callee(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    func.return %0 : tensor<2x2xf32>
  }
}
`
		const result = parseMLIRGraphs(mlir)
		const callerGraph = result.graphs.find((g: any) => g.id === 'caller')

		// Find call operation
		const callNode = callerGraph.nodes.find((n: any) => n.label === 'call')
		expect(callNode).toBeDefined()
		expect(callNode.subgraphIds).toBeDefined()
		expect(callNode.subgraphIds).toContain('callee')
	})

	it('should handle functions without inputs or outputs', () => {
		const mlir = `
module {
  func.func @void_function() {
    %0 = arith.constant dense<1.0> : tensor<2x2xf32>
    func.return
  }
}
`
		const result = parseMLIRGraphs(mlir)
		const graph = result.graphs[0]

		// Should have no input nodes
		const inputNodes = graph.nodes.filter((n: any) => n.label === 'Input')
		expect(inputNodes.length).toBe(0)

		// Should have no output nodes
		const outputNodes = graph.nodes.filter((n: any) => n.label === 'Output')
		expect(outputNodes.length).toBe(0)
	})
})
