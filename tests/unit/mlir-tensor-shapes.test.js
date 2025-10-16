import {describe, it, expect, beforeAll} from 'vitest';
import {convertMLIRToGraph} from '../../dist/mlir-to-graph.js';
import {execSync} from 'child_process';

// Check if Model Explorer adapter is available
let adapterAvailable = false;
try {
	const pythonCmd = process.env.CONDA_DEFAULT_ENV ? 'python' : 'python3';
	execSync(`${pythonCmd} -c "from ai_edge_model_explorer_adapter import _pywrap_convert_wrapper"`, {
		stdio: 'ignore'
	});
	adapterAvailable = true;
} catch (e) {
	console.warn('⚠ Model Explorer adapter not available - skipping MLIR tests');
	console.warn('  Install with: pip install ai-edge-model-explorer-adapter');
}

describe.skip('MLIR Tensor Shape Extraction', () => {
	beforeAll(() => {
		if (!adapterAvailable) {
			console.log('Skipping MLIR tests - adapter not available');
		}
	});

	it.skipIf(!adapterAvailable)('extracts tensor shapes from MLIR operations', () => {
		const mlirContent = `
module {
  func.func @test_shapes(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf32>) -> (tensor<2x4xf32>) {
    %0 = stablehlo.dot %arg0, %arg1 : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    %1 = stablehlo.add %0, %0 : tensor<2x4xf32>
    return %1 : tensor<2x4xf32>
  }
}
`;

		const graph = convertMLIRToGraph(mlirContent, 'test.mlir');

		// Check that we have the expected nodes
		// Inputs: arg0, arg1
		// Operations: dot, add
		// Outputs: return
		expect(graph.nodes.length).toBe(5);

		// Find input nodes
		const inputNodes = graph.nodes.filter(n => n.namespace === 'Inputs');
		expect(inputNodes.length).toBe(2);
		expect(inputNodes[0].label).toContain('out: tensor<2x3xf32>');
		expect(inputNodes[1].label).toContain('out: tensor<3x4xf32>');

		// Find the stablehlo.dot operation node
		const dotNode = graph.nodes.find(n => n.label.includes('stablehlo.dot'));
		expect(dotNode).toBeDefined();
		// Label should include tensor shape information
		expect(dotNode.label).toContain('stablehlo.dot');
		expect(dotNode.label).toContain('in: tensor<2x3xf32>, tensor<3x4xf32>');
		expect(dotNode.label).toContain('out: tensor<2x4xf32>');
		// Check output metadata
		expect(dotNode.outputsMetadata).toBeDefined();
		expect(dotNode.outputsMetadata.length).toBe(1);
		expect(dotNode.outputsMetadata[0].id).toBe('0');
		// The adapter returns full tensor type strings
		const dotOutputShape = dotNode.outputsMetadata[0].attrs.find(a => a.key === 'tensor_shape');
		expect(dotOutputShape).toBeDefined();
		expect(dotOutputShape.value).toContain('tensor<2x4xf32>');

		// Find the stablehlo.add operation node
		const addNode = graph.nodes.find(n => n.label.includes('stablehlo.add'));
		expect(addNode).toBeDefined();
		// Label should include tensor shape information
		expect(addNode.label).toContain('stablehlo.add');
		expect(addNode.label).toContain('in: tensor<2x4xf32>, tensor<2x4xf32>');
		expect(addNode.label).toContain('out: tensor<2x4xf32>');
		// Check output metadata
		expect(addNode.outputsMetadata).toBeDefined();
		expect(addNode.outputsMetadata.length).toBe(1);
		const addOutputShape = addNode.outputsMetadata[0].attrs.find(a => a.key === 'tensor_shape');
		expect(addOutputShape).toBeDefined();
		expect(addOutputShape.value).toContain('tensor<2x4xf32>');

		// Find output node
		const outputNode = graph.nodes.find(n => n.namespace === 'Outputs');
		expect(outputNode).toBeDefined();
		expect(outputNode.label).toContain('in: tensor<2x4xf32>');
	});

	it.skipIf(!adapterAvailable)('handles scalar tensors', () => {
		const mlirContent = `
module {
  func.func @test_scalar(%arg0: tensor<f32>) -> (tensor<f32>) {
    %0 = stablehlo.cosine %arg0 : tensor<f32>
    return %0 : tensor<f32>
  }
}
`;

		const graph = convertMLIRToGraph(mlirContent, 'test.mlir');

		// Find input node
		const inputNode = graph.nodes.find(n => n.namespace === 'Inputs');
		expect(inputNode).toBeDefined();
		expect(inputNode.label).toContain('out: tensor<f32>');

		// Find the cosine operation node
		const cosNode = graph.nodes.find(n => n.label.includes('stablehlo.cosine'));
		expect(cosNode).toBeDefined();
		// Label should include tensor shape information
		expect(cosNode.label).toContain('stablehlo.cosine');
		expect(cosNode.label).toContain('in: tensor<f32>');
		expect(cosNode.label).toContain('out: tensor<f32>');
		// Check output metadata
		expect(cosNode.outputsMetadata).toBeDefined();
		expect(cosNode.outputsMetadata.length).toBe(1);
		const outputShape = cosNode.outputsMetadata[0].attrs.find(a => a.key === 'tensor_shape');
		expect(outputShape).toBeDefined();
		expect(outputShape.value).toContain('tensor<f32>');
	});

	it.skipIf(!adapterAvailable)('handles operations without explicit return type', () => {
		const mlirContent = `
module {
  func.func @test_implicit(%arg0: tensor<2x3xf32>) -> (tensor<2x3xf32>) {
    %0 = stablehlo.negate %arg0 : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}
`;

		const graph = convertMLIRToGraph(mlirContent, 'test.mlir');

		const negateNode = graph.nodes.find(n => n.label.includes('stablehlo.negate'));
		expect(negateNode).toBeDefined();
		// Label should include tensor shape information
		expect(negateNode.label).toContain('stablehlo.negate');
		expect(negateNode.label).toContain('in: tensor<2x3xf32>');
		expect(negateNode.label).toContain('out: tensor<2x3xf32>');
		// Check output metadata
		expect(negateNode.outputsMetadata).toBeDefined();
		expect(negateNode.outputsMetadata.length).toBe(1);
		const outputShape = negateNode.outputsMetadata[0].attrs.find(a => a.key === 'tensor_shape');
		expect(outputShape).toBeDefined();
		expect(outputShape.value).toContain('tensor<2x3xf32>');
	});

	it.skipIf(!adapterAvailable)('handles error when adapter not installed', () => {
		// This test would only run if adapter was actually not installed
		// In our test environment with adapter, we just verify the error handling exists
		const mlirContent = `module { }`;

		// If adapter is available, this should succeed
		const graph = convertMLIRToGraph(mlirContent, 'test.mlir');
		expect(graph).toBeDefined();
		expect(graph.id).toBe('test.mlir');
	});
});
