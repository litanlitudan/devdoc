// Test MLIR file for tensor shape extraction
module {
  func.func @test_shapes(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf32>) -> (tensor<2x4xf32>) {
    %0 = stablehlo.dot %arg0, %arg1 : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    %1 = stablehlo.add %0, %0 : tensor<2x4xf32>
    return %1 : tensor<2x4xf32>
  }
}