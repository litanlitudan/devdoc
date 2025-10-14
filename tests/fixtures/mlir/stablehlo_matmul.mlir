// Matrix multiplication example with 2D tensors
module @matmul {
  func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf32>) -> tensor<2x4xf32> {
    %0 = stablehlo.dot %arg0, %arg1 : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
}
