// Complex MLIR example with various tensor shapes
module @complex_shapes {
  func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf32>) -> tensor<2x4xf32> {
    // Matrix multiplication: [2x3] @ [3x4] = [2x4]
    %0 = "stablehlo.dot_general"(%arg0, %arg1) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_contracting_dimensions = [1],
        rhs_contracting_dimensions = [0]
      >
    } : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>

    // Add bias: [2x4] + [4] = [2x4] (broadcast)
    %1 = stablehlo.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
    %2 = "stablehlo.broadcast_in_dim"(%1) {
      broadcast_dimensions = array<i64: 1>
    } : (tensor<4xf32>) -> tensor<2x4xf32>
    %3 = stablehlo.add %0, %2 : tensor<2x4xf32>

    // ReLU activation
    %4 = stablehlo.constant dense<0.0> : tensor<2x4xf32>
    %5 = stablehlo.maximum %3, %4 : tensor<2x4xf32>

    return %5 : tensor<2x4xf32>
  }
}
