// Example MLIR file for testing Model Explorer visualization
module {
  func.func @add(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> tensor<10xf32> {
    %0 = arith.addf %arg0, %arg1 : tensor<10xf32>
    return %0 : tensor<10xf32>
  }

  func.func @multiply(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> tensor<10xf32> {
    %0 = arith.mulf %arg0, %arg1 : tensor<10xf32>
    return %0 : tensor<10xf32>
  }

  func.func @compute(%input: tensor<10xf32>) -> tensor<10xf32> {
    %c1 = arith.constant dense<1.0> : tensor<10xf32>
    %c2 = arith.constant dense<2.0> : tensor<10xf32>

    %0 = func.call @add(%input, %c1) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
    %1 = func.call @multiply(%0, %c2) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>

    return %1 : tensor<10xf32>
  }
}
