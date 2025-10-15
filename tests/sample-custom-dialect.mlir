// Test MLIR file with custom/unknown dialects
// This tests the generic dialect parsing capability

module {
  // Standard MLIR operations (known dialect)
  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
    // Custom dialect operations (unknown dialects)
    %0 = "custom.preprocessing"(%arg0) {mode = "normalize"} : (tensor<2x3xf32>) -> tensor<2x3xf32>

    // Another custom dialect
    %1 = mycompany.transform %0 {scale = 2.0 : f32} : tensor<2x3xf32> -> tensor<2x3xf32>

    // Experimental dialect
    %2 = "experimental.fuse"(%1) {optimization_level = 3 : i32} : (tensor<2x3xf32>) -> tensor<2x3xf32>

    // Standard MLIR (known dialect)
    %3 = arith.addf %2, %arg0 : tensor<2x3xf32>

    // Another custom operation
    %result = acme.finalize %3 : tensor<2x3xf32>

    func.return %result : tensor<2x3xf32>
  }
}
