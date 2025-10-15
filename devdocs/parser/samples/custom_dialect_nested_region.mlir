module {
  func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    // Custom op with a nested region. Parsed and walked generically.
    %r = "mydialect.region"(%arg0) ({
      %x = "mydialect.inner"(%arg0) { k = 7 : i64 } : (tensor<2xf32>) -> tensor<2xf32>
      // No terminator required for unknown dialect op region in generic parsing.
    }) : (tensor<2xf32>) -> tensor<2xf32>
    return %r : tensor<2xf32>
  }
}

