module {
  func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    // Custom op from an unregistered dialect. Parsed generically.
    %0 = "mydialect.foo"(%arg0) { alpha = 0.5 : f64 } : (tensor<2xf32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }
}

