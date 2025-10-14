# MLIR Syntax Highlighting Test

This file tests the MLIR syntax highlighting using highlightjs-mlir.

## Basic MLIR Example

```mlir
module {
  func.func @simple_add(%arg0: i32, %arg1: i32) -> i32 {
    %result = arith.addi %arg0, %arg1 : i32
    return %result : i32
  }
}
```

## MLIR with Types and Operations

```mlir
func.func @matmul(%A: tensor<128x128xf32>, %B: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %c0 = arith.constant 0.0 : f32
  %C = linalg.fill ins(%c0 : f32) outs(%A : tensor<128x128xf32>) -> tensor<128x128xf32>
  %D = linalg.matmul ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%C : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %D : tensor<128x128xf32>
}
```

## MLIR with Attributes and Regions

```mlir
module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @vector_add(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>, %arg2: memref<1024xf32>)
      kernel attributes {gpu.known_block_size = array<i32: 256, 1, 1>} {
      %0 = gpu.block_id x
      %1 = gpu.block_dim x
      %2 = arith.muli %0, %1 : index
      %3 = gpu.thread_id x
      %4 = arith.addi %2, %3 : index
      %5 = memref.load %arg0[%4] : memref<1024xf32>
      %6 = memref.load %arg1[%4] : memref<1024xf32>
      %7 = arith.addf %5, %6 : f32
      memref.store %7, %arg2[%4] : memref<1024xf32>
      gpu.return
    }
  }
}
```

## Testing Comments and Strings

```mlir
// This is a single-line comment
module {
  /* This is a
     multi-line comment */
  func.func @string_test() -> !llvm.ptr<i8> {
    %str = llvm.mlir.addressof @"hello_world" : !llvm.ptr<array<12 x i8>>
    return %str : !llvm.ptr<i8>
  }
}
```
