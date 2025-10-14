# MLIR Code Examples

This document demonstrates MLIR (Multi-Level Intermediate Representation) syntax highlighting in markserv.

## Simple Arithmetic Function

Below is an example of a simple function that adds two integers:

```mlir
func.func @add(%arg0: i32, %arg1: i32) -> i32 {
  %sum = arith.addi %arg0, %arg1 : i32
  return %sum : i32
}
```

## Tensor Operations

MLIR supports various tensor and memref operations:

```mlir
func.func @tensor_ops(%input: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %c0 = arith.constant 0.0 : f32
  %c1 = arith.constant 1.0 : f32

  // Create a tensor filled with ones
  %ones = tensor.splat %c1 : tensor<4x8xf32>

  // Add tensors element-wise
  %result = arith.addf %input, %ones : tensor<4x8xf32>

  return %result : tensor<4x8xf32>
}
```

## Control Flow Example

MLIR supports structured control flow:

```mlir
func.func @max(%a: f32, %b: f32) -> f32 {
  %cmp = arith.cmpf ogt, %a, %b : f32
  %max = scf.if %cmp -> f32 {
    scf.yield %a : f32
  } else {
    scf.yield %b : f32
  }
  return %max : f32
}
```

## Loop Constructs

Example with affine loops:

```mlir
func.func @matrix_multiply(%A: memref<64x32xf32>,
                           %B: memref<32x64xf32>,
                           %C: memref<64x64xf32>) {
  affine.for %i = 0 to 64 {
    affine.for %j = 0 to 64 {
      affine.for %k = 0 to 32 {
        %a = affine.load %A[%i, %k] : memref<64x32xf32>
        %b = affine.load %B[%k, %j] : memref<32x64xf32>
        %c = affine.load %C[%i, %j] : memref<64x64xf32>
        %prod = arith.mulf %a, %b : f32
        %sum = arith.addf %c, %prod : f32
        affine.store %sum, %C[%i, %j] : memref<64x64xf32>
      }
    }
  }
  return
}
```

## GPU Kernel Example

MLIR can represent GPU computations:

```mlir
gpu.module @kernels {
  gpu.func @add_vectors(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>)
    kernel attributes {gpu.known_block_size = array<i32: 256, 1, 1>} {
    %bid = gpu.block_id x
    %tid = gpu.thread_id x
    %cst = arith.constant 256 : index

    %idx = arith.muli %bid, %cst : index
    %gid = arith.addi %idx, %tid : index

    %val0 = memref.load %arg0[%gid] : memref<?xf32>
    %val1 = memref.load %arg1[%gid] : memref<?xf32>
    %sum = arith.addf %val0, %val1 : f32
    memref.store %sum, %arg2[%gid] : memref<?xf32>

    gpu.return
  }
}
```

## Regular Code Block

For comparison, here's a regular code block:

```javascript
function add(a, b) {
  return a + b;
}
```

And some Python:

```python
def multiply(x, y):
    return x * y
```

This demonstrates that MLIR code blocks are properly highlighted alongside other languages.