// Example MLIR code for testing markserv rendering
module {
  // Function that adds two 32-bit integers
  func.func @add_integers(%arg0: i32, %arg1: i32) -> i32 {
    %result = arith.addi %arg0, %arg1 : i32
    return %result : i32
  }

  // Matrix multiplication using linalg dialect
  func.func @matmul(%A: memref<128x256xf32>,
                     %B: memref<256x512xf32>,
                     %C: memref<128x512xf32>) {
    linalg.matmul ins(%A, %B : memref<128x256xf32>, memref<256x512xf32>)
                  outs(%C : memref<128x512xf32>)
    return
  }

  // Example with control flow
  func.func @conditional_branch(%cond: i1, %a: f32, %b: f32) -> f32 {
    %result = scf.if %cond -> f32 {
      %sum = arith.addf %a, %b : f32
      scf.yield %sum : f32
    } else {
      %diff = arith.subf %a, %b : f32
      scf.yield %diff : f32
    }
    return %result : f32
  }

  // Vector operations example
  func.func @vector_add(%v1: vector<4xf32>, %v2: vector<4xf32>) -> vector<4xf32> {
    %result = arith.addf %v1, %v2 : vector<4xf32>
    return %result : vector<4xf32>
  }

  // GPU kernel example
  gpu.module @gpu_module {
    gpu.func @kernel(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>) kernel {
      %block_id = gpu.block_id x
      %thread_id = gpu.thread_id x
      %c64 = arith.constant 64 : index
      %idx = arith.muli %block_id, %c64 : index
      %final_idx = arith.addi %idx, %thread_id : index
      %val = memref.load %arg0[%final_idx] : memref<1024xf32>
      %squared = arith.mulf %val, %val : f32
      memref.store %squared, %arg1[%final_idx] : memref<1024xf32>
      gpu.return
    }
  }

  // Affine loop example
  func.func @affine_example(%buffer: memref<100x100xf32>) {
    %c0 = arith.constant 0.0 : f32
    affine.for %i = 0 to 100 {
      affine.for %j = 0 to 100 {
        affine.store %c0, %buffer[%i, %j] : memref<100x100xf32>
      }
    }
    return
  }
}