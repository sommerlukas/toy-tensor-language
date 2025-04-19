// RUN: ttlc --start-with bufferize --stop-after mlir-translate %s -o %t
 
module {
  func.func @matrix_rand_init() -> tensor<2x2xi32> {
    %0 = tensor.empty() : tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
  }
  func.func @matrix_scalar_init() -> tensor<3x4xf32> {
    %cst = arith.constant dense<4.200000e+01> : tensor<3x4xf32>
    return %cst : tensor<3x4xf32>
  }
  func.func @matrix_range_init() -> tensor<4x4xi32> {
    %idx4 = index.constant 4
    %c1_i32 = arith.constant 1 : i32
    %generated = tensor.generate  {
    ^bb0(%arg0: index, %arg1: index):
      %0 = index.mul %arg0, %idx4
      %1 = index.add %0, %arg1
      %2 = index.casts %1 : index to i32
      %3 = arith.addi %2, %c1_i32 : i32
      tensor.yield %3 : i32
    } : tensor<4x4xi32>
    return %generated : tensor<4x4xi32>
  }
  func.func @fixed_matrix_elem_assign(%arg0: f32) -> tensor<2x2xf32> {
    %cst = arith.constant dense<2.500000e+01> : tensor<2x2xf32>
    %idx0 = index.constant 0
    %idx1 = index.constant 1
    %inserted = tensor.insert %arg0 into %cst[%idx0, %idx1] : tensor<2x2xf32>
    return %inserted : tensor<2x2xf32>
  }
  func.func @dyn_matrix_elem_assign(%arg0: f32, %arg1: i32, %arg2: i32) -> tensor<2x2xf32> {
    %cst = arith.constant dense<2.500000e+01> : tensor<2x2xf32>
    %0 = index.casts %arg1 : i32 to index
    %1 = index.casts %arg2 : i32 to index
    %inserted = tensor.insert %arg0 into %cst[%0, %1] : tensor<2x2xf32>
    return %inserted : tensor<2x2xf32>
  }
}

