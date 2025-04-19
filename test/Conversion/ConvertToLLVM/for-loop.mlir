// RUN: ttlc --start-with bufferize --stop-after mlir-translate %s -o %t
 
module {
  func.func @control_flow_if(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32) -> f32 {
    %idx2 = index.constant 2
    %idx1 = index.constant 1
    %idx0 = index.constant 0
    %cst = arith.constant 4.200000e+01 : f32
    %c2_i32 = arith.constant 2 : i32
    %cst_0 = arith.constant 7.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant 8.000000e+00 : f32
    %cst_2 = arith.constant 6.000000e+00 : f32
    %cst_3 = arith.constant 5.000000e+00 : f32
    %0 = tensor.empty() : tensor<8x8xf32>
    %1 = tensor.empty() : tensor<8xf32>
    %2:5 = scf.for %arg4 = %arg1 to %arg2 step %c1_i32 iter_args(%arg5 = %cst_3, %arg6 = %cst_2, %arg7 = %cst_1, %arg8 = %0, %arg9 = %1) -> (f32, f32, f32, tensor<8x8xf32>, tensor<8xf32>)  : i32 {
      %7:2 = scf.for %arg10 = %c0_i32 to %arg3 step %c2_i32 iter_args(%arg11 = %arg7, %arg12 = %arg8) -> (f32, tensor<8x8xf32>)  : i32 {
        %9 = arith.cmpi ne, %arg0, %c0_i32 : i32
        %10 = arith.select %9, %cst, %arg11 : f32
        %11 = index.casts %arg4 : i32 to index
        %12 = index.casts %arg10 : i32 to index
        %inserted_5 = tensor.insert %cst_1 into %arg12[%11, %12] : tensor<8x8xf32>
        scf.yield %10, %inserted_5 : f32, tensor<8x8xf32>
      }
      %8 = index.casts %arg4 : i32 to index
      %inserted = tensor.insert %cst_0 into %arg9[%8] : tensor<8xf32>
      scf.yield %cst_1, %cst_0, %7#0, %7#1, %inserted : f32, f32, f32, tensor<8x8xf32>, tensor<8xf32>
    }
    %extracted = tensor.extract %2#3[%idx0, %idx1] : tensor<8x8xf32>
    %extracted_4 = tensor.extract %2#4[%idx2] : tensor<8xf32>
    %3 = arith.mulf %extracted, %extracted_4 : f32
    %4 = arith.mulf %3, %2#0 : f32
    %5 = arith.mulf %4, %2#1 : f32
    %6 = arith.mulf %5, %2#2 : f32
    return %6 : f32
  }
}

