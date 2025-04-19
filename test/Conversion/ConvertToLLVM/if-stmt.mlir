// RUN: ttlc --start-with bufferize --stop-after mlir-translate %s -o %t
 
module {
  func.func @control_flow_if(%arg0: i32, %arg1: i32) -> f32 {
    %idx1 = index.constant 1
    %idx0 = index.constant 0
    %cst = arith.constant 8.000000e+00 : f32
    %cst_0 = arith.constant 9.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e+01 : f32
    %cst_2 = arith.constant 7.000000e+00 : f32
    %cst_3 = arith.constant 6.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %0 = tensor.empty() : tensor<2x2xf32>
    %1 = arith.cmpi ne, %arg0, %c0_i32 : i32
    %2 = arith.select %1, %cst_0, %cst : f32
    %3 = scf.if %1 -> (tensor<2x2xf32>) {
      %5 = arith.cmpi ne, %arg1, %c0_i32 : i32
      %6 = scf.if %5 -> (tensor<2x2xf32>) {
        %inserted = tensor.insert %cst_3 into %0[%idx0, %idx1] : tensor<2x2xf32>
        %inserted_4 = tensor.insert %cst_2 into %inserted[%idx1, %idx0] : tensor<2x2xf32>
        scf.yield %inserted_4 : tensor<2x2xf32>
      } else {
        %inserted = tensor.insert %cst_2 into %0[%idx0, %idx1] : tensor<2x2xf32>
        %inserted_4 = tensor.insert %cst_1 into %inserted[%idx1, %idx1] : tensor<2x2xf32>
        scf.yield %inserted_4 : tensor<2x2xf32>
      }
      scf.yield %6 : tensor<2x2xf32>
    } else {
      scf.yield %0 : tensor<2x2xf32>
    }
    %extracted = tensor.extract %3[%idx1, %idx0] : tensor<2x2xf32>
    %4 = arith.addf %2, %extracted : f32
    return %4 : f32
  }
}

