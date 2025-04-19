// RUN: ttlc --start-with bufferize --stop-after mlir-translate %s -o %t
 
module {
  func.func @dim_dyn(%arg0: tensor<?x?x?xi32>, %arg1: i32) -> i32 {
    %0 = index.casts %arg1 : i32 to index
    %dim = tensor.dim %arg0, %0 : tensor<?x?x?xi32>
    %1 = index.casts %dim : index to i32
    return %1 : i32
  }
  func.func @slice_mixed(%arg0: tensor<?x?xi32>, %arg1: i32, %arg2: i32, %arg3: i32) -> tensor<?x1xi32> {
    %0 = arith.subi %arg2, %arg1 : i32
    %1 = index.casts %arg1 : i32 to index
    %2 = index.casts %0 : i32 to index
    %3 = index.casts %arg3 : i32 to index
    %extracted_slice = tensor.extract_slice %arg0[%1, %3] [%2, 1] [1, 1] : tensor<?x?xi32> to tensor<?x1xi32>
    return %extracted_slice : tensor<?x1xi32>
  }
  func.func @slice_range(%arg0: tensor<16x16x16xf32>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) -> tensor<?x?x?xf32> {
    %0 = arith.subi %arg2, %arg1 : i32
    %1 = arith.subi %arg4, %arg3 : i32
    %2 = arith.subi %arg6, %arg5 : i32
    %3 = index.casts %arg1 : i32 to index
    %4 = index.casts %0 : i32 to index
    %5 = index.casts %arg3 : i32 to index
    %6 = index.casts %1 : i32 to index
    %7 = index.casts %arg5 : i32 to index
    %8 = index.casts %2 : i32 to index
    %extracted_slice = tensor.extract_slice %arg0[%3, %5, %7] [%4, %6, %8] [1, 1, 1] : tensor<16x16x16xf32> to tensor<?x?x?xf32>
    return %extracted_slice : tensor<?x?x?xf32>
  }
  func.func @slice_int(%arg0: tensor<?x?x?xf32>, %arg1: i32, %arg2: i32, %arg3: i32) -> f32 {
    %0 = index.casts %arg1 : i32 to index
    %1 = index.casts %arg2 : i32 to index
    %2 = index.casts %arg3 : i32 to index
    %extracted = tensor.extract %arg0[%0, %1, %2] : tensor<?x?x?xf32>
    return %extracted : f32
  }
  func.func @matrix_init(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32) -> tensor<2x2xf32> {
    %0 = arith.addf %arg3, %arg4 : f32
    %from_elements = tensor.from_elements %arg0, %arg1, %arg2, %0 : tensor<2x2xf32>
    return %from_elements : tensor<2x2xf32>
  }
}

