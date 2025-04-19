// RUN: ttlc --start-with bufferize --stop-after mlir-translate %s -o %t
 
module {
  func.func @matmul(%arg0: tensor<4x3xf32>, %arg1: tensor<3x4xf32>) -> tensor<4x4xf32> {
    %0 = tensor.empty() : tensor<4x4xf32>
    %1 = linalg.matmul ins(%arg0, %arg1 : tensor<4x3xf32>, tensor<3x4xf32>) outs(%0 : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }
  func.func @matmul_dyn(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %idx1 = index.constant 1
    %idx0 = index.constant 0
    %dim = tensor.dim %arg0, %idx0 : tensor<?x?xf32>
    %dim_0 = tensor.dim %arg1, %idx1 : tensor<?x?xf32>
    %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
    %1 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    return %1 : tensor<?x?xf32>
  }
}

