// RUN: ttlc --start-with bufferize --stop-after mlir-translate %s -o %t
 
module {
  func.func @add(%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
    %0 = tensor.empty() : tensor<4x4xi32>
    %1 = linalg.add ins(%arg0, %arg1 : tensor<4x4xi32>, tensor<4x4xi32>) outs(%0 : tensor<4x4xi32>) -> tensor<4x4xi32>
    return %1 : tensor<4x4xi32>
  }
  func.func @sub(%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
    %0 = tensor.empty() : tensor<4x4xi32>
    %1 = linalg.sub ins(%arg0, %arg1 : tensor<4x4xi32>, tensor<4x4xi32>) outs(%0 : tensor<4x4xi32>) -> tensor<4x4xi32>
    return %1 : tensor<4x4xi32>
  }
  func.func @mul(%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
    %0 = tensor.empty() : tensor<4x4xi32>
    %1 = linalg.mul ins(%arg0, %arg1 : tensor<4x4xi32>, tensor<4x4xi32>) outs(%0 : tensor<4x4xi32>) -> tensor<4x4xi32>
    return %1 : tensor<4x4xi32>
  }
  func.func @div(%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
    %0 = tensor.empty() : tensor<4x4xi32>
    %1 = linalg.div ins(%arg0, %arg1 : tensor<4x4xi32>, tensor<4x4xi32>) outs(%0 : tensor<4x4xi32>) -> tensor<4x4xi32>
    return %1 : tensor<4x4xi32>
  }
  func.func @add_dyn(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?xi32>) -> tensor<?x?xi32> {
    %idx1 = index.constant 1
    %idx0 = index.constant 0
    %dim = tensor.dim %arg0, %idx0 : tensor<?x?xi32>
    %dim_0 = tensor.dim %arg0, %idx1 : tensor<?x?xi32>
    %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xi32>
    %1 = linalg.add ins(%arg0, %arg1 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%0 : tensor<?x?xi32>) -> tensor<?x?xi32>
    return %1 : tensor<?x?xi32>
  }
  func.func @sub_dyn(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?xi32>) -> tensor<?x?xi32> {
    %idx1 = index.constant 1
    %idx0 = index.constant 0
    %dim = tensor.dim %arg0, %idx0 : tensor<?x?xi32>
    %dim_0 = tensor.dim %arg0, %idx1 : tensor<?x?xi32>
    %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xi32>
    %1 = linalg.sub ins(%arg0, %arg1 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%0 : tensor<?x?xi32>) -> tensor<?x?xi32>
    return %1 : tensor<?x?xi32>
  }
  func.func @mul_dyn(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?xi32>) -> tensor<?x?xi32> {
    %idx1 = index.constant 1
    %idx0 = index.constant 0
    %dim = tensor.dim %arg0, %idx0 : tensor<?x?xi32>
    %dim_0 = tensor.dim %arg0, %idx1 : tensor<?x?xi32>
    %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xi32>
    %1 = linalg.mul ins(%arg0, %arg1 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%0 : tensor<?x?xi32>) -> tensor<?x?xi32>
    return %1 : tensor<?x?xi32>
  }
  func.func @div_dyn(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?xi32>) -> tensor<?x?xi32> {
    %idx1 = index.constant 1
    %idx0 = index.constant 0
    %dim = tensor.dim %arg0, %idx0 : tensor<?x?xi32>
    %dim_0 = tensor.dim %arg0, %idx1 : tensor<?x?xi32>
    %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xi32>
    %1 = linalg.div ins(%arg0, %arg1 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%0 : tensor<?x?xi32>) -> tensor<?x?xi32>
    return %1 : tensor<?x?xi32>
  }
}

