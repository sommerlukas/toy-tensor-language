// RUN: ttlc --start-with bufferize --stop-after mlir-translate %s -o %t
 
module {
  func.func @add(%arg0: f32, %arg1: f32) -> f32 {
    %0 = arith.addf %arg0, %arg1 : f32
    return %0 : f32
  }
  func.func @sub(%arg0: f32, %arg1: f32) -> f32 {
    %0 = arith.subf %arg0, %arg1 : f32
    return %0 : f32
  }
  func.func @mul(%arg0: f32, %arg1: f32) -> f32 {
    %0 = arith.mulf %arg0, %arg1 : f32
    return %0 : f32
  }
  func.func @div(%arg0: f32, %arg1: f32) -> f32 {
    %0 = arith.divf %arg0, %arg1 : f32
    return %0 : f32
  }
  func.func @minus(%arg0: f32) -> f32 {
    %0 = arith.negf %arg0 : f32
    return %0 : f32
  }
}

