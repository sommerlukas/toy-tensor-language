// RUN: ttlc --start-with bufferize --stop-after mlir-translate %s -o %t
 
module {
  func.func @gt(%arg0: f32, %arg1: f32) -> i32 {
    %0 = arith.cmpf ugt, %arg0, %arg1 : f32
    %1 = arith.extsi %0 : i1 to i32
    return %1 : i32
  }
  func.func @ge(%arg0: i32, %arg1: i32) -> i32 {
    %0 = arith.cmpi sge, %arg0, %arg1 : i32
    %1 = arith.extsi %0 : i1 to i32
    return %1 : i32
  }
  func.func @lt(%arg0: f32, %arg1: f32) -> i32 {
    %0 = arith.cmpf ult, %arg0, %arg1 : f32
    %1 = arith.extsi %0 : i1 to i32
    return %1 : i32
  }
  func.func @le(%arg0: i32, %arg1: i32) -> i32 {
    %0 = arith.cmpi sle, %arg0, %arg1 : i32
    %1 = arith.extsi %0 : i1 to i32
    return %1 : i32
  }
  func.func @eq(%arg0: f32, %arg1: f32) -> i32 {
    %0 = arith.cmpf ueq, %arg0, %arg1 : f32
    %1 = arith.extsi %0 : i1 to i32
    return %1 : i32
  }
  func.func @ne(%arg0: f32, %arg1: f32) -> i32 {
    %0 = arith.cmpf une, %arg0, %arg1 : f32
    %1 = arith.extsi %0 : i1 to i32
    return %1 : i32
  }
}

