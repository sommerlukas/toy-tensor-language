// RUN: ttlc --start-with bufferize --stop-after mlir-translate %s -o %t
 
module {
  func.func @add(%arg0: i32, %arg1: i32) -> i32 {
    %0 = arith.addi %arg0, %arg1 : i32
    return %0 : i32
  }
  func.func @sub(%arg0: i32, %arg1: i32) -> i32 {
    %0 = arith.subi %arg0, %arg1 : i32
    return %0 : i32
  }
  func.func @mul(%arg0: i32, %arg1: i32) -> i32 {
    %0 = arith.muli %arg0, %arg1 : i32
    return %0 : i32
  }
  func.func @div(%arg0: i32, %arg1: i32) -> i32 {
    %0 = arith.divsi %arg0, %arg1 : i32
    return %0 : i32
  }
  func.func @and(%arg0: i32, %arg1: i32) -> i32 {
    %0 = arith.andi %arg0, %arg1 : i32
    return %0 : i32
  }
  func.func @or(%arg0: i32, %arg1: i32) -> i32 {
    %0 = arith.ori %arg0, %arg1 : i32
    return %0 : i32
  }
  func.func @minus(%arg0: i32) -> i32 {
    %c-1_i32 = arith.constant -1 : i32
    %0 = arith.muli %arg0, %c-1_i32 : i32
    return %0 : i32
  }
  func.func @not(%arg0: i32) -> i32 {
    %c-1_i32 = arith.constant -1 : i32
    %0 = arith.xori %arg0, %c-1_i32 : i32
    return %0 : i32
  }
}

