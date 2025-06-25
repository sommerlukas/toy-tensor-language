// RUN: ttl-opt --ttl-eliminate-init-loops %s | FileCheck %s

func.func @tensor_func(%arg0: !ttl.tensor<8x!ttl.int>) -> !ttl.tensor<8x!ttl.int> {
  "ttl.return"(%arg0) : (!ttl.tensor<8x!ttl.int>) -> ()
}
func.func @int_passthrough(%arg0: !ttl.int) -> !ttl.int {
  "ttl.return"(%arg0) : (!ttl.int) -> ()
}


// COM: An init loop that gets replaced successfully
// CHECK-LABEL:   func.func @init_loop1() -> !ttl.tensor<8x!ttl.int> {
// CHECK:           %[[VAL_0:.*]] = ttl.const_int 0
// CHECK:           %[[VAL_1:.*]] = ttl.const_int 8
// CHECK:           %[[VAL_2:.*]] = ttl.const_int 0
// CHECK:           %[[VAL_3:.*]] = ttl.const_int 1
// CHECK:           %[[VAL_4:.*]] = "ttl.tensor_range_init"(%[[VAL_0]], %[[VAL_1]]) : (!ttl.int, !ttl.int) -> !ttl.tensor<8x!ttl.int>
// CHECK:           "ttl.return"(%[[VAL_4]]) : (!ttl.tensor<8x!ttl.int>) -> ()
func.func @init_loop1() -> !ttl.tensor<8x!ttl.int> {
  %0 = ttl.const_int 0
  %1 = ttl.const_int 8
  %2 = "ttl.tensor_empty"() : () -> !ttl.tensor<8x!ttl.int>
  %3 = ttl.const_int 0
  %4 = ttl.const_int 1
  %5 = "ttl.for"(%0, %1, %4, %2) ({
  ^bb0(%arg0: !ttl.int, %arg1: !ttl.tensor<8x!ttl.int>):
    %6 = "ttl.tensor_insert"(%arg1, %arg0, %arg0) : (!ttl.tensor<8x!ttl.int>, !ttl.int, !ttl.int) -> !ttl.tensor<8x!ttl.int>
    "ttl.yield"(%6) : (!ttl.tensor<8x!ttl.int>) -> ()
  }) : (!ttl.int, !ttl.int, !ttl.int, !ttl.tensor<8x!ttl.int>) -> !ttl.tensor<8x!ttl.int>
  "ttl.return"(%5) : (!ttl.tensor<8x!ttl.int>) -> ()
}

// COM: An init loop that gets replaced, but has a non-zero lower bound.
// CHECK-LABEL:   func.func @init_loop2() -> !ttl.tensor<8x!ttl.int> {
// CHECK:           %[[VAL_0:.*]] = ttl.const_int 4
// CHECK:           %[[VAL_1:.*]] = ttl.const_int 12
// CHECK:           %[[VAL_2:.*]] = ttl.const_int 0
// CHECK:           %[[VAL_3:.*]] = ttl.const_int 1
// CHECK:           %[[VAL_4:.*]] = "ttl.tensor_range_init"(%[[VAL_0]], %[[VAL_1]]) : (!ttl.int, !ttl.int) -> !ttl.tensor<8x!ttl.int>
// CHECK:           "ttl.return"(%[[VAL_4]]) : (!ttl.tensor<8x!ttl.int>) -> ()
// CHECK:         }
func.func @init_loop2() -> !ttl.tensor<8x!ttl.int> {
  %0 = ttl.const_int 4
  %1 = ttl.const_int 12
  %2 = "ttl.tensor_empty"() : () -> !ttl.tensor<8x!ttl.int>
  %3 = ttl.const_int 0
  %4 = ttl.const_int 1
  %5 = "ttl.for"(%0, %1, %4, %2) ({
  ^bb0(%arg0: !ttl.int, %arg1: !ttl.tensor<8x!ttl.int>):
    %6 = "ttl.tensor_insert"(%arg1, %arg0, %arg0) : (!ttl.tensor<8x!ttl.int>, !ttl.int, !ttl.int) -> !ttl.tensor<8x!ttl.int>
    "ttl.yield"(%6) : (!ttl.tensor<8x!ttl.int>) -> ()
  }) : (!ttl.int, !ttl.int, !ttl.int, !ttl.tensor<8x!ttl.int>) -> !ttl.tensor<8x!ttl.int>
  "ttl.return"(%5) : (!ttl.tensor<8x!ttl.int>) -> ()
}

// COM: An example that fails because the loop bounds are not constant.
// CHECK-LABEL:   func.func @init_loop3(
// CHECK-SAME:      %[[VAL_0:.*]]: !ttl.int,
// CHECK-SAME:      %[[VAL_1:.*]]: !ttl.int) -> !ttl.tensor<8x!ttl.int> {
// CHECK:           %[[VAL_2:.*]] = "ttl.tensor_empty"() : () -> !ttl.tensor<8x!ttl.int>
// CHECK:           %[[VAL_3:.*]] = ttl.const_int 0
// CHECK:           %[[VAL_4:.*]] = ttl.const_int 1
// CHECK:           %[[VAL_5:.*]] = "ttl.for"(%[[VAL_0]], %[[VAL_1]], %[[VAL_4]], %[[VAL_2]]) ({
// CHECK:           ^bb0(%[[VAL_6:.*]]: !ttl.int, %[[VAL_7:.*]]: !ttl.tensor<8x!ttl.int>):
// CHECK:             %[[VAL_8:.*]] = "ttl.tensor_insert"(%[[VAL_7]], %[[VAL_6]], %[[VAL_6]]) : (!ttl.tensor<8x!ttl.int>, !ttl.int, !ttl.int) -> !ttl.tensor<8x!ttl.int>
// CHECK:             "ttl.yield"(%[[VAL_8]]) : (!ttl.tensor<8x!ttl.int>) -> ()
// CHECK:           }) : (!ttl.int, !ttl.int, !ttl.int, !ttl.tensor<8x!ttl.int>) -> !ttl.tensor<8x!ttl.int>
// CHECK:           "ttl.return"(%[[VAL_5]]) : (!ttl.tensor<8x!ttl.int>) -> ()
// CHECK:         }


func.func @init_loop3(%arg0: !ttl.int, %arg1: !ttl.int) -> !ttl.tensor<8x!ttl.int> {
  %0 = "ttl.tensor_empty"() : () -> !ttl.tensor<8x!ttl.int>
  %1 = ttl.const_int 0
  %2 = ttl.const_int 1
  %3 = "ttl.for"(%arg0, %arg1, %2, %0) ({
  ^bb0(%arg2: !ttl.int, %arg3: !ttl.tensor<8x!ttl.int>):
    %4 = "ttl.tensor_insert"(%arg3, %arg2, %arg2) : (!ttl.tensor<8x!ttl.int>, !ttl.int, !ttl.int) -> !ttl.tensor<8x!ttl.int>
    "ttl.yield"(%4) : (!ttl.tensor<8x!ttl.int>) -> ()
  }) : (!ttl.int, !ttl.int, !ttl.int, !ttl.tensor<8x!ttl.int>) -> !ttl.tensor<8x!ttl.int>
  "ttl.return"(%3) : (!ttl.tensor<8x!ttl.int>) -> ()
}

// COM: An example that fails because the tensor is multi-dimensional.
// CHECK-LABEL:   func.func @init_loop4() -> !ttl.tensor<2x4x!ttl.int> {
// CHECK:           %[[VAL_0:.*]] = ttl.const_int 4
// CHECK:           %[[VAL_1:.*]] = ttl.const_int 12
// CHECK:           %[[VAL_2:.*]] = "ttl.tensor_empty"() : () -> !ttl.tensor<2x4x!ttl.int>
// CHECK:           %[[VAL_3:.*]] = ttl.const_int 0
// CHECK:           %[[VAL_4:.*]] = ttl.const_int 1
// CHECK:           %[[VAL_5:.*]] = "ttl.for"(%[[VAL_0]], %[[VAL_1]], %[[VAL_4]], %[[VAL_2]]) ({
// CHECK:           ^bb0(%[[VAL_6:.*]]: !ttl.int, %[[VAL_7:.*]]: !ttl.tensor<2x4x!ttl.int>):
// CHECK:             %[[VAL_8:.*]] = "ttl.tensor_insert"(%[[VAL_7]], %[[VAL_6]], %[[VAL_6]]) : (!ttl.tensor<2x4x!ttl.int>, !ttl.int, !ttl.int) -> !ttl.tensor<2x4x!ttl.int>
// CHECK:             "ttl.yield"(%[[VAL_8]]) : (!ttl.tensor<2x4x!ttl.int>) -> ()
// CHECK:           }) : (!ttl.int, !ttl.int, !ttl.int, !ttl.tensor<2x4x!ttl.int>) -> !ttl.tensor<2x4x!ttl.int>
// CHECK:           "ttl.return"(%[[VAL_5]]) : (!ttl.tensor<2x4x!ttl.int>) -> ()
func.func @init_loop4() -> !ttl.tensor<2x4x!ttl.int> {
  %0 = ttl.const_int 4
  %1 = ttl.const_int 12
  %2 = "ttl.tensor_empty"() : () -> !ttl.tensor<2x4x!ttl.int>
  %3 = ttl.const_int 0
  %4 = ttl.const_int 1
  %5 = "ttl.for"(%0, %1, %4, %2) ({
  ^bb0(%arg0: !ttl.int, %arg1: !ttl.tensor<2x4x!ttl.int>):
    %6 = "ttl.tensor_insert"(%arg1, %arg0, %arg0) : (!ttl.tensor<2x4x!ttl.int>, !ttl.int, !ttl.int) -> !ttl.tensor<2x4x!ttl.int>
    "ttl.yield"(%6) : (!ttl.tensor<2x4x!ttl.int>) -> ()
  }) : (!ttl.int, !ttl.int, !ttl.int, !ttl.tensor<2x4x!ttl.int>) -> !ttl.tensor<2x4x!ttl.int>
  "ttl.return"(%5) : (!ttl.tensor<2x4x!ttl.int>) -> ()
}

// COM: A loop that gets removed, while the other use continues to use the original TensorEmpty.
// CHECK-LABEL:   func.func @init_loop5() -> !ttl.tensor<8x!ttl.int> {
// CHECK:           %[[VAL_0:.*]] = ttl.const_int 0
// CHECK:           %[[VAL_1:.*]] = ttl.const_int 8
// CHECK:           %[[VAL_2:.*]] = "ttl.tensor_empty"() : () -> !ttl.tensor<8x!ttl.int>
// CHECK:           %[[VAL_3:.*]] = ttl.const_int 0
// CHECK:           %[[VAL_4:.*]] = call @tensor_func(%[[VAL_2]]) : (!ttl.tensor<8x!ttl.int>) -> !ttl.tensor<8x!ttl.int>
// CHECK:           %[[VAL_5:.*]] = ttl.const_int 1
// CHECK:           %[[VAL_6:.*]] = "ttl.tensor_range_init"(%[[VAL_0]], %[[VAL_1]]) : (!ttl.int, !ttl.int) -> !ttl.tensor<8x!ttl.int>
// CHECK:           "ttl.return"(%[[VAL_6]]) : (!ttl.tensor<8x!ttl.int>) -> ()
func.func @init_loop5() -> !ttl.tensor<8x!ttl.int> {
  %0 = ttl.const_int 0
  %1 = ttl.const_int 8
  %2 = "ttl.tensor_empty"() : () -> !ttl.tensor<8x!ttl.int>
  %3 = ttl.const_int 0
  %4 = call @tensor_func(%2) : (!ttl.tensor<8x!ttl.int>) -> !ttl.tensor<8x!ttl.int>
  %5 = ttl.const_int 1
  %6 = "ttl.for"(%0, %1, %5, %2) ({
  ^bb0(%arg0: !ttl.int, %arg1: !ttl.tensor<8x!ttl.int>):
    %7 = "ttl.tensor_insert"(%arg1, %arg0, %arg0) : (!ttl.tensor<8x!ttl.int>, !ttl.int, !ttl.int) -> !ttl.tensor<8x!ttl.int>
    "ttl.yield"(%7) : (!ttl.tensor<8x!ttl.int>) -> ()
  }) : (!ttl.int, !ttl.int, !ttl.int, !ttl.tensor<8x!ttl.int>) -> !ttl.tensor<8x!ttl.int>
  "ttl.return"(%6) : (!ttl.tensor<8x!ttl.int>) -> ()
}

// COM: An example that fails because there's another operation in the loop body.
// CHECK-LABEL:   func.func @init_loop6() -> !ttl.tensor<8x!ttl.int> {
// CHECK:           %[[VAL_0:.*]] = ttl.const_int 0
// CHECK:           %[[VAL_1:.*]] = ttl.const_int 8
// CHECK:           %[[VAL_2:.*]] = "ttl.tensor_empty"() : () -> !ttl.tensor<8x!ttl.int>
// CHECK:           %[[VAL_3:.*]] = ttl.const_int 0
// CHECK:           %[[VAL_4:.*]] = ttl.const_int 1
// CHECK:           %[[VAL_5:.*]] = "ttl.for"(%[[VAL_0]], %[[VAL_1]], %[[VAL_4]], %[[VAL_2]]) ({
// CHECK:           ^bb0(%[[VAL_6:.*]]: !ttl.int, %[[VAL_7:.*]]: !ttl.tensor<8x!ttl.int>):
// CHECK:             %[[VAL_8:.*]] = func.call @int_passthrough(%[[VAL_6]]) : (!ttl.int) -> !ttl.int
// CHECK:             %[[VAL_9:.*]] = "ttl.tensor_insert"(%[[VAL_7]], %[[VAL_6]], %[[VAL_6]]) : (!ttl.tensor<8x!ttl.int>, !ttl.int, !ttl.int) -> !ttl.tensor<8x!ttl.int>
// CHECK:             "ttl.yield"(%[[VAL_9]]) : (!ttl.tensor<8x!ttl.int>) -> ()
// CHECK:           }) : (!ttl.int, !ttl.int, !ttl.int, !ttl.tensor<8x!ttl.int>) -> !ttl.tensor<8x!ttl.int>
// CHECK:           "ttl.return"(%[[VAL_5]]) : (!ttl.tensor<8x!ttl.int>) -> ()
func.func @init_loop6() -> !ttl.tensor<8x!ttl.int> {
  %0 = ttl.const_int 0
  %1 = ttl.const_int 8
  %2 = "ttl.tensor_empty"() : () -> !ttl.tensor<8x!ttl.int>
  %3 = ttl.const_int 0
  %4 = ttl.const_int 1
  %5 = "ttl.for"(%0, %1, %4, %2) ({
  ^bb0(%arg0: !ttl.int, %arg1: !ttl.tensor<8x!ttl.int>):
    %6 = func.call @int_passthrough(%arg0) : (!ttl.int) -> !ttl.int
    %7 = "ttl.tensor_insert"(%arg1, %arg0, %arg0) : (!ttl.tensor<8x!ttl.int>, !ttl.int, !ttl.int) -> !ttl.tensor<8x!ttl.int>
    "ttl.yield"(%7) : (!ttl.tensor<8x!ttl.int>) -> ()
  }) : (!ttl.int, !ttl.int, !ttl.int, !ttl.tensor<8x!ttl.int>) -> !ttl.tensor<8x!ttl.int>
  "ttl.return"(%5) : (!ttl.tensor<8x!ttl.int>) -> ()
}

// COM: An example that fails because the inserted value is not the loop induction var.
// CHECK-LABEL:   func.func @init_loop7() -> !ttl.tensor<8x!ttl.int> {
// CHECK:           %[[VAL_0:.*]] = ttl.const_int 0
// CHECK:           %[[VAL_1:.*]] = ttl.const_int 8
// CHECK:           %[[VAL_2:.*]] = "ttl.tensor_empty"() : () -> !ttl.tensor<8x!ttl.int>
// CHECK:           %[[VAL_3:.*]] = ttl.const_int 0
// CHECK:           %[[VAL_4:.*]] = ttl.const_int 1
// CHECK:           %[[VAL_5:.*]] = "ttl.for"(%[[VAL_0]], %[[VAL_1]], %[[VAL_4]], %[[VAL_2]]) ({
// CHECK:           ^bb0(%[[VAL_6:.*]]: !ttl.int, %[[VAL_7:.*]]: !ttl.tensor<8x!ttl.int>):
// CHECK:             %[[VAL_8:.*]] = func.call @int_passthrough(%[[VAL_6]]) : (!ttl.int) -> !ttl.int
// CHECK:             %[[VAL_9:.*]] = "ttl.tensor_insert"(%[[VAL_7]], %[[VAL_8]], %[[VAL_6]]) : (!ttl.tensor<8x!ttl.int>, !ttl.int, !ttl.int) -> !ttl.tensor<8x!ttl.int>
// CHECK:             "ttl.yield"(%[[VAL_9]]) : (!ttl.tensor<8x!ttl.int>) -> ()
// CHECK:           }) : (!ttl.int, !ttl.int, !ttl.int, !ttl.tensor<8x!ttl.int>) -> !ttl.tensor<8x!ttl.int>
// CHECK:           "ttl.return"(%[[VAL_5]]) : (!ttl.tensor<8x!ttl.int>) -> ()
func.func @init_loop7() -> !ttl.tensor<8x!ttl.int> {
  %0 = ttl.const_int 0
  %1 = ttl.const_int 8
  %2 = "ttl.tensor_empty"() : () -> !ttl.tensor<8x!ttl.int>
  %3 = ttl.const_int 0
  %4 = ttl.const_int 1
  %5 = "ttl.for"(%0, %1, %4, %2) ({
  ^bb0(%arg0: !ttl.int, %arg1: !ttl.tensor<8x!ttl.int>):
    %6 = func.call @int_passthrough(%arg0) : (!ttl.int) -> !ttl.int
    %7 = "ttl.tensor_insert"(%arg1, %6, %arg0) : (!ttl.tensor<8x!ttl.int>, !ttl.int, !ttl.int) -> !ttl.tensor<8x!ttl.int>
    "ttl.yield"(%7) : (!ttl.tensor<8x!ttl.int>) -> ()
  }) : (!ttl.int, !ttl.int, !ttl.int, !ttl.tensor<8x!ttl.int>) -> !ttl.tensor<8x!ttl.int>
  "ttl.return"(%5) : (!ttl.tensor<8x!ttl.int>) -> ()
}

// COM: An example that fails because the index is not the loop induction var.
// CHECK-LABEL:   func.func @init_loop8() -> !ttl.tensor<8x!ttl.int> {
// CHECK:           %[[VAL_0:.*]] = ttl.const_int 0
// CHECK:           %[[VAL_1:.*]] = ttl.const_int 8
// CHECK:           %[[VAL_2:.*]] = "ttl.tensor_empty"() : () -> !ttl.tensor<8x!ttl.int>
// CHECK:           %[[VAL_3:.*]] = ttl.const_int 0
// CHECK:           %[[VAL_4:.*]] = ttl.const_int 1
// CHECK:           %[[VAL_5:.*]] = "ttl.for"(%[[VAL_0]], %[[VAL_1]], %[[VAL_4]], %[[VAL_2]]) ({
// CHECK:           ^bb0(%[[VAL_6:.*]]: !ttl.int, %[[VAL_7:.*]]: !ttl.tensor<8x!ttl.int>):
// CHECK:             %[[VAL_8:.*]] = func.call @int_passthrough(%[[VAL_6]]) : (!ttl.int) -> !ttl.int
// CHECK:             %[[VAL_9:.*]] = "ttl.tensor_insert"(%[[VAL_7]], %[[VAL_6]], %[[VAL_8]]) : (!ttl.tensor<8x!ttl.int>, !ttl.int, !ttl.int) -> !ttl.tensor<8x!ttl.int>
// CHECK:             "ttl.yield"(%[[VAL_9]]) : (!ttl.tensor<8x!ttl.int>) -> ()
// CHECK:           }) : (!ttl.int, !ttl.int, !ttl.int, !ttl.tensor<8x!ttl.int>) -> !ttl.tensor<8x!ttl.int>
// CHECK:           "ttl.return"(%[[VAL_5]]) : (!ttl.tensor<8x!ttl.int>) -> ()
func.func @init_loop8() -> !ttl.tensor<8x!ttl.int> {
  %0 = ttl.const_int 0
  %1 = ttl.const_int 8
  %2 = "ttl.tensor_empty"() : () -> !ttl.tensor<8x!ttl.int>
  %3 = ttl.const_int 0
  %4 = ttl.const_int 1
  %5 = "ttl.for"(%0, %1, %4, %2) ({
  ^bb0(%arg0: !ttl.int, %arg1: !ttl.tensor<8x!ttl.int>):
    %6 = func.call @int_passthrough(%arg0) : (!ttl.int) -> !ttl.int
    %7 = "ttl.tensor_insert"(%arg1, %arg0, %6) : (!ttl.tensor<8x!ttl.int>, !ttl.int, !ttl.int) -> !ttl.tensor<8x!ttl.int>
    "ttl.yield"(%7) : (!ttl.tensor<8x!ttl.int>) -> ()
  }) : (!ttl.int, !ttl.int, !ttl.int, !ttl.tensor<8x!ttl.int>) -> !ttl.tensor<8x!ttl.int>
  "ttl.return"(%5) : (!ttl.tensor<8x!ttl.int>) -> ()
}

