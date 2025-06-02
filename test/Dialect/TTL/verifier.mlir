// RUN: ttl-opt --verify-diagnostics --split-input-file %s

func.func @return_values_number_mismatch() -> () {
  %cst = ttl.const_int 42
  // @expected-error@below {{enclosing function @return_values_number_mismatch expects 0 results}}
  "ttl.return"(%cst) : (!ttl.int) -> ()
}

// -----

func.func @return_values_type_mismatch() -> (!ttl.float) {
  %cst = ttl.const_int 42
  // @expected-error@below {{operand is a '!ttl.int', but the enclosing function @return_values_type_mismatch expects '!ttl.float'}} 
  "ttl.return"(%cst) : (!ttl.int) -> ()
}

// -----

func.func @bin_op_elementwise(%lhs: !ttl.tensor<4x4x!ttl.float>, %rhs: !ttl.tensor<4x4x!ttl.int>) -> (!ttl.tensor<4x4x!ttl.float>) {
  // @expected-error@below {{tensor operands have different element types: '!ttl.float' != '!ttl.int'}}
  %0 = "ttl.add"(%lhs, %rhs) : (!ttl.tensor<4x4x!ttl.float>, !ttl.tensor<4x4x!ttl.int>) -> !ttl.tensor<4x4x!ttl.float>
  "ttl.return"(%0) : (!ttl.tensor<4x4x!ttl.float>) -> ()
}
