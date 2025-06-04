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

module {
  %cst = ttl.const_int 42
  // Note on testing: Normally, one would only lit-test custom verifiers, not
  // the automatically generated checks such as type consistency or trait-based
  // constraints. It's possible, though.
  // @expected-error@below {{'ttl.return' op expects parent op 'func.func'}}
  "ttl.return"(%cst) : (!ttl.int) -> ()
}

// -----

func.func @matmul_1(%a: !ttl.float, %b: !ttl.tensor<?x?x!ttl.float>) -> !ttl.tensor<?x?x!ttl.float> {
  // @expected-error@below {{operands and result must be tensors of the same element type}}
  %c = "ttl.matmul"(%a, %b) : (!ttl.float, !ttl.tensor<?x?x!ttl.float>) -> !ttl.tensor<?x?x!ttl.float>
  "ttl.return"(%c) : (!ttl.tensor<?x?x!ttl.float>) -> ()
}

// -----

func.func @matmul_2(%a: !ttl.tensor<?x?x!ttl.int>, %b: !ttl.tensor<?x?x!ttl.float>) -> !ttl.tensor<?x?x!ttl.float> {
  // @expected-error@below {{operands and result must be tensors of the same element type}}
  %c = "ttl.matmul"(%a, %b) : (!ttl.tensor<?x?x!ttl.int>, !ttl.tensor<?x?x!ttl.float>) -> !ttl.tensor<?x?x!ttl.float>
  "ttl.return"(%c) : (!ttl.tensor<?x?x!ttl.float>) -> ()
}

// -----

func.func @matmul_3(%a: !ttl.tensor<?x?x!ttl.float>, %b: !ttl.tensor<?x?x!ttl.float>) -> !ttl.tensor<?x?x?x!ttl.float> {
  // @expected-error@below {{operand- and result tensors must be 2-dimensional}}
  %c = "ttl.matmul"(%a, %b) : (!ttl.tensor<?x?x!ttl.float>, !ttl.tensor<?x?x!ttl.float>) -> !ttl.tensor<?x?x?x!ttl.float>
  "ttl.return"(%c) : (!ttl.tensor<?x?x?x!ttl.float>) -> ()
}

// -----

func.func @matmul_4(%a: !ttl.tensor<?x4x!ttl.float>, %b: !ttl.tensor<3x?x!ttl.float>) -> !ttl.tensor<?x?x!ttl.float> {
  // @expected-error@below {{shape mismatch in common dimension}}
  %c = "ttl.matmul"(%a, %b) : (!ttl.tensor<?x4x!ttl.float>, !ttl.tensor<3x?x!ttl.float>) -> !ttl.tensor<?x?x!ttl.float>
  "ttl.return"(%c) : (!ttl.tensor<?x?x!ttl.float>) -> ()
}

// -----

func.func @matmul_5(%a: !ttl.tensor<5x?x!ttl.float>, %b: !ttl.tensor<?x?x!ttl.float>) -> !ttl.tensor<4x?x!ttl.float> {
  // @expected-error@below {{result shape mismatch in first dimension}}
  %c = "ttl.matmul"(%a, %b) : (!ttl.tensor<5x?x!ttl.float>, !ttl.tensor<?x?x!ttl.float>) -> !ttl.tensor<4x?x!ttl.float>
  "ttl.return"(%c) : (!ttl.tensor<4x?x!ttl.float>) -> ()
}

// -----

func.func @matmul_6(%a: !ttl.tensor<?x?x!ttl.float>, %b: !ttl.tensor<?x3x!ttl.float>) -> !ttl.tensor<?x4x!ttl.float> {
  // @expected-error@below {{result shape mismatch in second dimension}}
  %c = "ttl.matmul"(%a, %b) : (!ttl.tensor<?x?x!ttl.float>, !ttl.tensor<?x3x!ttl.float>) -> !ttl.tensor<?x4x!ttl.float>
  "ttl.return"(%c) : (!ttl.tensor<?x4x!ttl.float>) -> ()
}
