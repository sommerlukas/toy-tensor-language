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
