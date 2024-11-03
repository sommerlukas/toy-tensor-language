#ifndef MLIR_TTL_CONVERSION_PASSES
#define MLIR_TTL_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TTLToTensor : Pass<"convert-ttl-to-tensor", "ModuleOp"> {

  let summary = "Convert TTL dialect operations to Tensor dialect operations";

  let dependentDialects = ["tensor::TensorDialect", "index::IndexDialect", 
                            "arith::ArithDialect"];

}

def TTLToLinalg : Pass<"convert-ttl-to-linalg", "ModuleOp"> {

  let summary = "Convert TTL dialect operations to Linalg dialect operations";

  let dependentDialects = ["linalg::LinalgDialect", "index::IndexDialect", 
                            "arith::ArithDialect"];

}

#endif // MLIR_TTL_CONVERSION_PASSES
