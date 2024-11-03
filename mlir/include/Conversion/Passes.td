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

def TTLToScalar : Pass<"convert-ttl-to-scalar", "ModuleOp"> {

  let summary = "Convert scalar and control flow TTL dialect operations upstream dialect operations";

  let dependentDialects = ["scf::SCFDialect", "index::IndexDialect", 
                            "arith::ArithDialect", "func::FuncDialect"];

}

#endif // MLIR_TTL_CONVERSION_PASSES
