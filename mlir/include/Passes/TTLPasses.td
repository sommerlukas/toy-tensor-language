#ifndef TTL_PASSES
#define TTL_PASSES

include "mlir/Pass/PassBase.td"

def TTLTiling : Pass<"ttl-tile", "::mlir::ModuleOp"> {

  let summary = "Perform matrix multiplication tiling";

  let dependentDialects = ["arith::ArithDialect", "scf::SCFDialect", "ttl::TTLDialect"];

}

#endif // TTL_PASSES
