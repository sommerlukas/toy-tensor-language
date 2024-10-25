#pragma once

#include "AST.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

namespace ttl::codegen {

class CodeGen {

public:
  static mlir::ModuleOp generate(ttl::ast::Module *, mlir::MLIRContext *);
};

} // namespace ttl::codegen
