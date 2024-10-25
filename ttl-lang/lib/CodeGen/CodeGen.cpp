#include "CodeGen.h"
#include "CodeGenVisitor.h"

mlir::ModuleOp ttl::codegen::CodeGen::generate(ttl::ast::Module *AST,
                                               mlir::MLIRContext *Ctx) {
  mlir::OpBuilder Builder{Ctx};

  auto MLIRModule = mlir::ModuleOp::create(Builder.getUnknownLoc());

  ttl::codegen::CodeGenVisitor Visitor(Ctx, Builder, MLIRModule);

  AST->accept(&Visitor);

  return MLIRModule;
}
