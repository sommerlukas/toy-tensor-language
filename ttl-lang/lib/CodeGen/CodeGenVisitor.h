#pragma once

#include "AST.h"
#include "ASTVisitor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"

namespace ttl::codegen {
class CodeGenVisitor : public ast::ASTVisitor {

private:
  mlir::MLIRContext *Ctx;
  mlir::OpBuilder Builder;
  mlir::ModuleOp MLIRModule;

  llvm::DenseMap<ttl::ast::VarRefPtr, mlir::Value> LastDefs;

  llvm::DenseMap<ttl::ast::ExprPtr, mlir::Value> ValueMap;

  mlir::Location translateLoc(ast::ASTNodePtrBase *Node) {
    return Builder.getUnknownLoc();
  }
  mlir::Value createIntConstant(::ttl::ast::ASTNodePtrBase *Node,
                                int32_t Value);

  mlir::Value createFloatConstant(::ttl::ast::ASTNodePtrBase *Node,
                                  float Value);

public:
  CodeGenVisitor(mlir::MLIRContext *Ctx, mlir::OpBuilder Builder,
                 mlir::ModuleOp MLIRModule)
      : Ctx{Ctx}, Builder{Builder}, MLIRModule{MLIRModule} {}

  void visit(ast::Module *) override;
  void visit(ast::Function *) override;
  void visit(ast::VarDef *) override;
  void visit(ast::MatrixAssign *) override;
  void visit(ast::ScalarAssign *) override;
  void visit(ast::CallStmt *) override;
  void visit(ast::ForLoop *) override;
  void visit(ast::IfStmt *) override;
  void visit(ast::ReturnStmt *) override;
  void visit(ast::CompoundStmt *) override;

  void visit(ast::CondExpr *) override;
  void visit(ast::SliceExpr *) override;
  void visit(ast::UnExpr *) override;
  void visit(ast::BinExpr *) override;
  void visit(ast::MatrixInit *) override;
  void visit(ast::RangeExpr *) override;
  void visit(ast::CallExpr *) override;
  void visit(ast::IDRef *) override;
  void visit(ast::FloatLiteral *) override;
  void visit(ast::IntLiteral *) override;
};
} // namespace ttl::codegen
