#pragma once

#include "AST.h"
#include "ASTContext.h"
#include "TTLBaseVisitor.h"

namespace ttl::parser {

class ASTBuilder : public TTLBaseVisitor {
public:
  ASTBuilder(ast::ASTContext *Ctx) : ASTCtx{Ctx} {}

  std::any visitModule(TTLParser::ModuleContext *ctx) override;

  std::any visitFunction(TTLParser::FunctionContext *FuncCtx) override;

  std::any visitArgument(TTLParser::ArgumentContext *Ctx) override;

  std::any visitIntType(TTLParser::IntTypeContext *Ctx) override;

  std::any visitFloatType(TTLParser::FloatTypeContext *) override;

  std::any visitVoidType(TTLParser::VoidTypeContext *ctx) override;

  std::any visitMatrixType(TTLParser::MatrixTypeContext *ctx) override;

  std::any visitVarDef(TTLParser::VarDefContext *ctx) override;

  std::any visitVarAssign(TTLParser::VarAssignContext *ctx) override;

  std::any
  visitMatrixElemAssign(TTLParser::MatrixElemAssignContext *ctx) override;

  std::any visitPrimAssign(TTLParser::PrimAssignContext *ctx) override;

  std::any visitCallStmt(TTLParser::CallStmtContext *ctx) override;

  std::any visitForLoop(TTLParser::ForLoopContext *ctx) override;

  std::any visitIfStmt(TTLParser::IfStmtContext *ctx) override;

  std::any visitReturnStmt(TTLParser::ReturnStmtContext *ctx) override;

  std::any visitCompoundStmt(TTLParser::CompoundStmtContext *ctx) override;

  std::any visitDim(TTLParser::DimContext *ctx) override;

  std::any visitCall(TTLParser::CallContext *ctx) override;

  std::any visitMatrixMul(TTLParser::MatrixMulContext *ctx) override;

  std::any visitMInit(TTLParser::MInitContext *ctx) override;

  std::any visitMultiplication(TTLParser::MultiplicationContext *ctx) override;

  std::any visitAddition(TTLParser::AdditionContext *ctx) override;

  std::any visitOr(TTLParser::OrContext *ctx) override;

  std::any visitBooleanNot(TTLParser::BooleanNotContext *ctx) override;

  std::any visitUnaryMinus(TTLParser::UnaryMinusContext *ctx) override;

  std::any visitIdRef(TTLParser::IdRefContext *ctx) override;

  std::any visitDimension(TTLParser::DimensionContext *ctx) override;

  std::any visitRange(TTLParser::RangeContext *ctx) override;

  std::any visitParExpr(TTLParser::ParExprContext *ctx) override;

  std::any visitAnd(TTLParser::AndContext *ctx) override;

  std::any visitCompare(TTLParser::CompareContext *ctx) override;

  std::any visitFloatAtom(TTLParser::FloatAtomContext *ctx) override;

  std::any visitSliceMatrix(TTLParser::SliceMatrixContext *ctx) override;

  std::any visitIntAtom(TTLParser::IntAtomContext *ctx) override;

  std::any visitCallExpr(TTLParser::CallExprContext *ctx) override;

private:
  ast::ASTContext *ASTCtx;

  template <typename ExprT, typename... Args>
  ast::ExprPtr createExpr(Args &&...args) {
    return static_cast<ast::ExprPtr>(
        ASTCtx->create<ExprT>(std::forward<Args>(args)...));
  }

  template <typename StmtT, typename... Args>
  ast::StmtPtr createStmt(Args &&...args) {
    return static_cast<ast::StmtPtr>(
        ASTCtx->create<StmtT>(std::forward<Args>(args)...));
  }

  template <typename BinOpCtx>
  ast::ExprPtr createBinaryOp(BinOpCtx *ctx, ast::BinOp OpCode) {
    auto Left = std::any_cast<ast::ExprPtr>(ctx->expr(0)->accept(this));
    auto Right = std::any_cast<ast::ExprPtr>(ctx->expr(1)->accept(this));
    return createExpr<ast::BinExpr>(OpCode, Left, Right);
  }

  template <typename UnOpCtx>
  ast::ExprPtr createUnaryOp(UnOpCtx *ctx, ast::UnOp OpCode) {
    auto Op = std::any_cast<ast::ExprPtr>(ctx->expr()->accept(this));
    return createExpr<ast::UnExpr>(OpCode, Op);
  }
};

} // namespace ttl::parser
