#include "ASTBuilder.h"
#include <iostream>

using namespace ttl::ast;
using namespace llvm;

namespace ttl::parser {

std::any ASTBuilder::visitModule(TTLParser::ModuleContext *ctx) {
  SmallVector<Function *> Funcs;
  for (auto *Func : ctx->function()) {
    Funcs.push_back(std::any_cast<Function *>(Func->accept(this)));
  }
  return ASTCtx->create<Module>(Funcs);
}

std::any ASTBuilder::visitFunction(TTLParser::FunctionContext *FuncCtx) {
  std::string FuncName = FuncCtx->ID()->getText();
  auto RetTy = std::any_cast<Type *>(FuncCtx->type()->accept(this));
  SmallVector<FuncParamPtr> ArgDecls;
  for (auto *Arg : FuncCtx->argumentList()->argument()) {
    ArgDecls.push_back(std::any_cast<FuncParamPtr>(Arg->accept(this)));
  }
  SmallVector<StmtPtr> Statements;
  for (auto *Stmt : FuncCtx->statement()) {
    Statements.push_back(std::any_cast<StmtPtr>(Stmt->accept(this)));
  }
  return ASTCtx->create<Function>(RetTy, FuncName, ArgDecls, Statements);
}

std::any ASTBuilder::visitArgument(TTLParser::ArgumentContext *Ctx) {
  auto Ty = std::any_cast<TypePtr>(Ctx->type()->accept(this));
  std::string Name = Ctx->ID()->getText();
  return ASTCtx->create<FuncParam>(Ty, Name);
}

std::any ASTBuilder::visitIntType(TTLParser::IntTypeContext *Ctx) {
  return static_cast<Type *>(ASTCtx->getIntTy());
}

std::any ASTBuilder::visitFloatType(TTLParser::FloatTypeContext *) {
  return static_cast<Type *>(ASTCtx->getFloatTy());
}

std::any ASTBuilder::visitVoidType(TTLParser::VoidTypeContext *ctx) {
  return static_cast<Type *>(ASTCtx->getVoidTy());
}

std::any ASTBuilder::visitMatrixType(TTLParser::MatrixTypeContext *ctx) {
  auto ElemTy = std::any_cast<TypePtr>(ctx->scalarType()->accept(this));
  SmallVector<MatrixSize> Sizes;
  for (auto *Size : ctx->dim()) {
    Sizes.push_back(std::any_cast<MatrixSize>(Size->accept(this)));
  }
  return static_cast<Type *>(ASTCtx->getMatrixTy(ElemTy, Sizes));
}

std::any ASTBuilder::visitDim(TTLParser::DimContext *ctx) {
  if (ctx->getText() == "?")
    return MatrixSize();

  return MatrixSize(std::stoi(ctx->INT()->getText()));
}

std::any ASTBuilder::visitVarDef(TTLParser::VarDefContext *ctx) {
  auto Ty = std::any_cast<TypePtr>(ctx->type()->accept(this));
  std::string Name = ctx->ID()->getText();
  ExprPtr Init = nullptr;
  if (ctx->expr()) {
    Init = std::any_cast<ExprPtr>(ctx->expr()->accept(this));
  }
  return createStmt<VarDef>(Ty, Name, Init);
}

namespace {

struct LHS {
  std::string Name;
  SmallVector<ExprPtr> Indices;
};
} // namespace

std::any ASTBuilder::visitVarAssign(TTLParser::VarAssignContext *ctx) {
  LHS Id = std::any_cast<LHS>(ctx->lhsID()->accept(this));
  auto Value = std::any_cast<ExprPtr>(ctx->expr()->accept(this));
  if (Id.Indices.empty()) {
    return createStmt<ScalarAssign>(Id.Name, Value);
  }
  return createStmt<MatrixAssign>(Id.Name, Id.Indices, Value);
}

std::any
ASTBuilder::visitMatrixElemAssign(TTLParser::MatrixElemAssignContext *ctx) {
  std::string Name = ctx->ID()->getText();
  SmallVector<ExprPtr> Indices;
  for (auto *Idx : ctx->expr()) {
    Indices.push_back(std::any_cast<ExprPtr>(Idx->accept(this)));
  }
  return LHS{Name, Indices};
}

std::any ASTBuilder::visitPrimAssign(TTLParser::PrimAssignContext *ctx) {
  std::string Name = ctx->ID()->getText();
  return LHS{Name, SmallVector<ExprPtr>()};
}

std::any ASTBuilder::visitCallStmt(TTLParser::CallStmtContext *ctx) {
  auto Expr = std::any_cast<ExprPtr>(ctx->callExpr()->accept(this));
  return createStmt<CallStmt>(Expr);
}

std::any ASTBuilder::visitForLoop(TTLParser::ForLoopContext *ctx) {
  std::string Name = ctx->ID()->getText();
  auto IdxRange = std::any_cast<ExprPtr>(ctx->expr(0)->accept(this));
  auto Step = std::any_cast<ExprPtr>(ctx->expr(1)->accept(this));
  auto Body = std::any_cast<StmtPtr>(ctx->statement()->accept(this));
  return createStmt<ForLoop>(Name, IdxRange, Step, Body);
}

std::any ASTBuilder::visitIfStmt(TTLParser::IfStmtContext *ctx) {
  auto Cond = std::any_cast<ExprPtr>(ctx->expr()->accept(this));
  auto Then = std::any_cast<StmtPtr>(ctx->statement(0)->accept(this));
  StmtPtr Else = nullptr;
  if (ctx->statement(1)) {
    Else = std::any_cast<StmtPtr>(ctx->statement(1)->accept(this));
  }
  return createStmt<IfStmt>(Cond, Then, Else);
}

std::any ASTBuilder::visitReturnStmt(TTLParser::ReturnStmtContext *ctx) {
  auto RetVal = std::any_cast<ExprPtr>(ctx->expr()->accept(this));
  return createStmt<ReturnStmt>(RetVal);
}

std::any ASTBuilder::visitCompoundStmt(TTLParser::CompoundStmtContext *ctx) {
  SmallVector<StmtPtr> Body;
  for (auto *Stmt : ctx->statement()) {
    Body.push_back(std::any_cast<StmtPtr>(Stmt->accept(this)));
  }
  return createStmt<CompoundStmt>(Body);
}

std::any ASTBuilder::visitCall(TTLParser::CallContext *ctx) {
  return ctx->callExpr()->accept(this);
}

std::any ASTBuilder::visitMatrixMul(TTLParser::MatrixMulContext *ctx) {
  return createBinaryOp(ctx, BinOp::MATMUL);
}

std::any ASTBuilder::visitMInit(TTLParser::MInitContext *ctx) {
  SmallVector<ExprPtr> Elems;
  for (auto *Elem : ctx->expr()) {
    Elems.push_back(std::any_cast<ExprPtr>(Elem->accept(this)));
  }
  return createExpr<MatrixInit>(Elems);
}

std::any
ASTBuilder::visitMultiplication(TTLParser::MultiplicationContext *ctx) {
  auto Operation = ctx->children.at(1)->getText();
  BinOp OpCode = BinOp::MUL;
  if (Operation == "/") {
    OpCode = BinOp::DIV;
  }
  return createBinaryOp(ctx, OpCode);
}

std::any ASTBuilder::visitAddition(TTLParser::AdditionContext *ctx) {
  auto Operation = ctx->children.at(1)->getText();
  BinOp OpCode = BinOp::ADD;
  if (Operation == "-") {
    OpCode = BinOp::SUB;
  }
  return createBinaryOp(ctx, OpCode);
}

std::any ASTBuilder::visitOr(TTLParser::OrContext *ctx) {
  return createBinaryOp(ctx, BinOp::OR);
}

std::any ASTBuilder::visitBooleanNot(TTLParser::BooleanNotContext *ctx) {
  return createUnaryOp(ctx, UnOp::NOT);
}

std::any ASTBuilder::visitUnaryMinus(TTLParser::UnaryMinusContext *ctx) {
  return createUnaryOp(ctx, UnOp::MINUS);
}

std::any ASTBuilder::visitIdRef(TTLParser::IdRefContext *ctx) {
  return createExpr<IDRef>(ctx->ID()->getText());
}

std::any ASTBuilder::visitDimension(TTLParser::DimensionContext *ctx) {
  return createBinaryOp(ctx, BinOp::DIM);
}

std::any ASTBuilder::visitRange(TTLParser::RangeContext *ctx) {
  auto Start = std::any_cast<ExprPtr>(ctx->expr(0)->accept(this));
  auto End = std::any_cast<ExprPtr>(ctx->expr(1)->accept(this));
  return createExpr<RangeExpr>(Start, End);
}

std::any ASTBuilder::visitParExpr(TTLParser::ParExprContext *ctx) {
  return ctx->expr()->accept(this);
}

std::any ASTBuilder::visitAnd(TTLParser::AndContext *ctx) {
  return createBinaryOp(ctx, BinOp::AND);
}

std::any ASTBuilder::visitCompare(TTLParser::CompareContext *ctx) {
  BinOp Opcode = BinOp::GT;
  std::string Operation = ctx->children.at(1)->getText();
  if (Operation == ">") {
    Opcode = BinOp::GT;
  } else if (Operation == "<") {
    Opcode = BinOp::LT;
  } else if (Operation == "<=") {
    Opcode = BinOp::LE;
  } else if (Operation == ">=") {
    Opcode = BinOp::GE;
  } else if (Operation == "==") {
    Opcode = BinOp::EQ;
  } else if (Operation == "!=") {
    Opcode = BinOp::NE;
  }
  return createBinaryOp(ctx, Opcode);
}

std::any ASTBuilder::visitFloatAtom(TTLParser::FloatAtomContext *ctx) {
  return createExpr<FloatLiteral>(std::stof(ctx->FLOAT()->getText()));
}

std::any ASTBuilder::visitSliceMatrix(TTLParser::SliceMatrixContext *ctx) {
  auto Matrix = std::any_cast<ExprPtr>(ctx->expr(0)->accept(this));
  SmallVector<ExprPtr> Indices;
  for (size_t i = 1; i < ctx->expr().size(); ++i) {
    Indices.push_back(std::any_cast<ExprPtr>(ctx->expr(i)->accept(this)));
  }
  return createExpr<SliceExpr>(Matrix, Indices);
}

std::any ASTBuilder::visitIntAtom(TTLParser::IntAtomContext *ctx) {
  return createExpr<IntLiteral>(std::stoi(ctx->INT()->getText()));
}

std::any ASTBuilder::visitCallExpr(TTLParser::CallExprContext *ctx) {
  std::string FuncName = ctx->ID()->getText();
  SmallVector<ExprPtr> Args;
  for (auto *Arg : ctx->parameterList()->expr()) {
    Args.push_back(std::any_cast<ExprPtr>(Arg->accept(this)));
  }
  return createExpr<CallExpr>(FuncName, Args);
}

} // namespace ttl::parser
