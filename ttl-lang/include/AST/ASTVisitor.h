#pragma once

#include "AST.h"

namespace ttl::ast {

class Module;
class Function;
class FuncParam;
class VarDef;
class MatrixAssign;
class ScalarAssign;
class CallStmt;
class ForLoop;
class IfStmt;
class ReturnStmt;
class CompoundStmt;
class SliceExpr;
class UnExpr;
class BinExpr;
class CondExpr;
class MatrixInit;
class RangeExpr;
class CallExpr;
class IDRef;
class FloatLiteral;
class IntLiteral;
class VarRef;
class ImplicitCast;

class ASTVisitor {
public:
  virtual ~ASTVisitor() = default;

  virtual void visit(ast::Module *){}
  virtual void visit(ast::Function *){}
  virtual void visit(ast::FuncParam*){}
  virtual void visit(ast::VarDef *){}
  virtual void visit(ast::MatrixAssign *){}
  virtual void visit(ast::ScalarAssign *){}
  virtual void visit(ast::CallStmt *){}
  virtual void visit(ast::ForLoop *){}
  virtual void visit(ast::IfStmt *){}
  virtual void visit(ast::ReturnStmt *){}
  virtual void visit(ast::CompoundStmt *){}

  virtual void visit(ast::SliceExpr *){}
  virtual void visit(ast::UnExpr *){}
  virtual void visit(ast::BinExpr *){}
  virtual void visit(ast::CondExpr *){}
  virtual void visit(ast::MatrixInit *){}
  virtual void visit(ast::RangeExpr *){}
  virtual void visit(ast::CallExpr *){}
  virtual void visit(ast::IDRef *){}
  virtual void visit(ast::FloatLiteral *){}
  virtual void visit(ast::IntLiteral *){}

  virtual void visit(ast::VarRef*) {}
  virtual void visit(ast::ImplicitCast*){}
};
} // namespace ttl::ast
