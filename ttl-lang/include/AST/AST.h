#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <cassert>
#include <memory>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace ttl::ast {
class ASTContext;

class ASTNodeBase {
  size_t ID = 0;

  void id(size_t I) { ID = I; }

  friend ASTContext;

public:
  size_t id() { return ID; }
};

template <class N> concept ASTNode = std::is_base_of<ASTNodeBase, N>::value;

struct Type {
  size_t ID;

  bool isIntTy() { return ID == 0; }
  bool isFloatTy() { return ID == 1; }
  bool isVoidTy() { return ID == 2; }
  bool isRangeTy() { return ID == 3; }
  bool isMatrixTy() { return ID > 3; }
};

using TypePtr = Type *;

class IntType : public Type {
  IntType() : Type{0} {}

  friend ASTContext;
};

class FloatType : public Type {
  FloatType() : Type{1} {}

  friend ASTContext;
};

class VoidType : public Type {
  VoidType() : Type{2} {}

  friend ASTContext;
};

class RangeType : public Type {
  RangeType() : Type{3} {}

  friend ASTContext;
};

class MatrixSize {
public:
  MatrixSize() : Size{0} {}
  MatrixSize(size_t Size) : Size{Size} {
    assert(Size > 0 && "Invalid static size");
  }

  bool isDynamic() { return Size == 0; }
  bool isStatic() { return Size > 0; }

private:
  size_t Size;
};

class MatrixType : public Type {

  MatrixType(size_t ID, TypePtr ElemTy, size_t Rank,
             llvm::ArrayRef<MatrixSize> Sizes)
      : Type{ID}, ElemTy{ElemTy}, Rank{Rank}, Sizes{Sizes} {}

  TypePtr ElemTy;

  size_t Rank;

  llvm::SmallVector<MatrixSize> Sizes;

  friend ASTContext;

public:
  TypePtr elem() { return ElemTy; }
  size_t rank() { return Rank; }

  MatrixSize size(size_t Dim) {
    assert(Dim < Rank && "Out of bounds");
    return Sizes[Dim];
  }
};

class Expr : public ASTNodeBase {};

using ExprPtr = Expr *;

class IntLiteral : public Expr {
  IntLiteral(int Value) : Value{Value} {}

  int Value;

  friend class ASTContext;

public:
  int value() { return Value; }
};

class FloatLiteral : public Expr {
  FloatLiteral(float Value) : Value{Value} {}

  float Value;

  friend class ASTContext;

public:
  float value() { return Value; }
};

class VarRef;
using VarRefPtr = VarRef *;

class IDRef : public Expr {
  IDRef(const std::string &Name) : Name{Name} {}

  std::string Name;

  VarRefPtr Ref = nullptr;

  friend class ASTContext;

public:
  const std::string &name() { return Name; }

  void ref(VarRefPtr Reference) { Ref = Reference; }

  VarRefPtr ref() { return Ref; }

  bool resolved() { return Ref != nullptr; }
};

class CallExpr : public Expr {

  CallExpr(const std::string &FName, llvm::ArrayRef<ExprPtr> Params)
      : FName{FName}, Params{Params} {}

  std::string FName;

  llvm::SmallVector<ExprPtr> Params;

  friend ASTContext;

public:
  const std::string &func() { return FName; }
  llvm::ArrayRef<ExprPtr> params() { return Params; }
};

class RangeExpr : public Expr {

  RangeExpr(ExprPtr Start, ExprPtr End) : Start{Start}, End{End} {}

  ExprPtr Start;

  ExprPtr End;

  friend ASTContext;

public:
  ExprPtr start() { return Start; }

  ExprPtr end() { return End; }
};

class MatrixInit : public Expr {

  MatrixInit(llvm::ArrayRef<ExprPtr> Elems) : Elems{Elems} {}

  llvm::SmallVector<ExprPtr> Elems;

  friend ASTContext;

public:
  size_t size() { return Elems.size(); }

  llvm::ArrayRef<ExprPtr> elems() { return Elems; }
};

enum BinOp { OR, AND, GT, LT, LE, GE, EQ, NE, ADD, SUB, MUL, DIV, MATMUL, DIM };

class BinExpr : public Expr {

  BinExpr(BinOp Op, ExprPtr Left, ExprPtr Right)
      : Op{Op}, Left{Left}, Right{Right} {}

  BinOp Op;

  ExprPtr Left;

  ExprPtr Right;

  friend ASTContext;

public:
  BinOp op() { return Op; }

  ExprPtr left() { return Left; }

  ExprPtr right() { return Right; }
};

enum UnOp { NOT, MINUS };

class UnExpr : public Expr {

  UnExpr(UnOp Op, ExprPtr Operand) : Op{Op}, Operand{Operand} {}

  UnOp Op;

  ExprPtr Operand;

  friend ASTContext;

public:
  UnOp op() { return Op; }

  ExprPtr operand() { return Operand; }
};

class SliceExpr : public Expr {
  SliceExpr(ExprPtr Matrix, llvm::ArrayRef<ExprPtr> Slices)
      : Matrix{Matrix}, Slices{Slices} {}

  ExprPtr Matrix;

  llvm::SmallVector<ExprPtr> Slices;

  friend ASTContext;

public:
  ExprPtr matrix() { return Matrix; }

  llvm::ArrayRef<ExprPtr> slices() { return Slices; }
};

class Statement : public ASTNodeBase {};

using StmtPtr = Statement *;

class CompoundStmt : public Statement {
  CompoundStmt(llvm::ArrayRef<StmtPtr> Statements) : Statements{Statements} {}

  llvm::SmallVector<StmtPtr> Statements;

  friend ASTContext;

public:
  llvm::ArrayRef<StmtPtr> statements() { return Statements; }
};

class ReturnStmt : public Statement {
  ReturnStmt(ExprPtr RetVal) : RetVal{RetVal} {}

  ExprPtr RetVal;

  friend ASTContext;

public:
  ExprPtr retVal() { return RetVal; }
};

class IfStmt : public Statement {
  IfStmt(ExprPtr Cond, StmtPtr Then, StmtPtr Else)
      : Cond{Cond}, Then{Then}, Else{Else} {}

  ExprPtr Cond;

  StmtPtr Then;

  StmtPtr Else;

  friend ASTContext;

public:
  ExprPtr cond() { return Cond; }
  StmtPtr then() { return Then; }
  StmtPtr elseStmt() { return Else; }
  bool hasElse() { return Else != nullptr; }
};

class ForLoop : public Statement {
  ForLoop(const std::string &Name, ExprPtr IdxRange, ExprPtr Step, StmtPtr Body)
      : Name{Name}, IdxRange{IdxRange}, Step{Step}, Body{Body} {}

  std::string Name;
  ExprPtr IdxRange;
  ExprPtr Step;
  StmtPtr Body;

  friend ASTContext;

public:
  const std::string &name() { return Name; }
  ExprPtr range() { return IdxRange; }
  ExprPtr step() { return Step; } 
  StmtPtr body() { return Body; }

};

class CallStmt : public Statement {
  CallStmt(ExprPtr CallExpr) : CallExpr{CallExpr} {}

  ExprPtr CallExpr;

  friend ASTContext;

public:
  ExprPtr call() { return CallExpr; }
};

class ScalarAssign : public Statement {
  ScalarAssign(const std::string &ID, ExprPtr Value) : ID{ID}, Value{Value} {}

  std::string ID;
  ExprPtr Value;

  VarRefPtr Ref = nullptr;

  friend ASTContext;

public:
  const std::string &name() { return ID; }
  ExprPtr value() { return Value; }
  void ref(VarRefPtr IDRef) { Ref = IDRef; }
  VarRefPtr ref() { return Ref; }
  bool resolved() { return Ref != nullptr; }
};

class MatrixAssign : public Statement {
  MatrixAssign(const std::string &ID, llvm::ArrayRef<ExprPtr> Index,
               ExprPtr Value)
      : ID{ID}, Index{Index}, Value{Value} {}

  std::string ID;
  llvm::SmallVector<ExprPtr> Index;
  ExprPtr Value;

  VarRefPtr Ref;

  friend ASTContext;

public:
  const std::string name() { return ID; }
  ExprPtr value() { return Value; }
  void ref(VarRefPtr IDRef) { Ref = IDRef; }
  VarRefPtr ref() { return Ref; }
  bool resolved() { return Ref != nullptr; }
  llvm::ArrayRef<ExprPtr> indices() { return Index; }
};

class VarDef : public Statement {
  VarDef(TypePtr Type, const std::string &ID, ExprPtr Init)
      : Type{Type}, ID{ID}, Init{Init} {}

  TypePtr Type;
  std::string ID;
  ExprPtr Init;

  friend ASTContext;

public:
  TypePtr typ() { return Type; }
  const std::string &name() { return ID; }
  ExprPtr init() { return Init; }
  bool hasInit() { return Init != nullptr; }
};

class FuncParam : public ASTNodeBase {
  FuncParam(TypePtr Ty, const std::string &Name) : Ty{Ty}, Name{Name} {}

  TypePtr Ty;
  std::string Name;

  friend ASTContext;

public:
  TypePtr ty() { return Ty; }
  const std::string &name() { return Name; }
};

using FuncParamPtr = FuncParam *;

class Function : public ASTNodeBase {

  Function(TypePtr RetTy, const std::string &FuncName,
           llvm::ArrayRef<FuncParamPtr> Params, llvm::ArrayRef<StmtPtr> Body)
      : RetTy{RetTy}, FuncName{FuncName}, Params{Params}, Body{Body} {}

  TypePtr RetTy;
  std::string FuncName;
  llvm::SmallVector<FuncParamPtr> Params;
  llvm::SmallVector<StmtPtr> Body;

  friend ASTContext;

public:
  TypePtr returnType() { return RetTy; }
  const std::string &name() { return FuncName; }
  llvm::ArrayRef<FuncParamPtr> params() { return Params; }
  llvm::ArrayRef<StmtPtr> body() { return Body; }
};

class Module : public ASTNodeBase {
  Module(llvm::ArrayRef<Function *> Funcs) : Funcs{Funcs} {}

  llvm::SmallVector<Function *> Funcs;

  friend ASTContext;

public:
  llvm::ArrayRef<Function *> funcs() { return Funcs; }
};

class VarRef : public ASTNodeBase {
  VarRef(VarDef *Var) : Value{Var} {}
  VarRef(FuncParam *Param) : Value{Param} {}

  std::variant<VarDef *, FuncParam *> Value;

  friend ASTContext;

public:
  bool isVariable() { return std::holds_alternative<VarDef *>(Value); }
  bool isParam() { return std::holds_alternative<FuncParam *>(Value); }

  VarDef *variable() {
    assert(isVariable());
    return std::get<VarDef *>(Value);
  }

  FuncParam *param() {
    assert(isParam());
    return std::get<FuncParam *>(Value);
  }

  ASTNodeBase *get() {
    if (isParam())
      return std::get<FuncParam *>(Value);
    return std::get<VarDef *>(Value);
  }

  bool resolved() {
    if (isParam())
      return std::get<FuncParam *>(Value) != nullptr;

    return std::get<VarDef *>(Value) != nullptr;
  }
};

} // namespace ttl::ast
