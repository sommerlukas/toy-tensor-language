#include "SemaVisitor.h"
#include <iostream>

using namespace ttl::sema;
using namespace ttl::ast;

namespace {
template <typename... Args>
void reportError(ASTNodePtrBase *Node, Args &&...args) {
  // TODO: Add file, line, col information to AST nodes and print here.
  (void)Node;
  ((std::cerr << " " << std::forward<Args>(args)), ...);
  std::cerr << "\n\n";
  throw std::runtime_error("Compilation error");
}

template <typename Ty> bool checkType(TypePtr ToCheck) { return false; }

template <> bool checkType<IntType>(TypePtr ToCheck) {
  return ToCheck->isIntTy();
}
template <> bool checkType<FloatType>(TypePtr ToCheck) {
  return ToCheck->isFloatTy();
}
template <> bool checkType<RangeType>(TypePtr ToCheck) {
  return ToCheck->isRangeTy();
}
template <> bool checkType<MatrixType>(TypePtr ToCheck) {
  return ToCheck->isMatrixTy();
}

template <typename... AllowedTys> void checkTypeConstraint(ExprPtr E) {
  bool AnyTypeMatches = (checkType<AllowedTys>(E->ty()) || ...);
  if (!AnyTypeMatches) {
    // TODO: Provide more information in error message.
    reportError(E, "Unexpected type");
  }
}

void checkTypeMatch(ASTNodePtrBase *Node, TypePtr Expected, TypePtr Actual) {
  if (*Expected != *Actual) {
    if (Expected->isMatrixTy() && Actual->isMatrixTy()) {
      auto ExMat = static_cast<MatrixType *>(Expected);
      auto AcMat = static_cast<MatrixType *>(Actual);
      if ((*(ExMat->elem()) == *(AcMat->elem())) &&
          ExMat->rank() == AcMat->rank()) {
        bool Match = true;
        for (auto EA : llvm::zip_equal(ExMat->sizes(), AcMat->sizes())) {
          MatrixSize ES = std::get<0>(EA);
          MatrixSize AS = std::get<1>(EA);
          if (ES.isStatic() && AS.isStatic()) {
            Match &= (ES.val() == AS.val());
          }
        }
        if (Match) {
          return;
        }
      }
    }
    reportError(Node, "Expected type", *Expected, "but got", *Actual);
  }
}

TypePtr unifyMatrixTypes(TypePtr LeftTy, TypePtr RightTy) {
  assert(LeftTy->isMatrixTy() && RightTy->isMatrixTy());
  if (*LeftTy == *RightTy) {
    return LeftTy;
  }
  auto LeftMat = static_cast<MatrixType *>(LeftTy);
  auto RightMat = static_cast<MatrixType *>(RightTy);
  llvm::SmallVector<MatrixSize> NewSize;
  for (auto LR : llvm::zip(LeftMat->sizes(), RightMat->sizes())) {
    if (std::get<0>(LR).isStatic()) {
      NewSize.push_back(std::get<0>(LR));
    } else if (std::get<1>(LR).isStatic()) {
      NewSize.push_back(std::get<1>(LR));
    } else {
      NewSize.push_back(MatrixSize());
    }
  }
  return LeftTy->context()->getMatrixTy(LeftMat->elem(), NewSize);
}

} // namespace

void SemaVisitor::visit(Module *Node) {
  for (auto *F : Node->funcs()) {
    F->accept(this);
  }
}

void SemaVisitor::visit(Function *Node) {
  if (Node->returnType()->isRangeTy()) {
    reportError(Node, "Cannot return range from function");
  }
  ReturnAllowed.push(Node->returnType());
  Table.push();
  for (auto *P : Node->params()) {
    if (P->ty()->isRangeTy() || P->ty()->isVoidTy()) {
      reportError(Node, "Cannot declare parameter of type void or range");
    }
    if (Table.add(P)) {
      reportError(Node, "Cannot re-declare parameter with same name",
                  P->name());
    }
  }
  for (auto *S : Node->body()) {
    S->accept(this);
  }
  Table.pop();
  ReturnAllowed.pop();
}

void SemaVisitor::visit(VarDef *Node) {
  if (Node->ty()->isRangeTy() || Node->ty()->isVoidTy()) {
    reportError(Node, "Cannot define variable of type range or void");
  }
  if (Table.add(Node)) {
    reportError(Node, "Redefinition of variable with name", Node->name(),
                "in the same scope");
  }
  if (Node->hasInit()) {
    Node->init()->accept(this);
    // TODO: Handle case of matrix init
    checkTypeMatch(Node, Node->ty(), Node->init()->ty());
  }
}

void SemaVisitor::visit(MatrixAssign *Node) {
  VarRefPtr Ref = Table.get(Node->name());
  if (!Ref) {
    reportError(Node, "Trying to assign to undefined matrix variable with name",
                Node->name());
  }
  if (!Ref->ty()->isMatrixTy()) {
    reportError(Node, "Can only assign elements of a matrix");
  }
  Node->ref(Ref);
  auto *MatrixTy = static_cast<MatrixType *>(Ref->ty());
  for (ExprPtr Idx : Node->indices()) {
    Idx->accept(this);
    checkTypeConstraint<IntType>(Idx);
  }
  Node->value()->accept(this);
  checkTypeMatch(Node, MatrixTy->elem(), Node->value()->ty());
}

void SemaVisitor::visit(ScalarAssign *Node) {
  VarRefPtr Ref = Table.get(Node->name());
  if (!Ref) {
    reportError(Node, "Trying to assign to undefined variable with name",
                Node->name());
  }
  auto LHSTy = Ref->ty();
  if (LHSTy->isVoidTy() || LHSTy->isRangeTy()) {
    reportError(Node, "Cannot assign to void or ranges");
  }
  Node->ref(Ref);
  Node->value()->accept(this);
  checkTypeMatch(Node, LHSTy, Node->value()->ty());
}

void SemaVisitor::visit(CallStmt *Node) {
  Node->call()->accept(this);
  if (!Node->call()->ty()->isVoidTy()) {
    std::cerr << "Warning: Result of function call unused";
  }
}

void SemaVisitor::visit(ForLoop *Node) {
  ReturnAllowed.push(nullptr);
  VarRefPtr ItVar = Table.get(Node->name());
  if (!ItVar) {
    reportError(Node, "Undefined reference to induction variable with name",
                Node->name());
  }
  if (!ItVar->ty()->isIntTy()) {
    reportError(Node, "Loop induction variable must be of type integer");
  }
  Node->range()->accept(this);
  checkTypeConstraint<IntType, RangeType>(Node->range());
  Node->step()->accept(this);
  checkTypeConstraint<IntType>(Node->step());
  Node->body()->accept(this);
  ReturnAllowed.pop();
}

void SemaVisitor::visit(IfStmt *Node) {
  Node->cond()->accept(this);
  checkTypeConstraint<IntType>(Node->cond());

  ReturnAllowed.push(nullptr);
  Node->then()->accept(this);
  if (Node->hasElse()) {
    Node->elseStmt()->accept(this);
  }
  ReturnAllowed.pop();
}

void SemaVisitor::visit(ReturnStmt *Node) {
  if (!ReturnAllowed.top()) {
    // A nullptr on top of the stack indicates that a return is not allowed in
    // this context.
    reportError(Node, "Return is only allowed at function scope");
  }
  Node->retVal()->accept(this);
  checkTypeMatch(Node, ReturnAllowed.top(), Node->retVal()->ty());
}

void SemaVisitor::visit(CompoundStmt *Node) {
  Table.push();
  ReturnAllowed.push(nullptr);
  for (StmtPtr S : Node->statements()) {
    S->accept(this);
  }
  ReturnAllowed.pop();
  Table.pop();
}

void SemaVisitor::visit(SliceExpr *Node) {
  Node->matrix()->accept(this);
  checkTypeConstraint<MatrixType>(Node->matrix());
  bool AllInteger = true;
  llvm::SmallVector<MatrixSize> SliceSize;
  for (ExprPtr Slice : Node->slices()) {
    Slice->accept(this);
    checkTypeConstraint<IntType, RangeType>(Slice);
    AllInteger &= Slice->ty()->isIntTy();
    // TODO: We could do better here if we can detect constant ranges.
    if (AllInteger) {
      SliceSize.push_back(MatrixSize(1));
    } else {
      SliceSize.push_back(MatrixSize());
    }
  }

  auto *MatrixTy = static_cast<MatrixType *>(Node->matrix()->ty());

  if (SliceSize.size() != MatrixTy->rank()) {
    reportError(Node, "Number of slices must match the rank of the matrix");
  }

  if (AllInteger) {
    // If all slices are a single index, we will retrieve a single element from
    // the matrix.
    auto ElemTy = MatrixTy->elem();
    Node->ty(ElemTy);
    return;
  }

  TypePtr NewTy = Node->context()->getMatrixTy(MatrixTy->elem(), SliceSize);
  Node->ty(NewTy);
}

void SemaVisitor::visit(UnExpr *Node) {
  Node->operand()->accept(this);
  switch (Node->op()) {
  case UnOp::NOT:
    checkTypeConstraint<IntType>(Node->operand());
    break;
  case UnOp::MINUS:
    checkTypeConstraint<FloatType, IntType>(Node->operand());
  }
  Node->ty(Node->operand()->ty());
}

namespace {

void checkCompare(BinExpr *Node) {
  checkTypeConstraint<IntType, FloatType>(Node->left());
  checkTypeConstraint<IntType, FloatType>(Node->right());
  checkTypeMatch(Node, Node->left()->ty(), Node->right()->ty());
}

TypePtr checkArithmetic(BinExpr *Node) {
  checkTypeConstraint<IntType, FloatType, MatrixType>(Node->left());
  checkTypeConstraint<IntType, FloatType, MatrixType>(Node->right());
  TypePtr LeftTy = Node->left()->ty();
  TypePtr RightTy = Node->right()->ty();
  // Binary operation with two matrices.
  if (LeftTy->isMatrixTy() && RightTy->isMatrixTy()) {
    checkTypeMatch(Node, LeftTy, RightTy);
    // TODO: Implement support for matrix multiplication
    return unifyMatrixTypes(LeftTy, RightTy);
  }
  // Binary operation with matrix on the left and scalar right.
  if (LeftTy->isMatrixTy()) {
    auto ElemTy = static_cast<MatrixType *>(LeftTy)->elem();
    checkTypeMatch(Node, ElemTy, RightTy);
    return LeftTy;
  }
  // Binary operation with matrix on the right and scalar left.
  if (RightTy->isMatrixTy()) {
    auto ElemTy = static_cast<MatrixType *>(RightTy)->elem();
    checkTypeMatch(Node, ElemTy, LeftTy);
    return RightTy;
  }
  // Scalar binary operation.
  checkTypeMatch(Node, LeftTy, RightTy);
  return LeftTy;
}
} // namespace

void SemaVisitor::visit(BinExpr *Node) {
  Node->left()->accept(this);
  Node->right()->accept(this);
  TypePtr ResultTy = Node->context()->getIntTy();
  switch (Node->op()) {
  case BinOp::GE:
  case BinOp::GT:
  case BinOp::LE:
  case BinOp::LT:
  case BinOp::EQ:
  case BinOp::NE:
    checkCompare(Node);
    break;
  case BinOp::DIM:
    checkTypeConstraint<MatrixType>(Node->left());
    checkTypeConstraint<IntType>(Node->right());
    break;
  default:
    ResultTy = checkArithmetic(Node);
  }
  Node->ty(ResultTy);
}

void SemaVisitor::visit(MatrixInit *Node) {
  for (auto *E : Node->elems()) {
    E->accept(this);
  }
  TypePtr ElemTy = Node->elems()[0]->ty();
  bool AllTypesMatch = llvm::all_of(
      Node->elems(), [&](ExprPtr Elem) { return *(Elem->ty()) == *ElemTy; });
  if (!AllTypesMatch) {
    reportError(Node,
                "All elements of matrix initializer must have the same type");
  }
  TypePtr MatrixTy =
      Node->context()->getMatrixTy(ElemTy, MatrixSize(Node->elems().size()));
  Node->ty(MatrixTy);
}

void SemaVisitor::visit(RangeExpr *Node) {
  Node->start()->accept(this);
  checkTypeConstraint<IntType>(Node->start());
  Node->end()->accept(this);
  checkTypeConstraint<IntType>(Node->end());
  Node->ty(Node->context()->getRangeTy());
}

void SemaVisitor::visit(CallExpr *Node) {
  if (!FuncDecls.contains(Node->func())) {
    reportError(Node, "Calling unknown function", Node->func());
  }

  for (auto *Arg : Node->params()) {
    Arg->accept(this);
  }

  auto *Func = FuncDecls.at(Node->func());
  if (Func->params().size() != Node->params().size()) {
    reportError(Node, "Argument count mismatch, expected",
                Func->params().size(), "but got", Node->params().size());
  }
  for (auto ArgParam : llvm::zip_equal(Node->params(), Func->params())) {
    auto ArgTy = std::get<0>(ArgParam)->ty();
    auto ParamTy = std::get<1>(ArgParam)->ty();
    checkTypeMatch(Node, ParamTy, ArgTy);
  }
  Node->ty(Func->returnType());
}

void SemaVisitor::visit(IDRef *Node) {
  VarRefPtr Ref = Table.get(Node->name());
  if (!Ref) {
    reportError(Node, "Reference to unknown identifier", Node->name());
  }
  Node->ref(Ref);
  Node->ty(Ref->ty());
}

void SemaVisitor::visit(FloatLiteral *Node) {
  Node->ty(Node->context()->getFloatTy());
}

void SemaVisitor::visit(IntLiteral *Node) {
  Node->ty(Node->context()->getIntTy());
}
