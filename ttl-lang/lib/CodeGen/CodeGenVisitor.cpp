#include "CodeGenVisitor.h"
#include "Dialect/TTL/TTLAttributes.h"
#include "Dialect/TTL/TTLDialect.h"
#include "Dialect/TTL/TTLOps.h"
#include "Dialect/TTL/TTLTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::ttl;
using namespace ::ttl::codegen;
using namespace ::ttl::ast;

namespace {

mlir::Type translateType(TypePtr Ty, mlir::MLIRContext *Ctx) {
  if (Ty->isIntTy()) {
    return mlir::ttl::IntType::get(Ctx);
  }
  if (Ty->isFloatTy()) {
    return mlir::ttl::FloatType::get(Ctx);
  }
  if (Ty->isVoidTy()) {
    return mlir::ttl::VoidType::get(Ctx);
  }
  assert(Ty->isMatrixTy());
  auto MatrixTy = static_cast<::ttl::ast::MatrixType *>(Ty);
  mlir::Type ElemTy = translateType(MatrixTy->elem(), Ctx);
  llvm::SmallVector<int64_t> Sizes;
  for (auto &S : MatrixTy->sizes()) {
    if (S.isDynamic()) {
      Sizes.push_back(mlir::ShapedType::kDynamic);
    } else {
      Sizes.push_back(S.val());
    }
  }
  return mlir::ttl::TensorType::get(Ctx, ElemTy, Sizes);
}
} // namespace

void CodeGenVisitor::visit(ast::Module *Node) {
  Builder.setInsertionPointToStart(MLIRModule.getBody());
  for (auto *Func : Node->funcs()) {
    Func->accept(this);
    Builder.setInsertionPointToEnd(MLIRModule.getBody());
  }
}

void CodeGenVisitor::visit(ast::Function *Node) {
  llvm::SmallVector<mlir::Type> ParamTys;
  for (auto P : Node->params()) {
    ParamTys.push_back(translateType(P->ty(), Ctx));
  }
  auto FuncTy =
      FunctionType::get(Ctx, ParamTys, translateType(Node->returnType(), Ctx));
  auto FuncOp =
      Builder.create<func::FuncOp>(translateLoc(Node), Node->name(), FuncTy);
  Block *FuncBody = Builder.createBlock(&FuncOp.getFunctionBody());
  for (auto PT : llvm::zip_equal(Node->params(), ParamTys)) {
    FuncParam *Param = std::get<0>(PT);
    mlir::Type Ty = std::get<1>(PT);
    VarRefPtr Ref = Param->ref();
    assert(Ref);
    LastDefs[Ref] = FuncBody->addArgument(Ty, translateLoc(Param));
  }

  Builder.setInsertionPointToStart(FuncBody);
  for (StmtPtr S : Node->body()) {
    S->accept(this);
    if (S->isReturn()) {
      break;
    }
  }
}

mlir::Value CodeGenVisitor::createIntConstant(::ttl::ast::ASTNodePtrBase *Node,
                                              int32_t Value) {
  return Builder.create<mlir::ttl::IntConstant>(translateLoc(Node), Value);
}

mlir::Value
CodeGenVisitor::createFloatConstant(::ttl::ast::ASTNodePtrBase *Node,
                                    float Value) {
  return Builder.create<mlir::ttl::FloatConstant>(
      translateLoc(Node), mlir::FloatAttr::get(Float32Type::get(Ctx), Value));
}

void CodeGenVisitor::visit(ast::VarDef *Node) {
  mlir::Type ResulTy = translateType(Node->ty(), Ctx);
  Value Init = [&](ast::VarDef *Node) -> Value {
    if (Node->ty()->isMatrixTy()) {
      // Tensor random initialization.
      if (!Node->hasInit()) {
        return Builder.create<TensorRandomInit>(translateLoc(Node), ResulTy);
      }
      Node->init()->accept(this);
      TypePtr InitTy = Node->init()->ty();
      // Tensor scalar initialization.
      if (InitTy->isIntTy() || InitTy->isFloatTy()) {
        Value Scalar = ValueMap[Node->init()];
        return Builder.create<TensorScalarInit>(translateLoc(Node), ResulTy,
                                                Scalar);
      }
      // Tensor range initialization.
      if (InitTy->isRangeTy()) {
        assert(Node->init()->isRangeExpr());
        auto Range = static_cast<RangeExpr *>(Node->init());
        Value Start = ValueMap[Range->start()];
        Value End = ValueMap[Range->end()];
        return Builder.create<TensorRangeInit>(translateLoc(Node), ResulTy,
                                               Start, End);
      }
      return ValueMap[Node->init()];
    }
    if (!Node->hasInit()) {
      if (Node->ty()->isIntTy()) {
        return createIntConstant(Node, 0);
      }
      if (Node->ty()->isFloatTy()) {
        return createFloatConstant(Node, 0.0);
      }
    }
    assert(Node->hasInit());
    Node->init()->accept(this);
    return ValueMap[Node->init()];
  }(Node);

  LastDefs[Node->ref()] = Init;
}

void CodeGenVisitor::visit(ast::MatrixAssign *Node) {
  llvm::SmallVector<mlir::Value> Indices;
  for (ExprPtr Idx : Node->indices()) {
    Idx->accept(this);
    Indices.push_back(ValueMap[Idx]);
  }
  Node->value()->accept(this);
  Value Val = ValueMap[Node->value()];
  assert(Node->resolved());
  Value LastDef = LastDefs[Node->ref()];
  Value NewDef = Builder.create<TensorInsert>(
      translateLoc(Node), LastDef.getType(), LastDef, Val, Indices);
  LastDefs[Node->ref()] = NewDef;
}

void CodeGenVisitor::visit(ast::ScalarAssign *Node) {
  assert(Node->resolved());
  Node->value()->accept(this);
  LastDefs[Node->ref()] = ValueMap[Node->value()];
}

void CodeGenVisitor::visit(ast::CallStmt *Node) { Node->call()->accept(this); }

namespace {

void collectAssignees(StmtPtr S, llvm::SmallPtrSetImpl<VarRefPtr> &Assignees) {
  if (S->isCompound()) {
    llvm::for_each(
        static_cast<CompoundStmt *>(S)->statements(),
        [&](StmtPtr Nested) { collectAssignees(Nested, Assignees); });
  }
  if (S->isIfStmt()) {
    IfStmt *IfS = static_cast<IfStmt *>(S);
    collectAssignees(IfS->then(), Assignees);
    if (IfS->hasElse()) {
      collectAssignees(IfS->elseStmt(), Assignees);
    }
  }
  if (S->isForLoop()) {
    collectAssignees(static_cast<ForLoop *>(S)->body(), Assignees);
  }
  auto Assign = S->assigns();
  if (Assign) {
    Assignees.insert(Assign);
  }
}
} // namespace

void CodeGenVisitor::visit(ast::ForLoop *Node) {
  llvm::SmallPtrSet<VarRefPtr, 10> Assignees;
  collectAssignees(Node, Assignees);
  llvm::SmallVector<VarRefPtr> AssignList(Assignees.begin(), Assignees.end());
  llvm::SmallVector<mlir::Type> AssignTypes;
  llvm::SmallVector<mlir::Value> InArgs;
  for (auto A : AssignList) {
    InArgs.push_back(LastDefs[A]);
    AssignTypes.push_back(translateType(A->ty(), Ctx));
  }
  Value Start;
  Value End;
  if (Node->range()->ty()->isRangeTy()) {
    assert(Node->range()->isRangeExpr());
    auto *Range = static_cast<RangeExpr *>(Node->range());
    Range->start()->accept(this);
    Start = ValueMap[Range->start()];
    Range->end()->accept(this);
    End = ValueMap[Range->end()];
  } else {
    Node->range()->accept(this);
    End = ValueMap[Node->range()];
    Start = Builder.create<mlir::ttl::IntConstant>(translateLoc(Node), 0);
  }
  Node->step()->accept(this);
  Value Step = ValueMap[Node->step()];
  auto ForOp =
      Builder.create<scf::ForOp>(translateLoc(Node), Start, End, Step, InArgs);
  LastDefs[Node->ref()] = ForOp.getInductionVar();
  size_t ArgIdx = 1;
  for (auto A : AssignList) {
    LastDefs[A] = ForOp.getBody()->getArgument(ArgIdx++);
  }
  Builder.setInsertionPointToStart(ForOp.getBody());
  Node->body()->accept(this);
  llvm::SmallVector<mlir::Value> NewDefs;
  for (auto A : AssignList) {
    NewDefs.push_back(LastDefs[A]);
  }
  Builder.create<scf::YieldOp>(translateLoc(Node), NewDefs);
  // After the loop, the last definition for values written inside the loop is
  // the values returned by the loop.
  for (auto AV : llvm::zip(AssignList, ForOp.getResults())) {
    LastDefs[std::get<0>(AV)] = std::get<1>(AV);
  }
  Builder.setInsertionPointAfter(ForOp);
}

void CodeGenVisitor::visit(ast::IfStmt *Node) {
  llvm::SmallPtrSet<VarRefPtr, 10> Assignees;
  collectAssignees(Node, Assignees);
  llvm::SmallVector<VarRefPtr> AssignList(Assignees.begin(), Assignees.end());
  llvm::DenseMap<VarRefPtr, mlir::Value> Before;
  llvm::SmallVector<mlir::Type> AssignTypes;
  for (auto A : AssignList) {
    Before[A] = LastDefs[A];
    AssignTypes.push_back(translateType(A->ty(), Ctx));
  }
  Node->cond()->accept(this);
  Value Cond = ValueMap[Node->cond()];
  auto IfOp =
      Builder.create<mlir::ttl::If>(translateLoc(Node), AssignTypes, Cond);
  auto *ThenBlock = Builder.createBlock(&IfOp.getThenRegion());
  Builder.setInsertionPointToStart(ThenBlock);
  Node->then()->accept(this);
  llvm::SmallVector<Value> ThenDefs;
  for (auto A : AssignList) {
    ThenDefs.push_back(LastDefs[A]);
  }
  Builder.create<mlir::ttl::Yield>(translateLoc(Node), ThenDefs);

  if (Node->hasElse()) {
    // Reset to the values before we entered the 'then' region.
    for (auto A : AssignList) {
      LastDefs[A] = Before[A];
    }
    auto *ElseBlock = Builder.createBlock(&IfOp.getElseRegion());
    Builder.setInsertionPointToStart(ElseBlock);
    Node->elseStmt()->accept(this);
    llvm::SmallVector<Value> ElseDefs;
    for (auto A : AssignList) {
      ElseDefs.push_back(LastDefs[A]);
    }
    Builder.create<mlir::ttl::Yield>(translateLoc(Node), ElseDefs);
  }
  // Outside the 'if', the last definition for values written inside the 'if'
  // is the values returned by the 'If'.
  for (auto AV : llvm::zip(AssignList, IfOp.getResults())) {
    LastDefs[std::get<0>(AV)] = std::get<1>(AV);
  }
  Builder.setInsertionPointAfter(IfOp);
}

void CodeGenVisitor::visit(ast::ReturnStmt *Node) {
  Node->retVal()->accept(this);
  Builder.create<mlir::ttl::Return>(translateLoc(Node),
                                    ValueMap[Node->retVal()]);
}

void CodeGenVisitor::visit(ast::CompoundStmt *Node) {
  llvm::for_each(Node->statements(), [&](StmtPtr S) { S->accept(this); });
}

void CodeGenVisitor::visit(ast::SliceExpr *Node) {
  Node->matrix()->accept(this);
  Value Mat = ValueMap[Node->matrix()];
  llvm::SmallVector<mlir::Value> Offsets;
  llvm::SmallVector<mlir::Value> Sizes;
  for (auto S : Node->slices()) {
    S->accept(this);
    if (S->ty()->isIntTy()) {
      Offsets.push_back(ValueMap[S]);
      Sizes.push_back(createIntConstant(Node, 1));
      continue;
    }
    assert(S->isRangeExpr());
    auto *Range = static_cast<RangeExpr *>(S);
    Value Start = ValueMap[Range->start()];
    Value End = ValueMap[Range->end()];
    Value Size = Builder.create<mlir::ttl::Sub>(translateLoc(Node),
                                                Start.getType(), End, Start);
    Offsets.push_back(Start);
    Sizes.push_back(Size);
  }
  mlir::Type ResultTy = translateType(Node->ty(), Ctx);
  auto Result = Builder.create<mlir::ttl::Slice>(translateLoc(Node), ResultTy,
                                                 Mat, Offsets, Sizes);
  ValueMap[Node] = Result;
}

void CodeGenVisitor::visit(ast::UnExpr *Node) {
  Node->operand()->accept(this);
  Value Opd = ValueMap[Node->operand()];
  Value Result =
      (Node->op() == UnOp::NOT)
          ? Builder.create<mlir::ttl::Not>(translateLoc(Node), Opd).getResult()
          : Builder.create<mlir::ttl::Minus>(translateLoc(Node), Opd)
                .getResult();
  ValueMap[Node] = Result;
}

namespace {
TTLCmpOpcodesAttr translateCmp(BinOp OpCode, mlir::MLIRContext *Ctx) {
  TTLCmpOpcodes Code = [](BinOp OpCode) {
    switch (OpCode) {
    case BinOp::GT:
      return TTLCmpOpcodes::GT;
    case BinOp::GE:
      return TTLCmpOpcodes::GE;
    case BinOp::LT:
      return TTLCmpOpcodes::LT;
    case BinOp::LE:
      return TTLCmpOpcodes::LE;
    case BinOp::EQ:
      return TTLCmpOpcodes::EQ;
    case BinOp::NE:
      return TTLCmpOpcodes::NE;
    default:
      return TTLCmpOpcodes::EQ;
    }
  }(OpCode);
  return TTLCmpOpcodesAttr::get(Ctx, Code);
}
} // namespace

void CodeGenVisitor::visit(ast::BinExpr *Node) {
  Node->left()->accept(this);
  Value Left = ValueMap[Node->left()];
  Node->right()->accept(this);
  Value Right = ValueMap[Node->right()];
  Value Result = [&](BinOp Opcode) -> mlir::Value {
    auto Loc = translateLoc(Node);
    auto Ty = translateType(Node->ty(), Ctx);
    switch (Opcode) {
    case BinOp::DIM:
      return Builder.create<mlir::ttl::Dim>(Loc, Left, Right);
    case BinOp::MATMUL:
      return Builder.create<mlir::ttl::MatMul>(Loc, Ty, Left, Right);
    case BinOp::ADD:
      return Builder.create<mlir::ttl::Add>(Loc, Ty, Left, Right);
    case BinOp::SUB:
      return Builder.create<mlir::ttl::Sub>(Loc, Ty, Left, Right);
    case BinOp::MUL:
      return Builder.create<mlir::ttl::Mul>(Loc, Ty, Left, Right);
    case BinOp::DIV:
      return Builder.create<mlir::ttl::Div>(Loc, Ty, Left, Right);
    case BinOp::AND:
      return Builder.create<mlir::ttl::And>(Loc, Ty, Left, Right);
    case BinOp::OR:
      return Builder.create<mlir::ttl::Or>(Loc, Ty, Left, Right);
    default:
      return Builder.create<mlir::ttl::Compare>(Loc, Left, Right,
                                                translateCmp(Opcode, Ctx));
    }
  }(Node->op());
  ValueMap[Node] = Result;
}

void CodeGenVisitor::visit(ast::MatrixInit *Node) {
  llvm::SmallVector<mlir::Value> Elems;
  for (ExprPtr E : Node->elems()) {
    E->accept(this);
    Elems.push_back(ValueMap[E]);
  }
  Value Result = Builder.create<mlir::ttl::TensorListInit>(
      translateLoc(Node), translateType(Node->ty(), Ctx), Elems);
  ValueMap[Node] = Result;
}

void CodeGenVisitor::visit(ast::RangeExpr *Node) {
  Node->start()->accept(this);
  Node->end()->accept(this);
}

void CodeGenVisitor::visit(ast::CallExpr *Node) {
  llvm::SmallVector<mlir::Value> Args;
  for (auto A : Node->params()) {
    A->accept(this);
    Args.push_back(ValueMap[A]);
  }
  Value Result = Builder
                     .create<func::CallOp>(translateLoc(Node), Node->func(),
                                           translateType(Node->ty(), Ctx), Args)
                     .getResults()
                     .front();
  ValueMap[Node] = Result;
}

void CodeGenVisitor::visit(ast::IDRef *Node) {
  ValueMap[Node] = LastDefs[Node->ref()];
}

void CodeGenVisitor::visit(ast::FloatLiteral *Node) {
  Value Result = Builder.create<mlir::ttl::FloatConstant>(
      translateLoc(Node),
      mlir::FloatAttr::get(Float32Type::get(Ctx), Node->value()));
  ValueMap[Node] = Result;
}

void CodeGenVisitor::visit(ast::IntLiteral *Node) {
  Value Result =
      Builder.create<mlir::ttl::IntConstant>(translateLoc(Node), Node->value());
  ValueMap[Node] = Result;
}
