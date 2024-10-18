#pragma once

#include "AST.h"
#include "llvm/ADT/IndexedMap.h"
#include <memory>
#include <vector>

namespace ttl::ast {
class ASTContext {

public:
  IntType *getIntTy() { return getTy<IntType, 0>(); }
  FloatType *getFloatTy() { return getTy<FloatType, 1>(); }
  VoidType *getVoidTy() { return getTy<VoidType, 2>(); }
  RangeType *getRangeTy() { return getTy<RangeType, 3>(); }

  MatrixType *getMatrixTy(TypePtr ElemTy,
                          llvm::ArrayRef<MatrixSize> Sizes) {
    size_t ID = NextMatrixTy++;
    Types[ID] = std::unique_ptr<MatrixType>{
        new MatrixType(ID, ElemTy, Sizes.size(), Sizes)};
    return static_cast<MatrixType *>(Types[ID].get());
  }

  TypePtr getTypeByID(size_t ID) {
    assert(Types.count(ID));
    auto Ty = Types[ID].get();
    assert(Ty->ID == ID);
    return Ty;
  }

  template <ASTNode Node, typename... Args> Node *create(Args &&...args) {
    size_t ID = NextNodeIdx++;
    Nodes[ID] =
        std::unique_ptr<ASTNodeBase>{new Node(std::forward<Args>(args)...)};
    Node *Ptr = static_cast<Node *>(Nodes[ID].get());
    Ptr->id(ID);
    return Ptr;
  }

private:
  template <class Ty, size_t ID> Ty *getTy() {
    static Ty *T = [&]() {
      Types[ID] = std::unique_ptr<Ty>{new Ty()};
      return static_cast<Ty *>(Types[ID].get());
    }();
    return T;
  }

  size_t NextNodeIdx = 0;
  size_t NextMatrixTy = 4;

  std::unordered_map<size_t, std::unique_ptr<ASTNodeBase>> Nodes;
  std::unordered_map<size_t, std::unique_ptr<Type>> Types;
};
} // namespace ttl::ast
