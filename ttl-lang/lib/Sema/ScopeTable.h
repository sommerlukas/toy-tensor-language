#pragma once

#include "AST.h"
#include "ASTContext.h"
#include "llvm/ADT/StringMap.h"

namespace ttl::sema {

class ScopeTable {
private:
  class Table {
  private:
    Table *Parent;

    llvm::StringMap<ast::VarRefPtr> Content;

  public:
    Table(Table *Parent) : Parent{Parent} {}

    bool add(ast::FuncParam *Param) {
      auto [It, Success] = Content.try_emplace(
          Param->name(), Param->context()->create<ast::VarRef>(Param));
      if (Success) {
        Param->ref(Content[Param->name()]);
      }
      return !Success;
    }

    bool add(ast::VarDef *Def) {
      auto [It, Success] = Content.try_emplace(
          Def->name(), Def->context()->create<ast::VarRef>(Def));
      if (Success) {
        Def->ref(Content[Def->name()]);
      }
      return !Success;
    }

    ast::VarRefPtr get(llvm::StringRef Name) {
      if (Content.contains(Name)) {
        return Content.at(Name);
      }

      if (!Parent) {
        return nullptr;
      }

      return Parent->get(Name);
    }

    Table *parent() { return Parent; }
  };

  Table *TopOfStack = nullptr;

public:
  void push() { TopOfStack = new Table(TopOfStack); }

  void pop() {
    Table *OldTop = TopOfStack;
    TopOfStack = OldTop->parent();
    delete OldTop;
  }

  bool add(ast::FuncParam *Param) {
    assert(TopOfStack);
    return TopOfStack->add(Param);
  }

  bool add(ast::VarDef *Def) {
    assert(TopOfStack);
    return TopOfStack->add(Def);
  }

  ast::VarRefPtr get(llvm::StringRef Name) {
    assert(TopOfStack);
    return TopOfStack->get(Name);
  }
};
} // namespace ttl::sema
