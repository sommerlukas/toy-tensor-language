#pragma once

#include "AST.h"
#include "ASTVisitor.h"
#include "ScopeTable.h"
#include "llvm/ADT/StringMap.h"
#include <stack>

namespace ttl::sema {
class SemaVisitor : public ast::ASTVisitor {

  private:
    ScopeTable Table;

    llvm::StringMap<ast::Function*> FuncDecls;
    
    std::stack<ast::TypePtr> ReturnAllowed;

  public:

    void visit(ast::Module*)override;
    void visit(ast::Function*)override;
    void visit(ast::VarDef*)override;
    void visit(ast::MatrixAssign*)override;
    void visit(ast::ScalarAssign*)override;
    void visit(ast::CallStmt*)override;
    void visit(ast::ForLoop*)override;
    void visit(ast::IfStmt*)override;
    void visit(ast::ReturnStmt*)override;
    void visit(ast::CompoundStmt*)override;

    void visit(ast::SliceExpr*)override;
    void visit(ast::UnExpr*)override;
    void visit(ast::BinExpr*)override;
    void visit(ast::MatrixInit*)override;
    void visit(ast::RangeExpr*)override;
    void visit(ast::CallExpr*)override;
    void visit(ast::IDRef*)override;
    void visit(ast::FloatLiteral*)override;
    void visit(ast::IntLiteral*)override;

};
}
