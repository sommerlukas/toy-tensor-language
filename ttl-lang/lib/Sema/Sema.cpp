#include "Sema.h"

#include "SemaVisitor.h"

using namespace ttl::sema;
using namespace ttl::ast;

void Sema::run(Module *Mod) {
  SemaVisitor Visitor;
  Mod->accept(&Visitor);
}
