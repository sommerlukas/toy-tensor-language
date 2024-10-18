#pragma once

#include "TTLParserContext.h"
#include "llvm/Support/Error.h"

namespace ttl::parser {

class TTLANTLRParser {

public:
  TTLANTLRParser(TTLParserContext *Ctx) : Context{Ctx} {}

  llvm::Expected<TTLParser::ModuleContext *>
  parseSourceModule(const std::string &FileName);

private:
  TTLParserContext *Context;
};

} // namespace ttl::parser
