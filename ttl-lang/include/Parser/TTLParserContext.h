#pragma once

#include "ANTLRInputStream.h"
#include "TTLParser.h"
#include "TTLLexer.h"

namespace ttl::parser {

class TTLParserContext {
public:
  const antlr4::ANTLRInputStream *inputStream() { return InputStream.get(); }

  const antlr4::CommonTokenStream *tokens() { return Tokens.get(); }

private:
  std::unique_ptr<antlr4::ANTLRInputStream> InputStream;

  std::unique_ptr<ttl::parser::TTLLexer> Lexer;

  std::unique_ptr<antlr4::CommonTokenStream> Tokens;

  std::unique_ptr<ttl::parser::TTLParser> Parser;

  friend class TTLANTLRParser;
};

} // namespace ttl::parser
