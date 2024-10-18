#include "TTLANTLRParser.h"

namespace ttl::parser {

llvm::Expected<TTLParser::ModuleContext *>
TTLANTLRParser::parseSourceModule(const std::string &FileName) {

  auto *FileStream = new antlr4::ANTLRFileStream{};
  FileStream->loadFromFile(FileName);
  Context->InputStream.reset(FileStream);

  Context->Lexer.reset(new ttl::parser::TTLLexer{Context->InputStream.get()});

  Context->Tokens.reset(new antlr4::CommonTokenStream{Context->Lexer.get()});
  Context->Tokens->fill();

  if (Context->Lexer->getNumberOfSyntaxErrors() > 0) {
    return llvm::make_error<llvm::StringError>(
        "Lexing source file " + FileName + "failed!",
        std::make_error_code(std::errc::invalid_argument));
  }

  Context->Parser.reset(new ttl::parser::TTLParser{Context->Tokens.get()});

  if (Context->Parser->getNumberOfSyntaxErrors() > 0) {
    return llvm::make_error<llvm::StringError>(
        "Parsing source file " + FileName + "failed!",
        std::make_error_code(std::errc::invalid_argument));
  }

  return Context->Parser->module();
}
} // namespace ttl::parser
