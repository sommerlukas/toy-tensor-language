#include "TTLANTLRParser.h"
#include "TTLParserContext.h"
#include "llvm/Support/CommandLine.h"
#include "ASTContext.h"
#include "ASTBuilder.h"

using namespace llvm;
using namespace ttl::parser;
using namespace ttl::ast;

ExitOnError ExitOnErr;

cl::opt<std::string> OutputFilename("o", cl::desc("Specify output filename"),
                                    cl::value_desc("filename"), cl::init("-"));

cl::opt<std::string> InputFilename(cl::Positional, cl::desc("<input file>"),
                                   cl::Required);

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv);

  TTLParserContext ParserContext{};

  TTLANTLRParser Parser{&ParserContext};

  // TODO: Verify availability of input file here or in the library.

  TTLParser::ModuleContext *ParsedModule =
      ExitOnErr(Parser.parseSourceModule(InputFilename.c_str()));
  
  ASTContext ASTCtx;

  ASTBuilder Builder(&ASTCtx);
  ParsedModule->accept(&Builder);

 }
