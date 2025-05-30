#include "ASTBuilder.h"
#include "ASTContext.h"
#include "CodeGen.h"
#include "Dialect/TTL/TTLDialect.h"
#include "Sema.h"
#include "TTLANTLRParser.h"
#include "TTLParserContext.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;
using namespace ttl::parser;
using namespace ttl::ast;

ExitOnError ExitOnErr;

cl::opt<std::string> OutputFilename("o", cl::desc("Specify output filename"),
                                    cl::value_desc("filename"), cl::init("-"));

cl::opt<std::string> InputFilename(cl::Positional, cl::desc("<input file>"),
                                   cl::Required);

cl::opt<bool> PrintDebugInfo("print-debug-info",
                             cl::desc{
                                 "Include debug information in the output"});

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv);

  TTLParserContext ParserContext{};

  TTLANTLRParser Parser{&ParserContext};

  // TODO: Verify availability of input file here or in the library.

  TTLParser::ModuleContext *ParsedModule =
      ExitOnErr(Parser.parseSourceModule(InputFilename.c_str()));

  ASTContext ASTCtx;

  ASTBuilder Builder(&ASTCtx);
  auto *ASTModule =
      std::any_cast<ttl::ast::Module *>(ParsedModule->accept(&Builder));

  ttl::sema::Sema::run(ASTModule);

  mlir::MLIRContext MLIRCtx;

  MLIRCtx.getOrLoadDialect<mlir::ttl::TTLDialect>();
  MLIRCtx.getOrLoadDialect<mlir::scf::SCFDialect>();
  MLIRCtx.getOrLoadDialect<mlir::func::FuncDialect>();

  mlir::ModuleOp MLIRModule =
      ttl::codegen::CodeGen::generate(ASTModule, &MLIRCtx);

  mlir::OpPrintingFlags PrintFlags;
  if (PrintDebugInfo) {
    PrintFlags.enableDebugInfo();
  }

  if (OutputFilename.getNumOccurrences() == 0) {
    MLIRModule.print(llvm::outs(), PrintFlags);
    return 0;
  }

  std::error_code EC;
  auto OutStream = llvm::raw_fd_ostream{OutputFilename.c_str(), EC};
  if (EC) {
    llvm::errs() << "Failed to open output stream for writing\n";
    return -1;
  }

  MLIRModule.print(OutStream, PrintFlags);
}
