
#include "Dialect/TTL/TTLDialect.h"
#include "Conversion/Passes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::ttl::registerTTLToTensor();
  mlir::ttl::registerTTLToLinalg();
  // TODO: Register more passes here.

  mlir::DialectRegistry Registry;
  Registry.insert<mlir::ttl::TTLDialect>();
  Registry.insert<mlir::func::FuncDialect>();
  Registry.insert<mlir::scf::SCFDialect>();
  Registry.insert<mlir::index::IndexDialect>();
  Registry.insert<mlir::linalg::LinalgDialect>();
  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  // registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "TTL MLIR optimizer driver\n", Registry));
}
