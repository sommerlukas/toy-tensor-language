
#include "Conversion/Passes.h"
#include "Dialect/TTL/TTLDialect.h"
#include "Transform/TTLPasses.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
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
  // Register passes, including TTL conversion passes.
  mlir::registerAllPasses();
  mlir::ttl::registerTTLEliminateInitLoops();
  mlir::ttl::registerTTLToTensor();
  mlir::ttl::registerTTLToLinalg();
  mlir::ttl::registerTTLToScalar();

  // Register dialects.
  mlir::DialectRegistry Registry;
  Registry.insert<mlir::ttl::TTLDialect>();
  Registry.insert<mlir::func::FuncDialect>();
  Registry.insert<mlir::scf::SCFDialect>();
  Registry.insert<mlir::index::IndexDialect>();
  Registry.insert<mlir::linalg::LinalgDialect>();
  Registry.insert<mlir::tensor::TensorDialect>();
  Registry.insert<mlir::arith::ArithDialect>();
  Registry.insert<mlir::cf::ControlFlowDialect>();
  Registry.insert<mlir::memref::MemRefDialect>();
  Registry.insert<mlir::LLVM::LLVMDialect>();

  // Register extension interfaces uses by bufferization.
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      Registry);
  mlir::arith::registerBufferDeallocationOpInterfaceExternalModels(Registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(Registry);
  mlir::arith::registerBufferViewFlowOpInterfaceExternalModels(Registry);
  mlir::cf::registerBufferizableOpInterfaceExternalModels(Registry);
  mlir::cf::registerBufferDeallocationOpInterfaceExternalModels(Registry);
  mlir::linalg::registerAllDialectInterfaceImplementations(Registry);
  mlir::memref::registerAllocationOpInterfaceExternalModels(Registry);
  mlir::memref::registerBufferViewFlowOpInterfaceExternalModels(Registry);
  mlir::scf::registerBufferDeallocationOpInterfaceExternalModels(Registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(Registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(Registry);
  mlir::func::registerInlinerExtension(Registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "TTL MLIR optimizer driver\n", Registry));
}
